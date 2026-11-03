//! The local models: where they come from, where they live, downloading them once.
//!
//! Each model is fetched from a pinned Hugging Face revision (so the bytes never change under
//! us), checked against its SHA-256 and kept in PhotoCraft's settings folder
//! (`~/Library/Application Support/PhotoCraft/KI-Modelle` on a Mac). `PHOTOCRAFT_KI_DIR`
//! overrides the folder (tests, a shared models folder).

use std::io::{Read, Write};
use std::path::PathBuf;
use std::time::Duration;

use sha2::{Digest, Sha256};

use crate::{Error, Progress, Result};

/// One local model.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Model {
    /// BiRefNet lite (MIT): the subject's matte, at 512 × 512 (the 1024 export needs about 6 GB
    /// while it runs; this one under 2 GB, with the same quality after the edge refinement).
    Matte,
    /// LaMa (Apache-2.0): fills a hole from its surroundings, 512 × 512.
    Inpaint,
    /// Real-ESRGAN x2plus (BSD-3-Clause).
    Upscale2,
    /// Real-ESRGAN x4plus (BSD-3-Clause).
    Upscale4,
}

pub struct Spec {
    pub file: &'static str,
    pub url: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
    /// What the model does, for the person.
    pub label: &'static str,
    pub source: &'static str,
    pub license: &'static str,
}

impl Model {
    pub const ALL: [Model; 4] = [Model::Matte, Model::Inpaint, Model::Upscale2, Model::Upscale4];

    pub fn spec(self) -> &'static Spec {
        match self {
            Model::Matte => &Spec {
                file: "birefnet-lite-512.onnx",
                url: "https://huggingface.co/studioludens/birefnet-lite-512/resolve/962e67969ec6acec9c19fab76acee4f5e22e990c/onnx/model.onnx",
                sha256: "1cb0fb360dadd15af77c639085d77a9df67db0c64315560c3de005f676345ac2",
                bytes: 191_877_254,
                label: "Freistellen (Motiv, Hintergrund)",
                source: "BiRefNet lite 512 – studioludens/birefnet-lite-512",
                license: "MIT",
            },
            Model::Inpaint => &Spec {
                file: "lama.onnx",
                url: "https://huggingface.co/Carve/LaMa-ONNX/resolve/c3c0c9e468934d62e79c329e35d82dd09ff8c444/lama_fp32.onnx",
                sha256: "1faef5301d78db7dda502fe59966957ec4b79dd64e16f03ed96913c7a4eb68d6",
                bytes: 208_044_816,
                label: "Objekte entfernen",
                source: "LaMa – Carve/LaMa-ONNX",
                license: "Apache-2.0",
            },
            Model::Upscale2 => &Spec {
                file: "real-esrgan-x2.onnx",
                url: "https://huggingface.co/SceneWorks/real-esrgan-onnx/resolve/09f741bac80a246b407da3ee902bf5f3291b602f/real_esrgan_x2.onnx",
                sha256: "7115ba92e8a1bfa63d68558ef006ef3d91273a068d321b1439f8bb1c9179002c",
                bytes: 67_073_434,
                label: "Hochskalieren ×2",
                source: "Real-ESRGAN x2plus – SceneWorks/real-esrgan-onnx",
                license: "BSD-3-Clause",
            },
            Model::Upscale4 => &Spec {
                file: "real-esrgan-x4.onnx",
                url: "https://huggingface.co/SceneWorks/real-esrgan-onnx/resolve/09f741bac80a246b407da3ee902bf5f3291b602f/real_esrgan_x4.onnx",
                sha256: "5c586662929cbc686c1a5c38d9c060dbdb4ea5863a1f7672b8c0761e6b89c033",
                bytes: 67_051_616,
                label: "Hochskalieren ×4",
                source: "Real-ESRGAN x4plus – SceneWorks/real-esrgan-onnx",
                license: "BSD-3-Clause",
            },
        }
    }

    /// Stable id for commands and settings.
    pub fn id(self) -> &'static str {
        match self {
            Model::Matte => "matte",
            Model::Inpaint => "inpaint",
            Model::Upscale2 => "upscale2",
            Model::Upscale4 => "upscale4",
        }
    }

    pub fn from_id(id: &str) -> Option<Model> {
        Model::ALL.into_iter().find(|m| m.id() == id)
    }

    pub fn path(self) -> PathBuf {
        dir().join(self.spec().file)
    }

    /// Downloaded and complete (the size is checked; the hash was checked when it arrived).
    pub fn installed(self) -> bool {
        std::fs::metadata(self.path()).is_ok_and(|m| m.len() == self.spec().bytes)
    }

    /// The model file, downloading it first if needed.
    pub fn ensure(self, p: &Progress<'_>) -> Result<PathBuf> {
        if self.installed() {
            return Ok(self.path());
        }
        download(self, p)?;
        Ok(self.path())
    }

    /// Delete the downloaded file (it is fetched again on next use).
    pub fn remove(self) -> std::io::Result<()> {
        crate::run::forget(self);
        match std::fs::remove_file(self.path()) {
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            r => r,
        }
    }
}

/// Where the models live.
pub fn dir() -> PathBuf {
    if let Some(d) = std::env::var_os("PHOTOCRAFT_KI_DIR").filter(|d| !d.is_empty()) {
        return PathBuf::from(d);
    }
    let home = std::env::var_os("HOME").map(PathBuf::from).unwrap_or_else(std::env::temp_dir);
    if cfg!(target_os = "macos") {
        home.join("Library/Application Support/PhotoCraft/KI-Modelle")
    } else if cfg!(target_os = "windows") {
        std::env::var_os("APPDATA").map(PathBuf::from).unwrap_or(home).join("PhotoCraft").join("KI-Modelle")
    } else {
        std::env::var_os("XDG_DATA_HOME").map(PathBuf::from).unwrap_or_else(|| home.join(".local/share")).join("photocraft/ki-modelle")
    }
}

fn agent() -> ureq::Agent {
    let tls = ureq::tls::TlsConfig::builder().root_certs(ureq::tls::RootCerts::PlatformVerifier).build();
    ureq::Agent::config_builder()
        .tls_config(tls)
        .timeout_connect(Some(Duration::from_secs(30)))
        .timeout_recv_body(Some(Duration::from_secs(120)))
        .http_status_as_error(false)
        .user_agent(concat!("PhotoCraft-KI/", env!("CARGO_PKG_VERSION")))
        .build()
        .new_agent()
}

fn download(model: Model, p: &Progress<'_>) -> Result<()> {
    let spec = model.spec();
    let what = format!("Lade KI-Modell „{}“ ({} MB, nur beim ersten Mal)", spec.label, spec.bytes / 1_000_000);
    p.at(0.0, &what);
    let dir = dir();
    std::fs::create_dir_all(&dir).map_err(|e| Error::Other(format!("Ordner {} nicht anlegbar ({e})", dir.display())))?;
    let part = dir.join(format!("{}.part", spec.file));
    let mut response = agent()
        .get(spec.url)
        .call()
        .map_err(|e| Error::Other(format!("Das KI-Modell lässt sich nicht laden – keine Internetverbindung? ({e})")))?;
    let status = response.status().as_u16();
    if !(200..300).contains(&status) {
        return Err(Error::Other(format!("Das KI-Modell lässt sich nicht laden (Fehler {status})")));
    }
    let mut reader = response.body_mut().with_config().limit(spec.bytes + 1024).reader();
    let mut file = std::fs::File::create(&part).map_err(|e| Error::Other(format!("{}: {e}", part.display())))?;
    let mut hash = Sha256::new();
    let mut buf = vec![0u8; 1 << 16];
    let mut done = 0u64;
    let result = (|| {
        loop {
            p.check()?;
            let n = reader.read(&mut buf).map_err(|e| Error::Other(format!("Download abgebrochen ({e})")))?;
            if n == 0 {
                break;
            }
            file.write_all(&buf[..n]).map_err(|e| Error::Other(format!("{}: {e}", part.display())))?;
            hash.update(&buf[..n]);
            done += n as u64;
            p.at(done as f32 / spec.bytes as f32, &what);
        }
        file.flush().map_err(|e| Error::Other(e.to_string()))?;
        let got: String = hash.finalize().iter().map(|b| format!("{b:02x}")).collect();
        if done != spec.bytes || got != spec.sha256 {
            return Err(Error::Other("Das geladene KI-Modell ist beschädigt (Prüfsumme falsch). Bitte noch einmal versuchen.".into()));
        }
        Ok(())
    })();
    drop(file);
    match result {
        Ok(()) => std::fs::rename(&part, model.path()).map_err(|e| Error::Other(format!("{}: {e}", model.path().display()))),
        Err(e) => {
            let _ = std::fs::remove_file(&part);
            Err(e)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ids_round_trip_and_files_differ() {
        for m in Model::ALL {
            assert_eq!(Model::from_id(m.id()), Some(m));
            assert_eq!(m.spec().sha256.len(), 64);
            assert!(m.spec().url.contains("/resolve/") && !m.spec().url.contains("/main/"), "pinned revision");
        }
        let files: std::collections::HashSet<_> = Model::ALL.iter().map(|m| m.spec().file).collect();
        assert_eq!(files.len(), 4);
    }
}
