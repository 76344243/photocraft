//! The online service's API key: in the login keychain on a Mac (service „PhotoCraft KI“),
//! elsewhere in a file only the user can read. Never in PhotoCraft's settings or the history.

use crate::{Error, Result};

#[cfg(target_os = "macos")]
const SERVICE: &str = "PhotoCraft KI";

/// Keychain account for the key of the service at `base` (one key per service).
fn account(base: &str) -> String {
    base.trim_end_matches('/').to_string()
}

#[cfg(target_os = "macos")]
pub fn get(base: &str) -> Option<String> {
    let out = std::process::Command::new("/usr/bin/security")
        .args(["find-generic-password", "-s", SERVICE, "-a", &account(base), "-w"])
        .output()
        .ok()?;
    let key = String::from_utf8(out.stdout).ok()?.trim().to_string();
    (out.status.success() && !key.is_empty()).then_some(key)
}

#[cfg(target_os = "macos")]
pub fn set(base: &str, key: &str) -> Result<()> {
    let out = std::process::Command::new("/usr/bin/security")
        .args(["add-generic-password", "-U", "-s", SERVICE, "-a", &account(base), "-w", key.trim()])
        .output()
        .map_err(|e| Error::Other(e.to_string()))?;
    if out.status.success() { Ok(()) } else { Err(Error::Other(format!("Schlüsselbund: {}", String::from_utf8_lossy(&out.stderr).trim()))) }
}

#[cfg(target_os = "macos")]
pub fn delete(base: &str) {
    let _ = std::process::Command::new("/usr/bin/security").args(["delete-generic-password", "-s", SERVICE, "-a", &account(base)]).output();
}

#[cfg(not(target_os = "macos"))]
fn file() -> std::path::PathBuf {
    crate::models::dir().join("api-keys.json")
}

#[cfg(not(target_os = "macos"))]
fn read_all() -> serde_json::Map<String, serde_json::Value> {
    std::fs::read_to_string(file()).ok().and_then(|t| serde_json::from_str(&t).ok()).unwrap_or_default()
}

#[cfg(not(target_os = "macos"))]
fn write_all(m: &serde_json::Map<String, serde_json::Value>) -> Result<()> {
    let f = file();
    if let Some(d) = f.parent() {
        std::fs::create_dir_all(d).map_err(|e| Error::Other(e.to_string()))?;
    }
    std::fs::write(&f, serde_json::Value::Object(m.clone()).to_string()).map_err(|e| Error::Other(e.to_string()))?;
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let _ = std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o600));
    }
    Ok(())
}

#[cfg(not(target_os = "macos"))]
pub fn get(base: &str) -> Option<String> {
    read_all().get(&account(base)).and_then(|v| v.as_str()).map(str::to_string).filter(|k| !k.is_empty())
}

#[cfg(not(target_os = "macos"))]
pub fn set(base: &str, key: &str) -> Result<()> {
    let mut m = read_all();
    m.insert(account(base), serde_json::Value::String(key.trim().to_string()));
    write_all(&m)
}

#[cfg(not(target_os = "macos"))]
pub fn delete(base: &str) {
    let mut m = read_all();
    if m.remove(&account(base)).is_some() {
        let _ = write_all(&m);
    }
}

/// The key, from the environment (`PHOTOCRAFT_KI_KEY`, for scripts and tests) or as stored.
pub fn find(base: &str) -> Option<String> {
    std::env::var("PHOTOCRAFT_KI_KEY").ok().filter(|k| !k.trim().is_empty()).map(|k| k.trim().to_string()).or_else(|| get(base))
}

/// The key shown shortened (`sk-…a1b2`), to recognise it without revealing it.
pub fn hint(key: &str) -> String {
    let k: Vec<char> = key.chars().collect();
    if k.len() <= 8 {
        return "•".repeat(k.len());
    }
    format!("{}…{}", k[..3].iter().collect::<String>(), k[k.len() - 4..].iter().collect::<String>())
}
