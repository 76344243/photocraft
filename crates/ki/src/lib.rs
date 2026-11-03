//! Fork (76344243/photocraft): PhotoCraft's AI features.
//!
//! - **Local, free, offline** ([`models`], [`run`]): a subject matte ([`matte`], BiRefNet lite,
//!   MIT), object removal ([`inpaint`], LaMa, Apache-2.0) and 2×/4× upscaling ([`upscale`],
//!   Real-ESRGAN, BSD-3-Clause), on ONNX Runtime (CoreML on Apple Silicon). Each model is
//!   downloaded once, on first use, from a pinned Hugging Face revision and checked by SHA-256.
//! - **Online** ([`online`]): generative fill and image edits through an OpenAI/Gemini-compatible
//!   image service (default LaoZhang, `https://api.laozhang.ai`): Nano Banana (Gemini image),
//!   GPT Image and Seedream. The API key lives in the macOS keychain ([`keys`]).
//!
//! Images cross this crate as interleaved RGB `f32` in 0–1 (sRGB-encoded), masks as one `f32`
//! (0–1) per pixel, both row-major.

pub mod img;
pub mod inpaint;
pub mod keys;
pub mod matte;
pub mod models;
pub mod online;
pub mod run;
pub mod settings;
pub mod upscale;

use std::fmt;

/// What went wrong, in words for the person (German where it reaches them).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Error {
    /// The person cancelled.
    Cancelled,
    /// Anything else, with its message.
    Other(String),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::Cancelled => f.write_str("abgebrochen"),
            Error::Other(m) => f.write_str(m),
        }
    }
}

impl std::error::Error for Error {}

impl From<ort::Error> for Error {
    fn from(e: ort::Error) -> Self {
        Error::Other(format!("KI-Modell: {e}"))
    }
}

pub type Result<T> = std::result::Result<T, Error>;

/// Progress (0–1) and what is happening; `cancelled` asks whether to stop.
pub struct Progress<'a> {
    pub report: &'a (dyn Fn(f32, &str) + Sync),
    pub cancelled: &'a (dyn Fn() -> bool + Sync),
}

impl Progress<'_> {
    /// No reporting, never cancelled.
    pub const NONE: Progress<'static> = Progress { report: &|_, _| {}, cancelled: &|| false };

    pub fn check(&self) -> Result<()> {
        if (self.cancelled)() { Err(Error::Cancelled) } else { Ok(()) }
    }

    pub fn at(&self, t: f32, what: &str) {
        (self.report)(t.clamp(0.0, 1.0), what);
    }

    /// The same, mapped into `lo..hi` of the overall progress.
    pub fn part<R>(&self, lo: f32, hi: f32, f: impl FnOnce(&Progress<'_>) -> R) -> R {
        let report = |t: f32, what: &str| (self.report)(lo + (hi - lo) * t.clamp(0.0, 1.0), what);
        f(&Progress { report: &report, cancelled: self.cancelled })
    }
}
