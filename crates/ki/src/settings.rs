//! The AI settings: which service and model, whether Select Subject uses the local model.
//! A small JSON file next to the models folder (`ki.json`); the key is in [`crate::keys`].

use serde_json::{Value, json};

#[derive(Clone, Debug, PartialEq)]
pub struct Settings {
    /// The online service (OpenAI/Gemini compatible), e.g. `https://api.laozhang.ai`.
    pub base: String,
    /// The model Generative Fill uses unless told otherwise.
    pub model: String,
    /// Select Subject and Remove Background use the local AI model once it is downloaded.
    pub subject_ai: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { base: crate::online::DEFAULT_BASE.into(), model: crate::online::DEFAULT_MODEL.into(), subject_ai: true }
    }
}

fn file() -> std::path::PathBuf {
    let dir = crate::models::dir();
    if std::env::var_os("PHOTOCRAFT_KI_DIR").is_some_and(|d| !d.is_empty()) {
        return dir.join("ki.json");
    }
    dir.parent().map(|p| p.join("ki.json")).unwrap_or_else(|| dir.join("ki.json"))
}

impl Settings {
    pub fn load() -> Settings {
        let d = Settings::default();
        let Some(v) = std::fs::read_to_string(file()).ok().and_then(|t| serde_json::from_str::<Value>(&t).ok()) else { return d };
        let s = |k: &str, def: &str| v.get(k).and_then(Value::as_str).filter(|s| !s.trim().is_empty()).unwrap_or(def).trim().to_string();
        Settings {
            base: s("baseUrl", &d.base),
            model: s("model", &d.model),
            subject_ai: v.get("subjectAi").and_then(Value::as_bool).unwrap_or(d.subject_ai),
        }
    }

    pub fn save(&self) -> crate::Result<()> {
        let f = file();
        if let Some(d) = f.parent() {
            std::fs::create_dir_all(d).map_err(|e| crate::Error::Other(e.to_string()))?;
        }
        let v = json!({"baseUrl": self.base, "model": self.model, "subjectAi": self.subject_ai});
        std::fs::write(&f, serde_json::to_string_pretty(&v).unwrap_or_default()).map_err(|e| crate::Error::Other(format!("{}: {e}", f.display())))
    }
}
