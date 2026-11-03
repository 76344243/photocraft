//! ONNX Runtime: one environment, one session per model, kept while the app runs.
//!
//! On a Mac the CoreML execution provider runs what it can on the GPU and Neural Engine (the
//! compiled CoreML model is cached next to the models, so only the first run compiles);
//! anything it can't take, and every other platform, runs on the CPU. `PHOTOCRAFT_KI_CPU=1`
//! forces the CPU (to compare, or should CoreML misbehave).

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use ort::environment::Environment;
use ort::session::Session;
use ort::session::builder::GraphOptimizationLevel;

use crate::models::Model;
use crate::{Error, Progress, Result};

fn environment() -> Result<&'static Environment> {
    static ENV: OnceLock<std::result::Result<Environment, String>> = OnceLock::new();
    ENV.get_or_init(|| ort::init().with_name("PhotoCraft").with_telemetry(false).build().map_err(|e| e.to_string()))
        .as_ref()
        .map_err(|e| Error::Other(format!("KI-Laufzeit nicht verfügbar: {e}")))
}

type Shared = Arc<Mutex<Session>>;

fn sessions() -> &'static Mutex<HashMap<Model, Shared>> {
    static S: OnceLock<Mutex<HashMap<Model, Shared>>> = OnceLock::new();
    S.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Drop the model's session (after deleting its file, or to free memory).
pub fn forget(model: Model) {
    if let Ok(mut s) = sessions().lock() {
        s.remove(&model);
    }
}

/// Free every session's memory.
pub fn forget_all() {
    if let Ok(mut s) = sessions().lock() {
        s.clear();
    }
}

/// Graph optimisation per model. BiRefNet's layout optimisations need much more memory while the
/// session is built, for no measurable gain.
fn level(model: Model) -> GraphOptimizationLevel {
    match model {
        Model::Matte => GraphOptimizationLevel::Level2,
        _ => GraphOptimizationLevel::Level3,
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

fn build(model: Model, path: &std::path::Path, accelerated: bool) -> Result<Session> {
    let env = environment()?;
    let mut b = Session::builder(env)?
        .with_optimization_level(level(model))
        .map_err(|e| Error::Other(e.to_string()))?
        .with_intra_threads(threads())
        .map_err(|e| Error::Other(e.to_string()))?
        .with_memory_pattern(false)
        .map_err(|e| Error::Other(e.to_string()))?;
    #[cfg(target_os = "macos")]
    if accelerated {
        use ort::ep::coreml::{ComputeUnits, ModelFormat};
        let cache = crate::models::dir().join("coreml-cache");
        let _ = std::fs::create_dir_all(&cache);
        let coreml = ort::ep::CoreML::default()
            .with_model_format(ModelFormat::MLProgram)
            .with_compute_units(ComputeUnits::All)
            .with_model_cache_dir(cache.display())
            .build();
        b = b.with_execution_providers([coreml]).map_err(|e| Error::Other(e.to_string()))?;
    }
    #[cfg(not(target_os = "macos"))]
    let _ = accelerated;
    Ok(b.commit_from_file(path)?)
}

fn accelerate() -> bool {
    cfg!(target_os = "macos") && std::env::var_os("PHOTOCRAFT_KI_CPU").is_none_or(|v| v.is_empty() || v == "0")
}

/// The model's session (downloading the model first if needed).
pub fn session(model: Model, p: &Progress<'_>) -> Result<Shared> {
    if let Some(s) = sessions().lock().ok().and_then(|m| m.get(&model).cloned()) {
        return Ok(s);
    }
    let path = model.ensure(p)?;
    p.at(1.0, "KI-Modell wird vorbereitet");
    let session = match build(model, &path, accelerate()) {
        Ok(s) => s,
        // CoreML refused the model: the CPU takes all of it.
        Err(_) if accelerate() => build(model, &path, false)?,
        Err(e) => return Err(e),
    };
    let shared = Arc::new(Mutex::new(session));
    if let Ok(mut m) = sessions().lock() {
        m.insert(model, shared.clone());
    }
    Ok(shared)
}

/// Run `model` on one input tensor, giving its first output (shape, values).
pub fn run1(model: Model, input: (&str, Vec<i64>, Vec<f32>), p: &Progress<'_>) -> Result<(Vec<i64>, Vec<f32>)> {
    run(model, vec![input], p)
}

/// Run `model` on named input tensors, giving its first output (shape, values).
pub fn run(model: Model, inputs: Vec<(&str, Vec<i64>, Vec<f32>)>, p: &Progress<'_>) -> Result<(Vec<i64>, Vec<f32>)> {
    let shared = session(model, p)?;
    p.check()?;
    let mut session = shared.lock().map_err(|_| Error::Other("KI-Modell blockiert".into()))?;
    let mut values = Vec::with_capacity(inputs.len());
    for (name, shape, data) in inputs {
        let t = ort::value::Tensor::from_array((shape, data))?;
        values.push((std::borrow::Cow::from(name.to_string()), ort::session::SessionInputValue::from(t)));
    }
    let outputs = session.run(values)?;
    let (shape, data) = outputs[0].try_extract_tensor::<f32>()?;
    let shape: Vec<i64> = shape.iter().copied().collect();
    let data = data.to_vec();
    if data.iter().any(|v| !v.is_finite()) {
        return Err(Error::Other("Das KI-Modell lieferte ungültige Werte".into()));
    }
    Ok((shape, data))
}
