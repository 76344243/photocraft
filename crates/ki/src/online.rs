//! Generative fill and image edits through an online image service.
//!
//! The default service is LaoZhang (`https://api.laozhang.ai`), which offers Google's, OpenAI's
//! and ByteDance's image models behind one key. Three protocols cover them:
//!
//! - [`Protocol::Gemini`]: Nano Banana, `POST /v1beta/models/{model}:generateContent` with the
//!   images inline; size and aspect ratio through `generationConfig.imageConfig`.
//! - [`Protocol::OpenAiEdit`]: GPT Image, `POST /v1/images/edits`, multipart with `image[]`.
//! - [`Protocol::Seedream`]: `POST /v1/images/generations`, JSON with the images as data URLs.
//!
//! None of the per-call routes takes a mask reliably, so the area to change is shown to the model
//! as a second copy of the picture with that area tinted magenta; PhotoCraft then keeps only the
//! selected area of the result (on a new layer with the selection as its mask), so whatever the
//! model changes elsewhere never reaches the document.

use std::io::Read;
use std::time::Duration;

use base64::Engine;
use serde_json::{Value, json};

use crate::{Error, Progress, Result, img};

pub const DEFAULT_BASE: &str = "https://api.laozhang.ai";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Protocol {
    Gemini,
    OpenAiEdit,
    Seedream,
}

/// One online model.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OnlineModel {
    pub id: &'static str,
    pub label: &'static str,
    pub protocol: Protocol,
    /// List price per image in US dollars (LaoZhang, autumn 2026).
    pub price_usd: f32,
    /// Largest output, long side, in pixels.
    pub max_side: u32,
    pub note: &'static str,
}

pub const MODELS: &[OnlineModel] = &[
    OnlineModel { id: "seedream-5-0-flash-260915", label: "Seedream 5.0 Flash", protocol: Protocol::Seedream, price_usd: 0.018, max_side: 2048, note: "am günstigsten" },
    OnlineModel { id: "gemini-2.5-flash-image", label: "Nano Banana (Flash)", protocol: Protocol::Gemini, price_usd: 0.02, max_side: 1024, note: "günstig; Google stellt es demnächst ein" },
    OnlineModel { id: "gemini-3.1-flash-lite-image", label: "Nano Banana 2 Lite", protocol: Protocol::Gemini, price_usd: 0.025, max_side: 1024, note: "günstig" },
    OnlineModel { id: "gpt-image-2-vip", label: "GPT Image 2", protocol: Protocol::OpenAiEdit, price_usd: 0.03, max_side: 2048, note: "" },
    OnlineModel { id: "gpt-image-2.5-sunburst-vip", label: "GPT Image 2.5", protocol: Protocol::OpenAiEdit, price_usd: 0.03, max_side: 2048, note: "" },
    OnlineModel { id: "seedream-5-0-260128", label: "Seedream 5.0", protocol: Protocol::Seedream, price_usd: 0.035, max_side: 3072, note: "" },
    OnlineModel { id: "gemini-nano-banana-2.1", label: "Nano Banana 2.1", protocol: Protocol::Gemini, price_usd: 0.045, max_side: 4096, note: "empfohlen" },
    OnlineModel { id: "gemini-3.1-flash-image", label: "Nano Banana 2", protocol: Protocol::Gemini, price_usd: 0.055, max_side: 4096, note: "" },
    OnlineModel { id: "gemini-3-pro-image", label: "Nano Banana Pro", protocol: Protocol::Gemini, price_usd: 0.09, max_side: 4096, note: "beste Qualität" },
];

pub const DEFAULT_MODEL: &str = "gemini-nano-banana-2.1";

/// The model with this id, or (an id not in the list, a newer model, say) one of the protocol its
/// name suggests.
pub fn model(id: &str) -> OnlineModel {
    if let Some(m) = MODELS.iter().find(|m| m.id == id) {
        return *m;
    }
    let protocol = if id.starts_with("gpt-image") {
        Protocol::OpenAiEdit
    } else if id.starts_with("seedream") {
        Protocol::Seedream
    } else {
        Protocol::Gemini
    };
    OnlineModel { id: "", label: "", protocol, price_usd: 0.0, max_side: 2048, note: "" }
}

/// Aspect ratios the Gemini image models take, as (w, h).
pub const RATIOS: [(u32, u32); 10] = [(1, 1), (2, 3), (3, 2), (3, 4), (4, 3), (4, 5), (5, 4), (9, 16), (16, 9), (21, 9)];

/// The listed ratio closest to `w / h` (on a log scale).
pub fn nearest_ratio(w: usize, h: usize) -> (u32, u32) {
    let r = (w.max(1) as f64 / h.max(1) as f64).ln();
    RATIOS.into_iter().min_by(|a, b| {
        let da = (r - (a.0 as f64 / a.1 as f64).ln()).abs();
        let db = (r - (b.0 as f64 / b.1 as f64).ln()).abs();
        da.total_cmp(&db)
    }).unwrap_or((1, 1))
}

/// The service and its key.
#[derive(Clone, Debug)]
pub struct Service {
    pub base: String,
    pub key: String,
}

/// One request: what to change, in which picture.
pub struct Request<'a> {
    pub model: &'a str,
    /// What the person asked for, in their words.
    pub prompt: &'a str,
    /// The picture (RGB 0–1), `w × h`.
    pub rgb: &'a [f32],
    pub w: usize,
    pub h: usize,
    /// Where to change it (0–1 per pixel); `None` edits the whole picture.
    pub mask: Option<&'a [f32]>,
}

/// The instruction sent with the person's words.
pub fn instruction(prompt: &str, masked: bool) -> String {
    let wish = prompt.trim();
    if masked {
        let wish = if wish.is_empty() {
            "fill it naturally so it matches its surroundings, removing whatever is there".to_string()
        } else {
            format!("make this change: {wish}")
        };
        format!(
            "You get two images of the same scene. Image 1 is the original. Image 2 is the same picture with one region tinted magenta. \
             Edit image 1 inside that region only — {wish}. Everything outside the region stays exactly as in image 1: same framing, \
             perspective, lighting, colours, grain and detail, so the change blends in seamlessly. Return only the edited image 1 at the \
             same framing, with no magenta, no border and no text."
        )
    } else if wish.is_empty() {
        "Improve this photo subtly. Keep its framing.".to_string()
    } else {
        format!("Edit this image: {wish}. Keep the framing and everything not mentioned unchanged.")
    }
}

/// The picture with the masked area tinted magenta and outlined.
pub fn marked(rgb: &[f32], w: usize, h: usize, mask: &[f32]) -> Vec<f32> {
    let edge = img::dilate(&mask.iter().map(|m| if *m > 0.5 { 1.0 } else { 0.0 }).collect::<Vec<_>>(), w, h, (w.max(h) / 300).max(2));
    let mut out = rgb.to_vec();
    for i in 0..w * h {
        let m = mask[i].clamp(0.0, 1.0);
        let ring = edge[i] > 0.0 && m <= 0.5;
        let t = if ring { 1.0 } else { m * 0.65 };
        let tint = [1.0, 0.0, 1.0];
        for c in 0..3 {
            out[i * 3 + c] += (tint[c] - out[i * 3 + c]) * t;
        }
    }
    out
}

fn agent() -> ureq::Agent {
    let tls = ureq::tls::TlsConfig::builder().root_certs(ureq::tls::RootCerts::PlatformVerifier).build();
    ureq::Agent::config_builder()
        .tls_config(tls)
        .timeout_global(Some(Duration::from_secs(360)))
        .http_status_as_error(false)
        .user_agent(concat!("PhotoCraft-KI/", env!("CARGO_PKG_VERSION")))
        .build()
        .new_agent()
}

fn b64(bytes: &[u8]) -> String {
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

/// Base64 as services send it: maybe with a `data:` prefix, maybe without padding.
pub fn unb64(s: &str) -> Result<Vec<u8>> {
    let s = s.trim();
    let s = s.split_once(";base64,").map_or(s, |(_, d)| d);
    let cfg = base64::engine::GeneralPurposeConfig::new().with_decode_padding_mode(base64::engine::DecodePaddingMode::Indifferent);
    let engine = base64::engine::GeneralPurpose::new(&base64::alphabet::STANDARD, cfg);
    let clean: String = s.chars().filter(|c| !c.is_whitespace()).collect();
    engine.decode(clean.as_bytes()).map_err(|e| Error::Other(format!("Antwort des KI-Dienstes nicht lesbar ({e})")))
}

/// The service's error message, in words.
fn service_error(status: u16, body: &str) -> Error {
    let v: Value = serde_json::from_str(body).unwrap_or(Value::Null);
    let msg = v
        .pointer("/error/message")
        .or_else(|| v.get("message"))
        .or_else(|| v.get("error"))
        .and_then(Value::as_str)
        .map(str::to_string)
        .unwrap_or_else(|| body.chars().take(300).collect());
    let lower = msg.to_lowercase();
    let why = if status == 401 || lower.contains("invalid token") || lower.contains("api key") && lower.contains("invalid") {
        "Der API-Schlüssel wird nicht angenommen. Bitte unter KI › Einstellungen prüfen.".to_string()
    } else if status == 402 || lower.contains("quota") || lower.contains("insufficient") || lower.contains("余额") {
        format!("Das Guthaben beim KI-Dienst reicht nicht ({msg}).")
    } else if status == 429 {
        format!("Der KI-Dienst ist gerade überlastet – bitte gleich noch einmal ({msg}).")
    } else {
        format!("Der KI-Dienst meldet (Fehler {status}): {msg}")
    };
    Error::Other(why)
}

/// Download an image the service returned as a link.
fn fetch(url: &str) -> Result<Vec<u8>> {
    let mut r = agent().get(url).call().map_err(|e| Error::Other(format!("Ergebnisbild nicht ladbar ({e})")))?;
    if !(200..300).contains(&r.status().as_u16()) {
        return Err(Error::Other(format!("Ergebnisbild nicht ladbar (Fehler {})", r.status().as_u16())));
    }
    let mut v = Vec::new();
    r.body_mut().with_config().limit(200 << 20).reader().read_to_end(&mut v).map_err(|e| Error::Other(e.to_string()))?;
    Ok(v)
}

/// The image in an OpenAI-style answer (`data[0].b64_json` or `data[0].url`).
pub fn openai_image(v: &Value) -> Result<Vec<u8>> {
    let first = v.pointer("/data/0").ok_or_else(|| Error::Other(format!("Der KI-Dienst lieferte kein Bild: {}", short(v))))?;
    if let Some(b) = first.get("b64_json").and_then(Value::as_str).filter(|s| !s.is_empty()) {
        return unb64(b);
    }
    if let Some(u) = first.get("url").and_then(Value::as_str) {
        if u.starts_with("data:") {
            return unb64(u);
        }
        return fetch(u);
    }
    Err(Error::Other(format!("Der KI-Dienst lieferte kein Bild: {}", short(v))))
}

/// The image in a Gemini answer (`candidates[].content.parts[].inlineData`); its text when it
/// sent only text (a refusal, say).
pub fn gemini_image(v: &Value) -> Result<Vec<u8>> {
    let mut text = String::new();
    for cand in v.get("candidates").and_then(Value::as_array).into_iter().flatten() {
        for part in cand.pointer("/content/parts").and_then(Value::as_array).into_iter().flatten() {
            let inline = part.get("inlineData").or_else(|| part.get("inline_data"));
            if let Some(d) = inline.and_then(|i| i.get("data")).and_then(Value::as_str) {
                return unb64(d);
            }
            if let Some(t) = part.get("text").and_then(Value::as_str) {
                text.push_str(t);
            }
        }
        if let Some(reason) = cand.get("finishReason").and_then(Value::as_str).filter(|r| *r != "STOP") {
            text.push_str(&format!(" ({reason})"));
        }
    }
    if let Some(block) = v.pointer("/promptFeedback/blockReason").and_then(Value::as_str) {
        return Err(Error::Other(format!("Der KI-Dienst hat die Anfrage abgelehnt ({block}).")));
    }
    Err(Error::Other(if text.trim().is_empty() {
        format!("Der KI-Dienst lieferte kein Bild: {}", short(v))
    } else {
        format!("Der KI-Dienst lieferte kein Bild, sondern nur Text: {}", text.trim())
    }))
}

fn short(v: &Value) -> String {
    v.to_string().chars().take(300).collect()
}

/// Output size for OpenAI-style models: the picture's aspect, multiples of 16, within the
/// service's limits (655,360 … 8,294,400 pixels, long side ≤ `max`).
pub fn openai_size(w: usize, h: usize, max: u32) -> (u32, u32) {
    let aspect = (w.max(1) as f64 / h.max(1) as f64).clamp(1.0 / 3.0, 3.0);
    let long = (w.max(h) as f64).clamp(1024.0, max as f64);
    let (mut ow, mut oh) = if aspect >= 1.0 { (long, long / aspect) } else { (long * aspect, long) };
    // At least 655,360 pixels.
    let px = ow * oh;
    if px < 655_360.0 {
        let s = (655_360.0 / px).sqrt();
        ow *= s;
        oh *= s;
    }
    let r16 = |v: f64| ((v / 16.0).round() as u32).max(1) * 16;
    (r16(ow), r16(oh))
}

/// Gemini's `imageSize` for a picture this large (None: the model takes no size).
fn gemini_size(model: &OnlineModel, w: usize, h: usize) -> Option<&'static str> {
    if model.id == "gemini-2.5-flash-image" || model.max_side <= 1024 {
        return None;
    }
    let long = w.max(h);
    Some(if long > 2600 && model.max_side >= 4096 {
        "4K"
    } else if long > 1300 {
        "2K"
    } else {
        "1K"
    })
}

struct Multipart {
    boundary: String,
    body: Vec<u8>,
}

impl Multipart {
    fn new() -> Self {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos()).unwrap_or(0);
        Multipart { boundary: format!("PhotoCraftKI{t:x}"), body: Vec::new() }
    }
    fn text(&mut self, name: &str, value: &str) {
        self.body.extend_from_slice(format!("--{}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n", self.boundary).as_bytes());
    }
    fn file(&mut self, name: &str, filename: &str, mime: &str, bytes: &[u8]) {
        self.body.extend_from_slice(
            format!("--{}\r\nContent-Disposition: form-data; name=\"{name}\"; filename=\"{filename}\"\r\nContent-Type: {mime}\r\n\r\n", self.boundary).as_bytes(),
        );
        self.body.extend_from_slice(bytes);
        self.body.extend_from_slice(b"\r\n");
    }
    fn finish(mut self) -> (String, Vec<u8>) {
        self.body.extend_from_slice(format!("--{}--\r\n", self.boundary).as_bytes());
        (format!("multipart/form-data; boundary={}", self.boundary), self.body)
    }
}

/// POST and read the JSON answer, giving up (and returning) when `p` is cancelled; the request
/// itself runs on to its end on its own thread.
fn post(url: String, key: String, content_type: String, body: Vec<u8>, p: &Progress<'_>) -> Result<Value> {
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let r = agent()
            .post(&url)
            .header("Authorization", &format!("Bearer {key}"))
            .header("Content-Type", &content_type)
            .send(&body[..])
            .map_err(|e| Error::Other(format!("Der KI-Dienst ist nicht erreichbar – keine Internetverbindung? ({e})")))
            .and_then(|mut r| {
                let status = r.status().as_u16();
                let text = r.body_mut().with_config().limit(300 << 20).read_to_string().map_err(|e| Error::Other(format!("Antwort abgebrochen ({e})")))?;
                if !(200..300).contains(&status) {
                    return Err(service_error(status, &text));
                }
                serde_json::from_str::<Value>(&text).map_err(|e| Error::Other(format!("Antwort des KI-Dienstes nicht lesbar ({e}): {}", text.chars().take(200).collect::<String>())))
            });
        let _ = tx.send(r);
    });
    let started = std::time::Instant::now();
    loop {
        match rx.recv_timeout(Duration::from_millis(200)) {
            Ok(r) => return r,
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => {
                p.check()?;
                let secs = started.elapsed().as_secs();
                // Most images take 10–60 s; the bar creeps towards 90 % meanwhile.
                p.at(0.15 + 0.75 * (1.0 - (-(secs as f32) / 40.0).exp()), &format!("KI erzeugt das Bild … {secs} s"));
            }
            Err(_) => return Err(Error::Other("Anfrage abgebrochen".into())),
        }
    }
}

/// Send `req` to the service; the result image (any size: the caller fits it to the picture).
pub fn edit(service: &Service, req: &Request<'_>, p: &Progress<'_>) -> Result<(Vec<f32>, usize, usize)> {
    if service.key.trim().is_empty() {
        return Err(Error::Other("Für Generatives Füllen braucht PhotoCraft einen API-Schlüssel (KI › Einstellungen).".into()));
    }
    let model = model(req.model);
    let base = service.base.trim_end_matches('/');
    p.at(0.05, "Bild wird vorbereitet");
    let original = img::png_rgb(req.rgb, req.w, req.h)?;
    let marked = match req.mask {
        Some(m) => Some(img::png_rgb(&marked(req.rgb, req.w, req.h, m), req.w, req.h)?),
        None => None,
    };
    let text = instruction(req.prompt, marked.is_some());
    p.check()?;
    p.at(0.15, "Anfrage wird gesendet");
    let bytes = match model.protocol {
        Protocol::Gemini => {
            let mut parts = vec![json!({"text": text}), json!({"inlineData": {"mimeType": "image/png", "data": b64(&original)}})];
            if let Some(m) = &marked {
                parts.push(json!({"inlineData": {"mimeType": "image/png", "data": b64(m)}}));
            }
            let (rw, rh) = nearest_ratio(req.w, req.h);
            let mut image_config = json!({"aspectRatio": format!("{rw}:{rh}")});
            if let Some(size) = gemini_size(&model, req.w, req.h) {
                image_config["imageSize"] = json!(size);
            }
            let body = json!({
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": {"responseModalities": ["IMAGE"], "imageConfig": image_config},
            });
            let v = post(format!("{base}/v1beta/models/{}:generateContent", req.model), service.key.clone(), "application/json".into(), body.to_string().into_bytes(), p)?;
            gemini_image(&v)?
        }
        Protocol::OpenAiEdit => {
            let (ow, oh) = openai_size(req.w, req.h, model.max_side);
            let mut form = Multipart::new();
            form.text("model", req.model);
            form.text("prompt", &text);
            form.text("size", &format!("{ow}x{oh}"));
            form.text("quality", "high");
            form.text("response_format", "b64_json");
            form.file("image[]", "original.png", "image/png", &original);
            if let Some(m) = &marked {
                form.file("image[]", "marked.png", "image/png", m);
            }
            let (ct, body) = form.finish();
            let v = post(format!("{base}/v1/images/edits"), service.key.clone(), ct, body, p)?;
            openai_image(&v)?
        }
        Protocol::Seedream => {
            let (ow, oh) = openai_size(req.w, req.h, model.max_side);
            let mut images = vec![json!(format!("data:image/png;base64,{}", b64(&original)))];
            if let Some(m) = &marked {
                images.push(json!(format!("data:image/png;base64,{}", b64(m))));
            }
            let body = json!({
                "model": req.model,
                "prompt": text,
                "image": images,
                "size": format!("{ow}x{oh}"),
                "response_format": "b64_json",
                "output_format": "png",
                "watermark": false,
            });
            let v = post(format!("{base}/v1/images/generations"), service.key.clone(), "application/json".into(), body.to_string().into_bytes(), p)?;
            openai_image(&v)?
        }
    };
    p.at(0.95, "Ergebnis wird eingesetzt");
    img::decode(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ratios_and_sizes() {
        assert_eq!(nearest_ratio(1000, 1000), (1, 1));
        assert_eq!(nearest_ratio(1920, 1080), (16, 9));
        assert_eq!(nearest_ratio(600, 800), (3, 4));
        for (w, h) in [(300, 200), (4000, 3000), (500, 3000), (1024, 1024)] {
            let (ow, oh) = openai_size(w, h, 2048);
            assert!(ow % 16 == 0 && oh % 16 == 0);
            let px = ow as u64 * oh as u64;
            assert!((655_360..=8_294_400).contains(&px), "{w}×{h} → {ow}×{oh}");
            assert!(ow.max(oh) <= 3840 && (ow.max(oh) as f64 / ow.min(oh) as f64) <= 3.01);
        }
    }

    #[test]
    fn answers_are_read() {
        let png = img::png_rgb(&[0.5; 12], 2, 2).unwrap();
        let enc = b64(&png);
        let trimmed = enc.trim_end_matches('=');
        assert_eq!(openai_image(&json!({"data": [{"b64_json": format!("data:image/png;base64,{trimmed}")}]})).unwrap(), png);
        assert_eq!(gemini_image(&json!({"candidates": [{"content": {"parts": [{"text": "hier"}, {"inlineData": {"mimeType": "image/png", "data": enc}}]}}]})).unwrap(), png);
        let e = gemini_image(&json!({"candidates": [{"content": {"parts": [{"text": "I can't do that"}]}, "finishReason": "SAFETY"}]})).unwrap_err();
        assert!(e.to_string().contains("I can't do that") && e.to_string().contains("SAFETY"), "{e}");
        assert!(openai_image(&json!({"data": []})).is_err());
    }

    #[test]
    fn errors_are_explained() {
        assert!(service_error(401, r#"{"error":{"message":"invalid token"}}"#).to_string().contains("API-Schlüssel"));
        assert!(service_error(500, r#"{"error":{"message":"boom"}}"#).to_string().contains("boom"));
    }

    #[test]
    fn unknown_ids_get_a_protocol() {
        assert_eq!(model("gpt-image-3").protocol, Protocol::OpenAiEdit);
        assert_eq!(model("seedream-6").protocol, Protocol::Seedream);
        assert_eq!(model("gemini-9-image").protocol, Protocol::Gemini);
        assert_eq!(model(DEFAULT_MODEL).label, "Nano Banana 2.1");
    }

    #[test]
    fn marking_tints_only_the_area_and_its_ring() {
        let (w, h) = (40, 40);
        let rgb = vec![0.5f32; w * h * 3];
        let mut mask = vec![0.0f32; w * h];
        for y in 15..25 {
            for x in 15..25 {
                mask[y * w + x] = 1.0;
            }
        }
        let m = marked(&rgb, w, h, &mask);
        let px = |x: usize, y: usize| &m[(y * w + x) * 3..(y * w + x) * 3 + 3];
        assert!(px(20, 20)[0] > 0.7 && px(20, 20)[1] < 0.3);
        assert_eq!(px(2, 2), &[0.5, 0.5, 0.5]);
        assert_eq!(px(14, 20), &[1.0, 0.0, 1.0]);
    }
}
