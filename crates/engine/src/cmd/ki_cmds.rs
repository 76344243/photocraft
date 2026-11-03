//! Fork patch (76344243/photocraft): the AI menu.
//!
//! - `ki.generativeFill`: the selection (or the whole picture) changed by an online image model
//!   as the prompt says, on a new layer masked by the selection.
//! - `ki.remove`: the selection filled from its surroundings by LaMa, on a new layer.
//! - `ki.selectSubject` / `ki.removeBackground`: the subject's matte from BiRefNet as the
//!   selection / the layer's mask. Once that model is downloaded, `select.subject` and
//!   `layer.removeBackground` use it too (unless switched off in the AI settings).
//! - `ki.upscale`: the picture 2× or 4× larger with Real-ESRGAN, as a new document.
//! - `ki.settings`, `ki.setKey`, `ki.models`, `ki.downloadModels`, `ki.deleteModels`.
//!
//! The local models run on the user's machine (downloaded once); generative fill sends the area
//! around the selection to the service set in the AI settings. The work runs as a background
//! job with progress and Cancel.

use photocraft_algo::selection::{Region, SelectionMode};
use photocraft_doc::{Document, Layer, LayerContent, LayerId, LayerMask, PixelFormat, SampleType};
use photocraft_geom::Rect;
use photocraft_ki as ki;
/// The AI crate, for the UI's dialogs.
pub use photocraft_ki;
use photocraft_raster::Surface;
use serde_json::{Value, json};

use crate::commands::CommandSpec;
use crate::jobs::JobCtx;
use crate::{EngineError, Result, Session};

pub const FILL: &str = "ki.generativeFill";
pub const REMOVE: &str = "ki.remove";
pub const SUBJECT: &str = "ki.selectSubject";
pub const BACKGROUND: &str = "ki.removeBackground";
pub const UPSCALE2: &str = "ki.upscale2";
pub const UPSCALE4: &str = "ki.upscale4";
pub const SETTINGS: &str = "ki.settings";
pub const SET_KEY: &str = "ki.setKey";

/// Most pixels sent to the online service along the long side.
const UPLOAD_SIDE: usize = 1536;

fn err(e: ki::Error) -> EngineError {
    match e {
        ki::Error::Cancelled => EngineError::Cancelled,
        ki::Error::Other(m) => EngineError::Other(m),
    }
}

/// `f` with the job's progress and Cancel as the AI crate's [`ki::Progress`].
fn with_progress<R>(ctx: &JobCtx, f: impl FnOnce(&ki::Progress<'_>) -> R) -> R {
    let report = |t: f32, what: &str| ctx.progress(t, what);
    let cancelled = || ctx.cancelled();
    f(&ki::Progress { report: &report, cancelled: &cancelled })
}

fn bad(cmd: &str, msg: impl Into<String>) -> EngineError {
    EngineError::BadParams { cmd: cmd.into(), msg: msg.into() }
}

fn has_doc(s: &Session) -> std::result::Result<(), String> {
    s.active().map(|_| ()).ok_or_else(|| "no document open".into())
}

fn has_selection(s: &Session) -> std::result::Result<(), String> {
    let d = s.active().ok_or("no document open")?;
    if d.doc.selection.as_ref().is_none_or(|m| m.content_bounds().intersect(&d.doc.bounds()).is_empty()) {
        return Err("make a selection around what to remove".into());
    }
    Ok(())
}

fn pixel_layer(s: &Session) -> std::result::Result<LayerId, String> {
    let d = s.active().ok_or("no document open")?;
    let l = crate::active_layer_of(s)?;
    if !matches!(l.content, LayerContent::Raster(_)) {
        return Err(format!("the layer is {} {} layer, not a pixel layer", l.content.article(), l.content.kind_name()));
    }
    let locks = d.doc.effective_locks(l.id);
    if locks.pixels || locks.all {
        return Err(format!("Could not complete your request because the layer \"{}\" is locked", l.name));
    }
    Ok(l.id)
}

// ---- Pixels in and out ---------------------------------------------------------------------

/// The visible picture over `rect` as RGB 0–1 (sRGB-encoded; transparency over white).
fn composite_rgb(doc: &Document, rect: Rect) -> Vec<f32> {
    let buf = photocraft_compose::render(doc, rect);
    rgba_to_rgb(buf.px.iter().copied(), doc.depth == SampleType::F32, buf.px.len())
}

fn rgba_to_rgb(px: impl Iterator<Item = [f32; 4]>, linear: bool, n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(n * 3);
    for p in px {
        let a = p[3].clamp(0.0, 1.0);
        for v in &p[..3] {
            let v = if linear { ki::img::encode_srgb(*v) } else { v.clamp(0.0, 1.0) };
            out.push(v * a + (1.0 - a));
        }
    }
    out
}

/// A layer's own pixels over `rect` as RGB (transparency over white).
fn layer_rgb(doc: &Document, surf: &Surface, rect: Rect) -> Vec<f32> {
    let fmt = surf.format();
    let n = fmt.channels();
    let raw = surf.read_region(rect);
    rgba_to_rgb(raw.chunks_exact(n.max(1)).map(|px| photocraft_raster::to_rgba(&fmt, px)), doc.depth == SampleType::F32, raw.len() / n.max(1))
}

/// RGB (and per-pixel opacity) as pixels of `fmt`.
fn encode(fmt: &PixelFormat, rgb: &[f32], alpha: Option<&[f32]>, linear: bool) -> Vec<f32> {
    let n = fmt.channels();
    let mut out = vec![0.0f32; rgb.len() / 3 * n];
    let mut enc = [0.0f32; 8];
    for (i, px) in rgb.chunks_exact(3).enumerate() {
        let d = |v: f32| if linear { ki::img::decode_srgb(v) } else { v.clamp(0.0, 1.0) };
        let a = alpha.map_or(1.0, |a| a[i]);
        photocraft_raster::from_rgba_into(fmt, [d(px[0]), d(px[1]), d(px[2]), a], &mut enc);
        out[i * n..(i + 1) * n].copy_from_slice(&enc[..n]);
    }
    out
}

/// Selection coverage over `rect` (0 outside the selection, or everywhere without one).
fn coverage(doc: &Document, rect: Rect) -> Vec<f32> {
    let n = rect.width() as usize * rect.height() as usize;
    let Some(sel) = doc.selection.as_ref() else { return vec![0.0; n] };
    crate::fill_cmds::read_window_mask(sel, rect, &|| false).unwrap_or_else(|| vec![0.0; n])
}

/// A gray mask surface over `rect` with these coverages (hidden outside `rect`).
fn mask_surface(rect: Rect, values: &[f32]) -> Surface {
    let bytes: Vec<u8> = values.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    let mut s = Surface::from_interleaved(PixelFormat::GRAY8, rect, &bytes);
    s.prune();
    s
}

fn rect_of(r: ki::img::Rect, origin: Rect) -> Rect {
    let (x, y) = (origin.x0 + r.x as i32, origin.y0 + r.y as i32);
    Rect::new(x, y, x + r.w as i32, y + r.h as i32)
}

/// A new pixel layer above `anchor` holding `rgb` over `rect`.
fn add_layer(doc: &mut Document, anchor: Option<LayerId>, name: &str, rect: Rect, rgb: &[f32], alpha: Option<&[f32]>, mask: Option<Surface>) -> Result<LayerId> {
    let fmt = doc.pixel_format();
    let linear = doc.depth == SampleType::F32;
    let mut layer = Layer::raster(doc.next_layer_name(name), fmt);
    let px = encode(&fmt, rgb, alpha, linear);
    if let Some(s) = layer.surface_mut() {
        s.write_region(rect, &px);
        s.prune();
    }
    if let Some(m) = mask {
        layer.mask = Some(crate::commands::mask_at_depth(LayerMask { surface: m, ..LayerMask::hide_all() }, doc.depth)?);
    }
    Ok(crate::layer_place::insert_new(doc, anchor, layer, crate::layer_place::Place::Above))
}

// ---- Generative Fill -------------------------------------------------------------------------

/// The area sent to the model: the selection's box with as much context around it (at least
/// 64 px), widened to the nearest aspect ratio the Gemini models take, inside the canvas.
pub fn fill_window(sel: Rect, canvas: Rect) -> Rect {
    let (bw, bh) = (sel.width().max(1), sel.height().max(1));
    let ctx = (bw.max(bh) / 2).max(64) as i32;
    let r = sel.inflate(ctx).intersect(&canvas);
    let (w, h) = (r.width() as i64, r.height() as i64);
    let (rw, rh) = ki::online::nearest_ratio(w as usize, h as usize);
    // Grow the short side to the ratio (never shrink what holds the selection).
    let up = |a: i64, b: i64| (a + b - 1) / b;
    let (tw, th) = if w * rh as i64 >= h * rw as i64 { (w, up(w * rh as i64, rw as i64)) } else { (up(h * rw as i64, rh as i64), h) };
    let (tw, th) = (tw.min(canvas.width() as i64), th.min(canvas.height() as i64));
    let cx = (r.x0 as i64 + r.x1 as i64) / 2;
    let cy = (r.y0 as i64 + r.y1 as i64) / 2;
    let x0 = (cx - tw / 2).clamp(canvas.x0 as i64, canvas.x1 as i64 - tw);
    let y0 = (cy - th / 2).clamp(canvas.y0 as i64, canvas.y1 as i64 - th);
    Rect::new(x0 as i32, y0 as i32, (x0 + tw) as i32, (y0 + th) as i32)
}

fn generative_fill(s: &mut Session, p: &Value) -> Result<Value> {
    let prompt = p.get("prompt").and_then(Value::as_str).unwrap_or("").trim().to_string();
    let settings = ki::settings::Settings::load();
    let model = p.get("model").and_then(Value::as_str).filter(|m| !m.trim().is_empty()).map_or(settings.model.clone(), |m| m.trim().to_string());
    let key = ki::keys::find(&settings.base).ok_or_else(|| EngineError::Other("Für Generatives Füllen braucht PhotoCraft einen API-Schlüssel: KI › KI-Einstellungen…".into()))?;
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let doc = d.doc.clone();
    let anchor = d.active_layer;
    let canvas = doc.bounds();
    let sel_box = doc.selection.as_ref().map(|m| m.content_bounds().intersect(&canvas)).filter(|r| !r.is_empty());
    let window = sel_box.map_or(canvas, |b| fill_window(b, canvas));
    if window.is_empty() {
        return Err(bad(FILL, "the document is empty"));
    }
    let service = ki::online::Service { base: settings.base.clone(), key };
    let name = if prompt.is_empty() { "Generative Füllung".to_string() } else { prompt.chars().take(60).collect() };
    let label = "Generative Fill";
    let (ww, wh) = (window.width() as usize, window.height() as usize);
    crate::jobs::run(
        s,
        label,
        true,
        move |ctx| {
            ctx.progress(0.0, "Bild wird vorbereitet");
            let rgb = composite_rgb(&doc, window);
            let cover = sel_box.map(|_| coverage(&doc, window));
            ctx.check()?;
            // Sent at most UPLOAD_SIDE px along the long side.
            let scale = (UPLOAD_SIDE as f64 / ww.max(wh) as f64).min(1.0);
            let (uw, uh) = (((ww as f64 * scale).round() as usize).max(1), ((wh as f64 * scale).round() as usize).max(1));
            let up = ki::img::resize(&rgb, ww, wh, 3, uw, uh);
            let up_mask = cover.as_ref().map(|c| ki::img::resize(c, ww, wh, 1, uw, uh));
            let req = ki::online::Request { model: &model, prompt: &prompt, rgb: &up, w: uw, h: uh, mask: up_mask.as_deref() };
            let (out, ow, oh) = with_progress(ctx, |pr| ki::online::edit(&service, &req, pr)).map_err(err)?;
            ctx.check()?;
            let result = ki::img::resize(&out, ow, oh, 3, ww, wh);
            // The mask: the selection, feathered a little so the new content blends in.
            let mask = cover.map(|c| {
                let r = (ww.max(wh) / 200).clamp(1, 12);
                let soft = ki::img::blur(&ki::img::dilate(&c.iter().map(|v| if *v > 0.5 { 1.0 } else { 0.0 }).collect::<Vec<_>>(), ww, wh, r), ww, wh, r);
                soft.iter().zip(&c).map(|(s, c)| s.max(*c)).collect::<Vec<f32>>()
            });
            Ok((result, mask))
        },
        move |s, (result, mask): (Vec<f32>, Option<Vec<f32>>)| {
            let id = s.edit(label, |doc, active| {
                let m = mask.as_ref().map(|m| mask_surface(window, m));
                let nid = add_layer(doc, anchor, &name, window, &result, None, m)?;
                doc.selection = None;
                *active = Some(nid);
                Ok(nid)
            })?;
            Ok(json!({"layer": id.0, "bounds": [window.x0, window.y0, window.width(), window.height()]}))
        },
    )
}

// ---- Remove (LaMa) ------------------------------------------------------------------------

fn remove(s: &mut Session, _p: &Value) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let doc = d.doc.clone();
    let anchor = d.active_layer;
    let canvas = doc.bounds();
    let sel = doc.selection.as_ref().map(|m| m.content_bounds().intersect(&canvas)).filter(|r| !r.is_empty()).ok_or_else(|| bad(REMOVE, "make a selection around what to remove"))?;
    // Enough around the selection for LaMa's context window (twice the hole, ≥ 512 px).
    let reach = (sel.width().max(sel.height()) as i32).max(256) + 32;
    let area = sel.inflate(reach).intersect(&canvas);
    let label = "Remove";
    crate::jobs::run(
        s,
        label,
        true,
        move |ctx| {
            ctx.progress(0.0, "Objekt wird entfernt");
            let (w, h) = (area.width() as usize, area.height() as usize);
            let rgb = composite_rgb(&doc, area);
            let cover = coverage(&doc, area);
            ctx.check()?;
            let patch = with_progress(ctx, |pr| ki::inpaint::remove(&rgb, w, h, &cover, pr)).map_err(err)?;
            Ok(patch)
        },
        move |s, patch: ki::inpaint::Patch| {
            let rect = rect_of(patch.rect, area);
            let id = s.edit(label, |doc, active| {
                let nid = add_layer(doc, anchor, "Entfernt", rect, &patch.rgb, Some(&patch.alpha), None)?;
                doc.selection = None;
                *active = Some(nid);
                Ok(nid)
            })?;
            Ok(json!({"layer": id.0, "bounds": [rect.x0, rect.y0, rect.width(), rect.height()]}))
        },
    )
}

// ---- Subject (BiRefNet) ---------------------------------------------------------------------

/// Whether `select.subject` and `layer.removeBackground` should use the local model: it is
/// downloaded, the AI settings allow it, and the call doesn't ask for the classic algorithm.
pub fn subject_ai_ready(p: &Value) -> bool {
    !p.get("classic").and_then(Value::as_bool).unwrap_or(false) && ki::settings::Settings::load().subject_ai && ki::models::Model::Matte.installed()
}

/// The matte over `area` as a selection region.
/// The edge refinement Remove Background applies after the matte (upstream's, see
/// `cutout_cmds`): the matte is computed at 512 × 512, this snaps its edge to the full-size pixels.
const REFINE: photocraft_algo::matting::RefineParams =
    photocraft_algo::matting::RefineParams { radius: 2.0, smart_radius: true, smooth: 10.0, feather: 0.5, contrast: 10.0, shift_edge: 0.0 };

fn matte_region(doc: &Document, layer: Option<LayerId>, all_layers: bool, area: Rect, refine: bool, ctx: &JobCtx) -> Result<Option<Region>> {
    let (w, h) = (area.width() as usize, area.height() as usize);
    let rgb = match (all_layers, layer.and_then(|id| doc.layer(id)).and_then(Layer::surface)) {
        (false, Some(surf)) => layer_rgb(doc, surf, area),
        _ => composite_rgb(doc, area),
    };
    let matte = with_progress(ctx, |pr| ki::matte::subject(&rgb, w, h, pr)).map_err(err)?;
    // Specks under 2 % are noise from the background.
    let mask: Vec<u8> = matte.iter().map(|v| if *v < 0.02 { 0 } else { (v * 255.0).round() as u8 }).collect();
    if mask.iter().all(|v| *v == 0) {
        return Ok(None);
    }
    // The region's box: just around what was found (the refinement works over the box).
    let found: Vec<f32> = mask.iter().map(|v| f32::from(*v)).collect();
    let b = ki::img::mask_bounds(&found, w, h, 0.0).unwrap_or(ki::img::Rect { x: 0, y: 0, w, h });
    let bbox = rect_of(b, area);
    let mut tight = Vec::with_capacity(b.w * b.h);
    for y in b.y as usize..b.y as usize + b.h {
        tight.extend_from_slice(&mask[y * w + b.x as usize..y * w + b.x as usize + b.w]);
    }
    let region = Region { bbox, mask: tight };
    if !refine {
        return Ok(Some(region));
    }
    ctx.progress(0.95, "Kante wird verfeinert");
    let refined = crate::smartselect_cmds::with_doc_sampler(doc, layer, all_layers, |smp, _| {
        photocraft_algo::matting::refine_mask(smp, &photocraft_algo::matting::region_reader(&region), region.bbox, area, &REFINE)
    });
    Ok(Some(refined.unwrap_or(region)))
}

/// Where the subject is looked for: the layer's pixels within the canvas (a placed photo that
/// doesn't fill the canvas), else the canvas.
fn subject_area(doc: &Document, layer: Option<LayerId>, all_layers: bool) -> Rect {
    let canvas = doc.bounds();
    let Some(surf) = layer.and_then(|id| doc.layer(id)).and_then(Layer::surface).filter(|_| !all_layers) else { return canvas };
    let empty_around = surf.format().alpha && surf.default_pixel().last().is_some_and(|a| *a <= 0.0);
    let b = surf.content_bounds().intersect(&canvas);
    if empty_around && !b.is_empty() { b } else { canvas }
}

/// Select Subject with the local model (also `select.subject` once the model is there).
pub fn select_subject(s: &mut Session, p: &Value) -> Result<Value> {
    let all_layers = p.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(false);
    let m = p.get("mode").and_then(Value::as_str).unwrap_or("replace");
    let mode = SelectionMode::parse(if m == "new" { "replace" } else { m });
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let doc = d.doc.clone();
    let layer = d.active_layer;
    let area = subject_area(&doc, layer, all_layers);
    crate::jobs::run(
        s,
        "Select Subject",
        true,
        move |ctx| matte_region(&doc, layer, all_layers, area, false, ctx),
        move |s, region: Option<Region>| crate::smartselect_cmds::apply_region(s, "Select Subject", region.as_ref(), mode),
    )
}

/// Remove Background with the local model: the matte as the layer's mask.
pub fn remove_background(s: &mut Session, p: &Value) -> Result<Value> {
    let id = match p.get("layer").and_then(Value::as_u64) {
        Some(l) => LayerId(l),
        None => pixel_layer(s).map_err(EngineError::Other)?,
    };
    let all_layers = p.get("sampleAllLayers").and_then(Value::as_bool).unwrap_or(false);
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let doc = d.doc.clone();
    if doc.layer(id).is_none() {
        return Err(EngineError::NoLayer(id));
    }
    let area = subject_area(&doc, Some(id), all_layers);
    let p_refine = p.get("refine").and_then(Value::as_bool).unwrap_or(true);
    crate::jobs::run(
        s,
        "Remove Background",
        true,
        move |ctx| matte_region(&doc, Some(id), all_layers, area, p_refine, ctx),
        move |s, region: Option<Region>| {
            let region = region.ok_or_else(|| EngineError::Other("no subject found".into()))?;
            let b = region.bbox;
            s.edit("Remove Background", |doc, _| {
                crate::extra_cmds::background_to_layer_for_mask(doc, id);
                let mask = crate::commands::mask_at_depth(LayerMask { surface: photocraft_algo::matting::region_surface(&region), ..LayerMask::reveal_all() }, doc.depth)?;
                doc.layer_mut(id).ok_or(EngineError::NoLayer(id))?.mask = Some(mask);
                doc.selection = None;
                Ok(())
            })?;
            Ok(json!({"layer": id.0, "bounds": [b.x0, b.y0, b.width(), b.height()]}))
        },
    )
}

// ---- Upscale (Real-ESRGAN) ----------------------------------------------------------------

fn upscale(s: &mut Session, factor: usize) -> Result<Value> {
    let d = s.active().ok_or(EngineError::NoDocument)?;
    let doc = d.doc.clone();
    let canvas = doc.bounds();
    let (w, h) = (canvas.width() as usize, canvas.height() as usize);
    if (w * factor).checked_mul(h * factor).is_none_or(|n| n > 400_000_000) {
        return Err(EngineError::Other("Das Ergebnis wäre zu groß (über 400 Megapixel)".into()));
    }
    let label = if factor == 2 { "Upscale ×2" } else { "Upscale ×4" };
    crate::jobs::run(
        s,
        label,
        false,
        move |ctx| {
            ctx.progress(0.0, "Bild wird vorbereitet");
            let buf = photocraft_compose::render(&doc, canvas);
            let linear = doc.depth == SampleType::F32;
            let alpha: Vec<f32> = buf.px.iter().map(|p| p[3]).collect();
            let opaque = alpha.iter().all(|a| *a >= 1.0);
            let rgb: Vec<f32> = buf.px.iter().flat_map(|p| (0..3).map(move |c| if linear { ki::img::encode_srgb(p[c]) } else { p[c].clamp(0.0, 1.0) })).collect();
            drop(buf);
            let big = with_progress(ctx, |pr| ki::upscale::upscale(&rgb, w, h, factor, pr)).map_err(err)?;
            let big_alpha = (!opaque).then(|| ki::img::resize(&alpha, w, h, 1, w * factor, h * factor));
            let mut out = Document::new(format!("{} (KI ×{factor})", doc.name), photocraft_geom::Size::new((w * factor) as u32, (h * factor) as u32), doc.mode, doc.depth);
            out.resolution_dpi = doc.resolution_dpi * factor as f32;
            out.icc_profile = doc.icc_profile.clone();
            let fmt = out.pixel_format();
            let mut layer = Layer::raster(if opaque { "Background".to_string() } else { "Layer 1".to_string() }, fmt);
            let px = encode(&fmt, &big, big_alpha.as_deref(), linear);
            if let Some(s) = layer.surface_mut() {
                s.write_region_par(out.bounds(), &px);
                s.prune();
            }
            out.layers.push(layer);
            Ok(out)
        },
        move |s, out: Document| {
            let (w, h) = (out.size.width, out.size.height);
            let i = s.add_document(out, None);
            Ok(json!({"document": i, "width": w, "height": h}))
        },
    )
}

// ---- Settings and models --------------------------------------------------------------------

fn settings_json() -> Value {
    let st = ki::settings::Settings::load();
    let key = ki::keys::find(&st.base);
    json!({
        "baseUrl": st.base,
        "model": st.model,
        "subjectAi": st.subject_ai,
        "keySet": key.is_some(),
        "keyHint": key.as_deref().map(ki::keys::hint),
        "onlineModels": ki::online::MODELS.iter().map(|m| json!({"id": m.id, "label": m.label, "priceUsd": m.price_usd, "note": m.note})).collect::<Vec<_>>(),
        "localModels": models_json(),
        "modelsFolder": ki::models::dir().display().to_string(),
    })
}

fn models_json() -> Vec<Value> {
    ki::models::Model::ALL
        .iter()
        .map(|m| {
            let s = m.spec();
            json!({"id": m.id(), "label": s.label, "installed": m.installed(), "bytes": s.bytes, "source": s.source, "license": s.license})
        })
        .collect()
}

fn settings(_s: &mut Session, p: &Value) -> Result<Value> {
    let mut st = ki::settings::Settings::load();
    let mut changed = false;
    if let Some(b) = p.get("baseUrl").and_then(Value::as_str) {
        let b = b.trim().trim_end_matches('/');
        if !b.is_empty() && !b.starts_with("https://") && !b.starts_with("http://127.0.0.1") && !b.starts_with("http://localhost") {
            return Err(bad(SETTINGS, "baseUrl must start with https://"));
        }
        st.base = if b.is_empty() { ki::online::DEFAULT_BASE.into() } else { b.into() };
        changed = true;
    }
    if let Some(m) = p.get("model").and_then(Value::as_str) {
        st.model = if m.trim().is_empty() { ki::online::DEFAULT_MODEL.into() } else { m.trim().into() };
        changed = true;
    }
    if let Some(v) = p.get("subjectAi").and_then(Value::as_bool) {
        st.subject_ai = v;
        changed = true;
    }
    if changed {
        st.save().map_err(err)?;
    }
    Ok(settings_json())
}

fn set_key(_s: &mut Session, p: &Value) -> Result<Value> {
    let key = p.get("key").and_then(Value::as_str).ok_or_else(|| bad(SET_KEY, "needs `key` (empty removes it)"))?.trim();
    let base = ki::settings::Settings::load().base;
    if key.is_empty() {
        ki::keys::delete(&base);
    } else {
        if key.chars().any(char::is_whitespace) || key.len() < 8 {
            return Err(bad(SET_KEY, "that doesn't look like an API key"));
        }
        ki::keys::set(&base, key).map_err(err)?;
    }
    Ok(settings_json())
}

fn models(_s: &mut Session, _p: &Value) -> Result<Value> {
    Ok(json!({"models": models_json(), "folder": ki::models::dir().display().to_string()}))
}

fn which_models(p: &Value) -> Result<Vec<ki::models::Model>> {
    match p.get("models").and_then(Value::as_array) {
        None => Ok(ki::models::Model::ALL.to_vec()),
        Some(a) => a
            .iter()
            .map(|v| v.as_str().and_then(ki::models::Model::from_id).ok_or_else(|| bad("ki.downloadModels", "models: matte|inpaint|upscale2|upscale4")))
            .collect(),
    }
}

fn download_models(s: &mut Session, p: &Value) -> Result<Value> {
    let list = which_models(p)?;
    crate::jobs::run(
        s,
        "Download AI Models",
        false,
        move |ctx| {
            let n = list.len().max(1) as f32;
            for (i, m) in list.iter().enumerate() {
                let report = |t: f32, what: &str| ctx.progress((i as f32 + t) / n, what);
                let cancelled = || ctx.cancelled();
                m.ensure(&ki::Progress { report: &report, cancelled: &cancelled }).map_err(err)?;
            }
            Ok(())
        },
        |_, ()| Ok(json!({"models": models_json()})),
    )
}

fn delete_models(_s: &mut Session, p: &Value) -> Result<Value> {
    for m in which_models(p)? {
        m.remove().map_err(|e| EngineError::Other(e.to_string()))?;
    }
    Ok(json!({"models": models_json()}))
}

macro_rules! spec {
    ($id:expr, $label:literal, [$($m:literal),*], $params:literal, $en:expr, $run:expr, $journal:expr) => {
        CommandSpec { id: $id, label: $label, menu: &[$($m),*], shortcut: None, params: $params, enabled: $en, run: $run, journal: $journal }
    };
}

/// The AI menu's command specs.
pub fn specs() -> Vec<CommandSpec> {
    vec![
        spec!(
            FILL,
            "Generative Fill…",
            ["AI"],
            r##"{"prompt":str="" (what to put there; empty: fill to match the surroundings),"model":str? (online model id; default: the AI settings')} → {layer,bounds} (sends the area around the selection, or the whole picture without one, to the online service; the result is a new layer masked by the selection)"##,
            has_doc,
            generative_fill,
            true
        ),
        spec!(REMOVE, "Remove Selected Object", ["AI"], r##"{} → {layer,bounds} (the selection filled from its surroundings by the local LaMa model, on a new layer)"##, has_selection, remove, true),
        spec!(
            SUBJECT,
            "Select Subject",
            ["AI"],
            r##"{"sampleAllLayers":bool=false,"mode":"replace|add|subtract|intersect"="replace"} (the local BiRefNet model's matte as the selection)"##,
            has_doc,
            select_subject,
            true
        ),
        spec!(
            BACKGROUND,
            "Remove Background",
            ["AI"],
            r##"{"layer":id?,"sampleAllLayers":bool=false,"refine":bool=true} → {layer,bounds} (the local BiRefNet model's matte as the layer mask)"##,
            |s| pixel_layer(s).map(|_| ()),
            remove_background,
            true
        ),
        spec!(UPSCALE2, "Upscale 2×", ["AI"], r##"{} → {document,width,height} (Real-ESRGAN; a new document)"##, has_doc, |s, _| upscale(s, 2), true),
        spec!(UPSCALE4, "Upscale 4×", ["AI"], r##"{} → {document,width,height} (Real-ESRGAN; a new document)"##, has_doc, |s, _| upscale(s, 4), true),
        spec!(
            SETTINGS,
            "AI Settings…",
            ["AI"],
            r##"{"baseUrl":str?,"model":str?,"subjectAi":bool?} → the settings, whether a key is stored, the online and local models"##,
            |_| Ok(()),
            settings,
            false
        ),
        spec!(SET_KEY, "Set AI Key", [], r##"{"key":str} (stored in the keychain; empty removes it)"##, |_| Ok(()), set_key, false),
        spec!("ki.models", "AI Models", [], r##"{} → {models:[{id,label,installed,bytes,source,license}],folder}"##, |_| Ok(()), models, false),
        spec!("ki.downloadModels", "Download AI Models", [], r##"{"models":["matte|inpaint|upscale2|upscale4"]?=all}"##, |_| Ok(()), download_models, false),
        spec!("ki.deleteModels", "Delete AI Models", [], r##"{"models":["matte|inpaint|upscale2|upscale4"]?=all}"##, |_| Ok(()), delete_models, false),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_fill_window_holds_the_selection_at_a_listed_ratio() {
        let canvas = Rect::new(0, 0, 4000, 3000);
        for sel in [Rect::new(100, 100, 300, 200), Rect::new(3800, 2900, 3990, 2995), Rect::new(1000, 500, 1100, 2500), Rect::new(0, 0, 4000, 3000)] {
            let w = fill_window(sel, canvas);
            assert_eq!(w.intersect(&canvas), w, "{sel:?} → {w:?} inside");
            assert_eq!(w.intersect(&sel), sel.intersect(&canvas), "{sel:?} → {w:?} holds it");
            let (rw, rh) = ki::online::nearest_ratio(w.width() as usize, w.height() as usize);
            let r = w.width() as f64 / w.height() as f64;
            let fits_canvas = w.width() == canvas.width() || w.height() == canvas.height();
            assert!(fits_canvas || (r - rw as f64 / rh as f64).abs() < 0.02, "{sel:?} → {w:?}");
        }
    }

    #[test]
    fn generative_fill_needs_a_key() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
        if ki::keys::find(&ki::settings::Settings::load().base).is_none() {
            let e = s.execute(FILL, json!({"prompt": "a boat"})).unwrap_err();
            assert!(e.to_string().contains("API-Schlüssel"), "{e}");
        }
    }

    #[test]
    fn remove_needs_a_selection_and_the_settings_answer() {
        let mut s = Session::new();
        s.execute("file.new", json!({"width": 64, "height": 48})).unwrap();
        assert!(!s.is_enabled(REMOVE));
        let v = s.execute("ki.models", json!({})).unwrap();
        assert_eq!(v["models"].as_array().map(Vec::len), Some(4));
    }

    /// The local models on a real photo, through the commands (PHOTOCRAFT_KI_DIR with the models,
    /// PHOTOCRAFT_KI_PHOTO; PHOTOCRAFT_KI_OUT gets the composites). Skipped without them.
    #[test]
    fn local_models_on_a_photo() {
        let Ok(path) = std::env::var("PHOTOCRAFT_KI_PHOTO") else { return };
        if !ki::models::Model::ALL.iter().all(|m| m.installed()) {
            return;
        }
        let (rgb, w, h) = ki::img::decode(&std::fs::read(path).unwrap()).unwrap();
        let mut s = Session::new();
        s.execute("file.new", json!({"width": w, "height": h})).unwrap();
        s.edit("photo", |doc, _| {
            let bg = doc.layers[0].surface_mut().unwrap();
            let n = bg.channels();
            let data: Vec<f32> = rgb.chunks_exact(3).flat_map(|p| p.iter().copied().chain((n == 4).then_some(1.0))).collect();
            bg.write_region(Rect::new(0, 0, w as i32, h as i32), &data);
            Ok(())
        })
        .unwrap();
        let out = std::env::var("PHOTOCRAFT_KI_OUT").ok();
        let save = |s: &Session, name: &str| {
            if let Some(dir) = &out {
                let doc = &s.active().unwrap().doc;
                let px = composite_rgb(doc, doc.bounds());
                let (w, h) = (doc.size.width as usize, doc.size.height as usize);
                std::fs::write(format!("{dir}/{name}.png"), ki::img::png_rgb(&px, w, h).unwrap()).unwrap();
            }
        };
        // Select Subject: a selection covering the middle, not the corners.
        s.execute("ki.selectSubject", json!({})).unwrap();
        let sel = s.active().unwrap().doc.selection.clone().expect("a selection");
        let cov = |x: i32, y: i32| crate::fill_cmds::read_window_mask(&sel, Rect::new(x, y, x + 1, y + 1), &|| false).unwrap()[0];
        assert!(cov(2, 2) < 0.2, "corner selected");
        // Remove Background: a mask on the photo's layer.
        s.execute("edit.undo", json!({})).unwrap();
        let r = s.execute("ki.removeBackground", json!({})).unwrap();
        let id = LayerId(r["layer"].as_u64().unwrap());
        assert!(s.active().unwrap().doc.layer(id).unwrap().mask.is_some());
        save(&s, "cmd-removeBackground");
        s.execute("edit.undo", json!({})).unwrap();
        // Remove: a rectangle filled on a new layer.
        let (x0, y0) = ((w / 10) as i32, (h / 2) as i32);
        s.execute("select.rect", json!({"x": x0, "y": y0, "width": w / 8, "height": h / 8})).unwrap();
        let layers = s.active().unwrap().doc.layers.len();
        let r = s.execute(REMOVE, json!({})).unwrap();
        assert_eq!(s.active().unwrap().doc.layers.len(), layers + 1, "{r}");
        assert!(s.active().unwrap().doc.selection.is_none());
        save(&s, "cmd-remove");
        // Upscale 2×: a new document twice the size.
        let r = s.execute(UPSCALE2, json!({})).unwrap();
        assert_eq!((r["width"].as_u64(), r["height"].as_u64()), (Some(w as u64 * 2), Some(h as u64 * 2)));
        save(&s, "cmd-upscale2");
    }
}
