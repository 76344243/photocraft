//! Fork patch (76344243/photocraft): the AI menu's dialogs.
//!
//! - AI › Generative Fill…: the prompt and the online model (with its price); OK runs
//!   `ki.generativeFill`.
//! - AI › AI Settings…: the online service, its API key (stored in the keychain, never shown
//!   again), the default model, whether Select Subject uses the local model, and the local models
//!   (download, delete).

use photocraft_engine::ki_cmds::{self, photocraft_ki as ki};
use serde_json::{Map, Value, json};

use crate::PhotocraftApp;
use crate::state::DialogKind;

const FILL_MARK: &str = "__kiFill";
const SETTINGS_MARK: &str = "__kiSettings";

pub fn owns(fields: &Map<String, Value>) -> bool {
    fields.contains_key(FILL_MARK) || fields.contains_key(SETTINGS_MARK)
}

pub fn is_fill(fields: &Map<String, Value>) -> bool {
    fields.contains_key(FILL_MARK)
}

fn s<'a>(f: &'a Map<String, Value>, k: &str) -> &'a str {
    f.get(k).and_then(Value::as_str).unwrap_or("")
}

/// A model's line in the dropdown: name, price per image in cents, note.
fn model_line(m: &ki::online::OnlineModel) -> String {
    let cents = m.price_usd * 100.0;
    let price = if (cents - cents.round()).abs() < 0.05 { format!("{cents:.0}") } else { format!("{cents:.1}").replace('.', ",") };
    if m.note.is_empty() { format!("{} – {price} ct", m.label) } else { format!("{} – {price} ct ({})", m.label, m.note) }
}

fn model_options(current: &str) -> Vec<(String, String)> {
    let mut v: Vec<(String, String)> = ki::online::MODELS.iter().map(|m| (m.id.to_string(), model_line(m))).collect();
    if !current.is_empty() && !v.iter().any(|(id, _)| id == current) {
        v.push((current.to_string(), current.to_string()));
    }
    v
}

fn model_dropdown(ui: &mut egui::Ui, id: &str, f: &mut Map<String, Value>) {
    let mut model = s(f, "model").to_string();
    let opts = model_options(&model);
    let refs: Vec<(String, &str)> = opts.iter().map(|(id, l)| (id.clone(), l.as_str())).collect();
    if crate::widgets::dropdown(ui, id, tl!("Model"), &mut model, &refs, 330.0) {
        f.insert("model".into(), json!(model));
    }
}

fn row_label(ui: &mut egui::Ui, text: &str) {
    ui.allocate_ui_with_layout(egui::vec2(110.0, 22.0), egui::Layout::right_to_left(egui::Align::Center), |ui| ui.label(text));
}

// ---- Generative Fill ------------------------------------------------------------------------

pub fn open_fill(app: &mut PhotocraftApp) -> Result<u64, String> {
    let st = app.session.active().ok_or("no document open")?;
    let selection = st.doc.selection.as_ref().is_some_and(|m| !m.content_bounds().intersect(&st.doc.bounds()).is_empty());
    let settings = ki::settings::Settings::load();
    let key = ki::keys::find(&settings.base).is_some();
    let fields = json!({
        FILL_MARK: true, "__command": ki_cmds::FILL, "__label": "Generative Fill",
        "prompt": "", "model": settings.model, "selection": selection, "keySet": key, "base": settings.base,
    });
    Ok(app.ui.open_dialog(DialogKind::Command, fields.as_object().cloned().unwrap_or_default()))
}

fn fill_body(ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    ui.spacing_mut().item_spacing.y = 8.0;
    ui.set_max_width(460.0);
    let selection = f.get("selection").and_then(Value::as_bool).unwrap_or(false);
    ui.label(if selection {
        tl!("Describe what should appear in the selected area. Leave it empty to fill the area to match its surroundings.")
    } else {
        tl!("Nothing is selected, so the whole picture is changed. Describe the change.")
    });
    let mut prompt = s(f, "prompt").to_string();
    let r = ui.add(egui::TextEdit::multiline(&mut prompt).desired_rows(3).desired_width(440.0).hint_text(tl!("e.g. a small wooden boat on the water")));
    if !ui.is_sizing_pass() && !f.contains_key("__focused") {
        r.request_focus();
        f.insert("__focused".into(), json!(true));
    }
    if r.changed() {
        f.insert("prompt".into(), json!(prompt));
    }
    ui.horizontal(|ui| {
        ui.label(tl!("Model:"));
        model_dropdown(ui, "ki-fill-model", f);
    });
    let host = s(f, "base").trim_start_matches("https://").trim_end_matches('/').to_string();
    let note = crate::i18n::fmt(
        tl!("The area around the selection is sent to {service}, which bills each image at the price shown. The result arrives as a new layer with the selection as its mask."),
        &[("service", host.as_str())],
    );
    ui.label(egui::RichText::new(note).small().weak());
    if !f.get("keySet").and_then(Value::as_bool).unwrap_or(false) {
        ui.label(egui::RichText::new(tl!("No API key yet: enter it under AI › AI Settings…")).color(egui::Color32::from_rgb(0xd0, 0x80, 0x20)));
    }
}

fn fill_confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    let model = s(f, "model").to_string();
    // The model chosen becomes the default.
    if !model.is_empty() && model != ki::settings::Settings::load().model {
        let _ = app.run(ki_cmds::SETTINGS, json!({"model": model}));
    }
    app.run(ki_cmds::FILL, json!({"prompt": s(f, "prompt"), "model": model}))
}

// ---- AI Settings -------------------------------------------------------------------------

pub fn open_settings(app: &mut PhotocraftApp) -> Result<u64, String> {
    let st = ki::settings::Settings::load();
    let hint = ki::keys::find(&st.base).map(|k| ki::keys::hint(&k));
    let fields = json!({
        SETTINGS_MARK: true, "__command": ki_cmds::SETTINGS, "__label": "AI Settings",
        "baseUrl": st.base, "model": st.model, "subjectAi": st.subject_ai, "key": "", "keyHint": hint, "removeKey": false,
    });
    Ok(app.ui.open_dialog(DialogKind::Command, fields.as_object().cloned().unwrap_or_default()))
}

fn mb(bytes: u64) -> String {
    format!("{} MB", bytes.div_ceil(1_000_000))
}

fn settings_body(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    ui.spacing_mut().item_spacing.y = 6.0;
    ui.set_max_width(560.0);
    ui.strong(tl!("Online (Generative Fill)"));
    ui.horizontal(|ui| {
        row_label(ui, tl!("Service:"));
        let mut base = s(f, "baseUrl").to_string();
        if ui.add(egui::TextEdit::singleline(&mut base).desired_width(330.0)).changed() {
            f.insert("baseUrl".into(), json!(base));
        }
    });
    ui.horizontal(|ui| {
        row_label(ui, tl!("API key:"));
        let mut key = s(f, "key").to_string();
        let hint = match f.get("keyHint").and_then(Value::as_str) {
            Some(h) if !f.get("removeKey").and_then(Value::as_bool).unwrap_or(false) => crate::i18n::fmt(tl!("stored ({hint}) – type to replace"), &[("hint", h)]),
            _ => tl!("not stored yet").to_string(),
        };
        if ui.add(egui::TextEdit::singleline(&mut key).password(true).desired_width(250.0).hint_text(hint)).changed() {
            f.insert("key".into(), json!(key));
        }
        if f.get("keyHint").and_then(Value::as_str).is_some() && ui.button(tl!("Remove")).clicked() {
            f.insert("removeKey".into(), json!(true));
            f.insert("key".into(), json!(""));
        }
    });
    ui.horizontal(|ui| {
        row_label(ui, tl!("Default model:"));
        model_dropdown(ui, "ki-settings-model", f);
    });
    ui.label(egui::RichText::new(tl!("The key is kept in the macOS keychain, not in PhotoCraft's settings.")).small().weak());
    ui.add_space(8.0);
    ui.separator();
    ui.strong(tl!("On this Mac (free, offline)"));
    let mut subject = f.get("subjectAi").and_then(Value::as_bool).unwrap_or(true);
    if crate::widgets::checkbox(ui, &mut subject, tl!("Select Subject and Remove Background use the AI model")).changed() {
        f.insert("subjectAi".into(), json!(subject));
    }
    let missing: u64 = ki::models::Model::ALL.iter().filter(|m| !m.installed()).map(|m| m.spec().bytes).sum();
    egui::Grid::new("ki-models").num_columns(3).spacing([12.0, 4.0]).show(ui, |ui| {
        for m in ki::models::Model::ALL {
            let spec = m.spec();
            ui.label(tl!(spec.label));
            if m.installed() {
                ui.label(egui::RichText::new(tl!("loaded")).color(egui::Color32::from_rgb(0x3c, 0xa0, 0x50)));
            } else {
                ui.label(egui::RichText::new(crate::i18n::fmt(tl!("loads on first use ({size})"), &[("size", mb(spec.bytes).as_str())])).weak());
            }
            ui.label(egui::RichText::new(format!("{} · {}", spec.source, spec.license)).small().weak());
            ui.end_row();
        }
    });
    ui.horizontal(|ui| {
        if missing > 0 && ui.button(crate::i18n::fmt(tl!("Load all now ({size})"), &[("size", mb(missing).as_str())])).clicked() {
            let _ = app.run("ki.downloadModels", json!({}));
        }
        if ki::models::Model::ALL.iter().any(|m| m.installed()) && ui.button(tl!("Delete models")).clicked() {
            let _ = app.run("ki.deleteModels", json!({}));
        }
    });
    ui.label(egui::RichText::new(ki::models::dir().display().to_string()).small().weak());
}

fn settings_confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    let r = app.run(ki_cmds::SETTINGS, json!({"baseUrl": s(f, "baseUrl"), "model": s(f, "model"), "subjectAi": f.get("subjectAi").and_then(Value::as_bool).unwrap_or(true)}))?;
    let key = s(f, "key").trim();
    if !key.is_empty() {
        return app.run(ki_cmds::SET_KEY, json!({"key": key}));
    }
    if f.get("removeKey").and_then(Value::as_bool).unwrap_or(false) {
        return app.run(ki_cmds::SET_KEY, json!({"key": ""}));
    }
    Ok(r)
}

// ---- Routing ---------------------------------------------------------------------------------

pub fn body(app: &mut PhotocraftApp, ui: &mut egui::Ui, f: &mut Map<String, Value>) {
    if is_fill(f) { fill_body(ui, f) } else { settings_body(app, ui, f) }
}

pub fn confirm(app: &mut PhotocraftApp, f: &Map<String, Value>) -> Result<Value, String> {
    if is_fill(f) { fill_confirm(app, f) } else { settings_confirm(app, f) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn model_lines_show_the_price_in_cents() {
        let line = |id: &str| model_line(ki::online::MODELS.iter().find(|m| m.id == id).unwrap());
        assert_eq!(line("seedream-5-0-flash-260915"), "Seedream 5.0 Flash – 1,8 ct (am günstigsten)");
        assert_eq!(line("gpt-image-2-vip"), "GPT Image 2 – 3 ct");
        assert!(model_options("my-new-model").iter().any(|(id, _)| id == "my-new-model"));
    }
}
