//! One generative fill against the real service. Costs money, so it runs only when asked:
//! PHOTOCRAFT_KI_KEY (the key), PHOTOCRAFT_KI_ONLINE_MODEL (model id) and PHOTOCRAFT_KI_PHOTO;
//! PHOTOCRAFT_KI_OUT gets the result.

use photocraft_ki::{Progress, img, online};

#[test]
fn generative_fill_against_the_service() {
    let (Ok(key), Ok(model), Ok(photo)) = (std::env::var("PHOTOCRAFT_KI_KEY"), std::env::var("PHOTOCRAFT_KI_ONLINE_MODEL"), std::env::var("PHOTOCRAFT_KI_PHOTO")) else {
        return;
    };
    let (rgb, w, h) = img::decode(&std::fs::read(photo).unwrap()).unwrap();
    // A box in the snow, bottom left.
    let mut mask = vec![0.0f32; w * h];
    let (x0, y0, x1, y1) = (w * 2 / 100, h * 72 / 100, w * 30 / 100, h * 98 / 100);
    for y in y0..y1 {
        for x in x0..x1 {
            mask[y * w + x] = 1.0;
        }
    }
    let service = online::Service { base: online::DEFAULT_BASE.into(), key };
    let req = online::Request { model: &model, prompt: "a small red rubber ball lying in the snow", rgb: &rgb, w, h, mask: Some(&mask) };
    let t = std::time::Instant::now();
    let report = |p: f32, what: &str| eprintln!("{:>5.1}% {what}", p * 100.0);
    let (out, ow, oh) = online::edit(&service, &req, &Progress { report: &report, cancelled: &|| false }).unwrap();
    eprintln!("{model}: {ow}×{oh} in {:?}", t.elapsed());
    assert!(ow > 100 && oh > 100);
    let aspect_in = w as f64 / h as f64;
    let aspect_out = ow as f64 / oh as f64;
    assert!((aspect_in / aspect_out - 1.0).abs() < 0.2, "aspect {aspect_in} → {aspect_out}");
    if let Ok(dir) = std::env::var("PHOTOCRAFT_KI_OUT") {
        let back = img::resize(&out, ow, oh, 3, w, h);
        std::fs::write(format!("{dir}/online-{model}.png"), img::png_rgb(&back, w, h).unwrap()).unwrap();
    }
}
