//! The local models on a real photo. Needs the models (PHOTOCRAFT_KI_DIR) and a test photo
//! (PHOTOCRAFT_KI_PHOTO); skipped otherwise.

use photocraft_ki::{Progress, img, inpaint, matte, models::Model, upscale};

fn photo() -> Option<(Vec<f32>, usize, usize)> {
    let path = std::env::var("PHOTOCRAFT_KI_PHOTO").ok()?;
    if !Model::ALL.iter().all(|m| m.installed()) {
        return None;
    }
    img::decode(&std::fs::read(path).ok()?).ok()
}

fn save(name: &str, rgb: &[f32], w: usize, h: usize) {
    if let Ok(dir) = std::env::var("PHOTOCRAFT_KI_OUT") {
        std::fs::write(format!("{dir}/{name}.png"), img::png_rgb(rgb, w, h).unwrap()).unwrap();
    }
}

#[test]
fn matte_finds_the_subject() {
    let Some((rgb, w, h)) = photo() else { return };
    let t = std::time::Instant::now();
    let m = matte::subject(&rgb, w, h, &Progress::NONE).unwrap();
    eprintln!("matte {w}×{h}: {:?}", t.elapsed());
    assert_eq!(m.len(), w * h);
    let mean = m.iter().sum::<f32>() / m.len() as f32;
    assert!((0.05..0.95).contains(&mean), "mean coverage {mean}");
    // Corners are background.
    for (x, y) in [(2, 2), (w - 3, 2)] {
        assert!(m[y * w + x] < 0.2, "corner {x},{y}: {}", m[y * w + x]);
    }
    let vis: Vec<f32> = m.iter().flat_map(|v| [*v, *v, *v]).collect();
    save("matte", &vis, w, h);
}

#[test]
fn inpaint_fills_a_hole_from_its_surroundings() {
    let Some((rgb, w, h)) = photo() else { return };
    let mut mask = vec![0.0f32; w * h];
    let (x0, y0, x1, y1) = (w / 10, h / 2, w / 10 + w / 8, h / 2 + h / 8);
    for y in y0..y1 {
        for x in x0..x1 {
            mask[y * w + x] = 1.0;
        }
    }
    let t = std::time::Instant::now();
    let patch = inpaint::remove(&rgb, w, h, &mask, &Progress::NONE).unwrap();
    eprintln!("inpaint: {:?} {:?}", t.elapsed(), patch.rect);
    assert!(patch.rect.x as usize <= x0 && patch.rect.right() as usize >= x1);
    assert_eq!(patch.rgb.len(), patch.rect.w * patch.rect.h * 3);
    let blended = patch.blended(&img::crop(&rgb, w, 3, patch.rect));
    let mut out = rgb.clone();
    for y in 0..patch.rect.h {
        for x in 0..patch.rect.w {
            let d = ((patch.rect.y as usize + y) * w + patch.rect.x as usize + x) * 3;
            out[d..d + 3].copy_from_slice(&blended[(y * patch.rect.w + x) * 3..(y * patch.rect.w + x) * 3 + 3]);
        }
    }
    assert!(patch.alpha.iter().all(|a| (0.0..=1.0).contains(a)));
    save("inpaint", &out, w, h);
    // Outside the grown hole nothing changes.
    let far = (2 * w + 2) * 3;
    assert_eq!(&out[far..far + 3], &rgb[far..far + 3]);
}

#[test]
fn upscale_doubles_and_keeps_the_picture() {
    let Some((rgb, w, h)) = photo() else { return };
    let r = img::Rect { x: (w / 3) as i64, y: (h / 3) as i64, w: 300.min(w / 2), h: 260.min(h / 2) };
    let crop = img::crop(&rgb, w, 3, r);
    let t = std::time::Instant::now();
    let big = upscale::upscale(&crop, r.w, r.h, 2, &Progress::NONE).unwrap();
    eprintln!("upscale ×2 {}×{}: {:?}", r.w, r.h, t.elapsed());
    assert_eq!(big.len(), r.w * 2 * r.h * 2 * 3);
    save("upscale", &big, r.w * 2, r.h * 2);
    // Shrunk back it is close to the original.
    let back = img::resize(&big, r.w * 2, r.h * 2, 3, r.w, r.h);
    let err = back.iter().zip(&crop).map(|(a, b)| (a - b).abs()).sum::<f32>() / crop.len() as f32;
    assert!(err < 0.05, "mean error {err}");
}
