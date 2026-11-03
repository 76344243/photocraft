//! The subject's matte (Select Subject, Remove Background): BiRefNet lite at 512 × 512.

use crate::models::Model;
use crate::{Error, Progress, Result, img};

const SIDE: usize = 512;
const MEAN: [f32; 3] = [0.485, 0.456, 0.406];
const STD: [f32; 3] = [0.229, 0.224, 0.225];

/// Coverage 0–1 of the main subject for each pixel of the `w × h` RGB image.
pub fn subject(rgb: &[f32], w: usize, h: usize, p: &Progress<'_>) -> Result<Vec<f32>> {
    if w == 0 || h == 0 || rgb.len() != w * h * 3 {
        return Err(Error::Other("kein Bild".into()));
    }
    let input = p.part(0.0, 0.5, |p| {
        crate::run::session(Model::Matte, p)?;
        p.at(1.0, "Motiv wird gesucht");
        Ok::<_, Error>(())
    });
    input?;
    let small = img::resize(rgb, w, h, 3, SIDE, SIDE);
    // NCHW, ImageNet-normalised.
    let mut x = vec![0.0f32; 3 * SIDE * SIDE];
    for (i, px) in small.chunks_exact(3).enumerate() {
        for c in 0..3 {
            x[c * SIDE * SIDE + i] = (px[c] - MEAN[c]) / STD[c];
        }
    }
    p.at(0.55, "Motiv wird gesucht");
    let (shape, logits) = crate::run::run1(Model::Matte, ("input_image", vec![1, 3, SIDE as i64, SIDE as i64], x), p)?;
    if shape.iter().product::<i64>() as usize != SIDE * SIDE {
        return Err(Error::Other(format!("Freistell-Modell: unerwartete Ausgabe {shape:?}")));
    }
    p.at(0.9, "Kante wird übertragen");
    let matte: Vec<f32> = logits.iter().map(|l| 1.0 / (1.0 + (-l).exp())).collect();
    let full = img::resize(&matte, SIDE, SIDE, 1, w, h);
    p.at(1.0, "");
    Ok(full.into_iter().map(|v| v.clamp(0.0, 1.0)).collect())
}
