//! 2× and 4× upscaling with Real-ESRGAN, tile by tile.
//!
//! Every tile has the same size (224 px plus 16 px of overlap on each side, the image's edge
//! repeated where it runs out), so CoreML compiles one shape; only each tile's middle is kept,
//! so the seams don't show.

use crate::models::Model;
use crate::{Error, Progress, Result};

const TILE: usize = 224;
const PAD: usize = 16;

/// Upscale the `w × h` RGB image by `factor` (2 or 4).
pub fn upscale(rgb: &[f32], w: usize, h: usize, factor: usize, p: &Progress<'_>) -> Result<Vec<f32>> {
    let model = match factor {
        2 => Model::Upscale2,
        4 => Model::Upscale4,
        _ => return Err(Error::Other("nur ×2 oder ×4".into())),
    };
    if w == 0 || h == 0 || rgb.len() != w * h * 3 {
        return Err(Error::Other("kein Bild".into()));
    }
    let (ow, oh) = (w.checked_mul(factor).ok_or_else(too_big)?, h.checked_mul(factor).ok_or_else(too_big)?);
    if ow.checked_mul(oh).is_none_or(|n| n > 400_000_000) {
        return Err(too_big());
    }
    p.part(0.0, 0.05, |p| crate::run::session(model, p).map(|_| ()))?;
    let side = TILE + 2 * PAD;
    let (tx, ty) = (w.div_ceil(TILE), h.div_ceil(TILE));
    let total = tx * ty;
    let mut out = vec![0.0f32; ow * oh * 3];
    let at = |x: i64, y: i64| -> usize { (y.clamp(0, h as i64 - 1) as usize * w + x.clamp(0, w as i64 - 1) as usize) * 3 };
    for j in 0..ty {
        for i in 0..tx {
            p.check()?;
            let done = j * tx + i;
            p.at(0.05 + 0.95 * done as f32 / total as f32, &format!("Hochskalieren: Kachel {} von {total}", done + 1));
            let (x0, y0) = ((i * TILE) as i64 - PAD as i64, (j * TILE) as i64 - PAD as i64);
            let mut x = vec![0.0f32; 3 * side * side];
            for yy in 0..side {
                for xx in 0..side {
                    let s = at(x0 + xx as i64, y0 + yy as i64);
                    for c in 0..3 {
                        x[c * side * side + yy * side + xx] = rgb[s + c];
                    }
                }
            }
            let (_, y) = crate::run::run1(model, ("input", vec![1, 3, side as i64, side as i64], x), p)?;
            let os = side * factor;
            if y.len() != 3 * os * os {
                return Err(Error::Other("Hochskalier-Modell: unerwartete Ausgabe".into()));
            }
            // Keep the tile's middle (what lies inside the image).
            let (cw, chh) = (TILE.min(w - i * TILE) * factor, TILE.min(h - j * TILE) * factor);
            let (dx, dy) = (i * TILE * factor, j * TILE * factor);
            for yy in 0..chh {
                for xx in 0..cw {
                    let (sx, sy) = (PAD * factor + xx, PAD * factor + yy);
                    let o = ((dy + yy) * ow + dx + xx) * 3;
                    for c in 0..3 {
                        out[o + c] = y[c * os * os + sy * os + sx].clamp(0.0, 1.0);
                    }
                }
            }
        }
    }
    p.at(1.0, "");
    Ok(out)
}

fn too_big() -> Error {
    Error::Other("Das Ergebnis wäre zu groß (über 400 Megapixel)".into())
}
