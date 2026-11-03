//! Object removal: LaMa fills the selected area from its surroundings.
//!
//! LaMa works at 512 × 512. The area around the selection is cut out with as much context as the
//! hole is large (at least 512 px, so a small hole is filled at full resolution), scaled to
//! 512 × 512, filled, and scaled back.

use crate::img::{self, Rect};
use crate::models::Model;
use crate::{Error, Progress, Result};

const SIDE: usize = 512;

/// The fill over `rect`: its pixels and how much of each to use (1 inside the hole, falling
/// off over a few pixels at its edge, 0 outside).
#[derive(Clone, Debug, PartialEq)]
pub struct Patch {
    pub rect: Rect,
    pub rgb: Vec<f32>,
    pub alpha: Vec<f32>,
}

impl Patch {
    /// The patch laid over the original pixels it came from (`crop` over the same `rect`).
    pub fn blended(&self, original: &[f32]) -> Vec<f32> {
        self.rgb.chunks_exact(3).zip(original.chunks_exact(3)).zip(&self.alpha).flat_map(|((f, o), a)| (0..3).map(move |c| o[c] + (f[c] - o[c]) * a)).collect()
    }
}

/// The square-ish window of `side` around `bbox`, inside the `w × h` image.
fn window(bbox: Rect, w: usize, h: usize, side: usize) -> Rect {
    let (cw, ch) = (side.min(w), side.min(h));
    let cx = bbox.x + bbox.w as i64 / 2;
    let cy = bbox.y + bbox.h as i64 / 2;
    let x = (cx - cw as i64 / 2).clamp(0, (w - cw) as i64);
    let y = (cy - ch as i64 / 2).clamp(0, (h - ch) as i64);
    Rect { x, y, w: cw, h: ch }
}

/// Fill the area `mask` covers (> 0) in the `w × h` RGB image.
pub fn remove(rgb: &[f32], w: usize, h: usize, mask: &[f32], p: &Progress<'_>) -> Result<Patch> {
    if w == 0 || h == 0 || rgb.len() != w * h * 3 || mask.len() != w * h {
        return Err(Error::Other("kein Bild".into()));
    }
    let Some(bbox) = img::mask_bounds(mask, w, h, 0.0) else {
        return Err(Error::Other("Bitte zuerst auswählen, was entfernt werden soll".into()));
    };
    p.part(0.0, 0.6, |p| crate::run::session(Model::Inpaint, p).map(|_| ()))?;
    p.at(0.6, "Objekt wird entfernt");
    // A few pixels more than selected: LaMa needs the object's edge and shadow inside the hole.
    let grow = (bbox.w.max(bbox.h) / 100).clamp(3, 24);
    let hole = img::dilate(mask, w, h, grow);
    let hb = img::mask_bounds(&hole, w, h, 0.0).unwrap_or(bbox);
    let side = (hb.w.max(hb.h) * 2).max(SIDE);
    let win = window(hb, w, h, side);
    let crop = img::crop(rgb, w, 3, win);
    let crop_hole = img::crop(&hole, w, 1, win);
    let small = img::resize(&crop, win.w, win.h, 3, SIDE, SIDE);
    let small_hole: Vec<f32> = img::resize(&crop_hole, win.w, win.h, 1, SIDE, SIDE).into_iter().map(|v| if v > 0.01 { 1.0 } else { 0.0 }).collect();
    let mut x = vec![0.0f32; 3 * SIDE * SIDE];
    for (i, px) in small.chunks_exact(3).enumerate() {
        for c in 0..3 {
            // The hole's own pixels are not looked at; zero them as LaMa was trained.
            x[c * SIDE * SIDE + i] = if small_hole[i] > 0.0 { 0.0 } else { px[c] };
        }
    }
    p.check()?;
    let (shape, out) = crate::run::run(
        Model::Inpaint,
        vec![("image", vec![1, 3, SIDE as i64, SIDE as i64], x), ("mask", vec![1, 1, SIDE as i64, SIDE as i64], small_hole)],
        p,
    )?;
    if out.len() != 3 * SIDE * SIDE {
        return Err(Error::Other(format!("Entfern-Modell: unerwartete Ausgabe {shape:?}")));
    }
    p.at(0.9, "Ergebnis wird eingesetzt");
    // The output is 0–255, planar.
    let scale = if out.iter().any(|v| *v > 1.5) { 1.0 / 255.0 } else { 1.0 };
    let mut filled_small = vec![0.0f32; SIDE * SIDE * 3];
    for i in 0..SIDE * SIDE {
        for c in 0..3 {
            filled_small[i * 3 + c] = (out[c * SIDE * SIDE + i] * scale).clamp(0.0, 1.0);
        }
    }
    let filled = img::resize(&filled_small, SIDE, SIDE, 3, win.w, win.h);
    // Over the hole's box: the filled pixels inside the (grown) hole, the original outside, with
    // a soft seam of a few pixels.
    let soft = img::blur(&crop_hole, win.w, win.h, (grow / 2).max(1));
    let rect = Rect { x: hb.x, y: hb.y, w: hb.w, h: hb.h };
    let mut rgb_out = Vec::with_capacity(rect.w * rect.h * 3);
    let mut alpha = Vec::with_capacity(rect.w * rect.h);
    for y in rect.y..rect.bottom() {
        for x in rect.x..rect.right() {
            let (lx, ly) = ((x - win.x) as usize, (y - win.y) as usize);
            let k = ly * win.w + lx;
            alpha.push(soft[k].max(mask[y as usize * w + x as usize]).clamp(0.0, 1.0));
            rgb_out.extend_from_slice(&filled[k * 3..k * 3 + 3]);
        }
    }
    p.at(1.0, "");
    Ok(Patch { rect, rgb: rgb_out, alpha })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_is_inside_and_around_the_hole() {
        let b = Rect { x: 900, y: 10, w: 40, h: 30 };
        let win = window(b, 1000, 600, 512);
        assert_eq!((win.w, win.h), (512, 512));
        assert!(win.x >= 0 && win.right() <= 1000 && win.y == 0);
        assert!(win.x <= b.x && win.right() >= b.right());
        let small = window(b, 300, 200, 512);
        assert_eq!(small, Rect { x: 0, y: 0, w: 300, h: 200 });
    }
}
