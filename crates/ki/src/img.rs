//! Small image helpers: resampling, masks, PNG in and out.

use rayon::prelude::*;

/// A rectangle `x, y, w, h` in pixels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rect {
    pub x: i64,
    pub y: i64,
    pub w: usize,
    pub h: usize,
}

impl Rect {
    pub fn right(&self) -> i64 {
        self.x + self.w as i64
    }
    pub fn bottom(&self) -> i64 {
        self.y + self.h as i64
    }
}

/// Resample an interleaved `sw × sh` image of `ch` channels to `dw × dh`. A triangle filter
/// scaled to the reduction when shrinking (so nothing aliases), bilinear when enlarging.
pub fn resize(src: &[f32], sw: usize, sh: usize, ch: usize, dw: usize, dh: usize) -> Vec<f32> {
    if sw == dw && sh == dh {
        return src.to_vec();
    }
    if sw == 0 || sh == 0 || dw == 0 || dh == 0 {
        return vec![0.0; dw * dh * ch];
    }
    // Horizontal pass: sw × sh → dw × sh.
    let wx = weights(sw, dw);
    let mut mid = vec![0.0f32; dw * sh * ch];
    mid.par_chunks_mut(dw * ch).enumerate().for_each(|(y, row)| {
        let s = &src[y * sw * ch..(y + 1) * sw * ch];
        for (x, (start, ws)) in wx.iter().enumerate() {
            for c in 0..ch {
                let mut v = 0.0;
                for (k, w) in ws.iter().enumerate() {
                    v += s[(start + k) * ch + c] * w;
                }
                row[x * ch + c] = v;
            }
        }
    });
    // Vertical pass: dw × sh → dw × dh.
    let wy = weights(sh, dh);
    let mut out = vec![0.0f32; dw * dh * ch];
    out.par_chunks_mut(dw * ch).enumerate().for_each(|(y, row)| {
        let (start, ws) = &wy[y];
        for (k, w) in ws.iter().enumerate() {
            let s = &mid[(start + k) * dw * ch..(start + k + 1) * dw * ch];
            for (o, v) in row.iter_mut().zip(s) {
                *o += v * w;
            }
        }
    });
    out
}

/// For each output sample: the first source sample and the normalised weights from there.
fn weights(src: usize, dst: usize) -> Vec<(usize, Vec<f32>)> {
    let scale = src as f64 / dst as f64;
    let support = scale.max(1.0);
    (0..dst)
        .map(|i| {
            let center = (i as f64 + 0.5) * scale;
            let lo = ((center - support).floor().max(0.0)) as usize;
            let hi = ((center + support).ceil() as usize).min(src);
            let mut ws: Vec<f32> = (lo..hi)
                .map(|j| {
                    let d = ((j as f64 + 0.5 - center) / support).abs();
                    (1.0 - d).max(0.0) as f32
                })
                .collect();
            let sum: f32 = ws.iter().sum();
            if sum > 0.0 {
                ws.iter_mut().for_each(|w| *w /= sum);
            } else {
                // Degenerate (a sample exactly between two at the very edge): the nearest one.
                let n = (center.floor() as usize).min(src - 1);
                return (n, vec![1.0]);
            }
            (lo, ws)
        })
        .collect()
}

/// Bounding box of the mask values above `t`, or `None` when there are none.
pub fn mask_bounds(mask: &[f32], w: usize, h: usize, t: f32) -> Option<Rect> {
    let (mut x0, mut y0, mut x1, mut y1) = (usize::MAX, usize::MAX, 0, 0);
    for y in 0..h {
        let row = &mask[y * w..(y + 1) * w];
        if let (Some(a), Some(b)) = (row.iter().position(|v| *v > t), row.iter().rposition(|v| *v > t)) {
            x0 = x0.min(a);
            x1 = x1.max(b + 1);
            y0 = y0.min(y);
            y1 = y + 1;
        }
    }
    (x1 > x0).then(|| Rect { x: x0 as i64, y: y0 as i64, w: x1 - x0, h: y1 - y0 })
}

/// Grow the set pixels (> 0) of a mask by `r` pixels (square neighbourhood), to 1.
pub fn dilate(mask: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 {
        return mask.to_vec();
    }
    // Separable max: rows, then columns.
    let mut rows = vec![0.0f32; w * h];
    rows.par_chunks_mut(w).enumerate().for_each(|(y, out)| {
        let src = &mask[y * w..(y + 1) * w];
        for x in 0..w {
            let (a, b) = (x.saturating_sub(r), (x + r + 1).min(w));
            out[x] = if src[a..b].iter().any(|v| *v > 0.0) { 1.0 } else { 0.0 };
        }
    });
    let mut out = vec![0.0f32; w * h];
    out.par_chunks_mut(w).enumerate().for_each(|(y, o)| {
        let (a, b) = (y.saturating_sub(r), (y + r + 1).min(h));
        for x in 0..w {
            o[x] = if (a..b).any(|yy| rows[yy * w + x] > 0.0) { 1.0 } else { 0.0 };
        }
    });
    out
}

/// A box blur of radius `r` (separable, two passes ≈ a soft feather).
pub fn blur(mask: &[f32], w: usize, h: usize, r: usize) -> Vec<f32> {
    if r == 0 || w == 0 || h == 0 {
        return mask.to_vec();
    }
    let pass = |src: &[f32], horizontal: bool| -> Vec<f32> {
        let mut out = vec![0.0f32; w * h];
        if horizontal {
            out.par_chunks_mut(w).enumerate().for_each(|(y, o)| {
                let s = &src[y * w..(y + 1) * w];
                for x in 0..w {
                    let (a, b) = (x.saturating_sub(r), (x + r + 1).min(w));
                    o[x] = s[a..b].iter().sum::<f32>() / (b - a) as f32;
                }
            });
        } else {
            out.par_chunks_mut(w).enumerate().for_each(|(y, o)| {
                let (a, b) = (y.saturating_sub(r), (y + r + 1).min(h));
                for x in 0..w {
                    o[x] = (a..b).map(|yy| src[yy * w + x]).sum::<f32>() / (b - a) as f32;
                }
            });
        }
        out
    };
    let a = pass(mask, true);
    let b = pass(&a, false);
    let c = pass(&b, true);
    pass(&c, false)
}

/// Cut `r` out of an interleaved image `w` wide with `ch` channels (`r` must lie inside).
pub fn crop(src: &[f32], w: usize, ch: usize, r: Rect) -> Vec<f32> {
    let mut out = Vec::with_capacity(r.w * r.h * ch);
    for y in r.y as usize..r.y as usize + r.h {
        let s = (y * w + r.x as usize) * ch;
        out.extend_from_slice(&src[s..s + r.w * ch]);
    }
    out
}

/// sRGB-encode a linear value (for 32-bit documents).
pub fn encode_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.003_130_8 { v * 12.92 } else { 1.055 * v.powf(1.0 / 2.4) - 0.055 }
}

/// Decode an sRGB value to linear.
pub fn decode_srgb(v: f32) -> f32 {
    let v = v.clamp(0.0, 1.0);
    if v <= 0.040_45 { v / 12.92 } else { ((v + 0.055) / 1.055).powf(2.4) }
}

/// PNG bytes of an RGB image.
pub fn png_rgb(rgb: &[f32], w: usize, h: usize) -> crate::Result<Vec<u8>> {
    let bytes: Vec<u8> = rgb.iter().map(|v| (v.clamp(0.0, 1.0) * 255.0).round() as u8).collect();
    let img = image::RgbImage::from_raw(w as u32, h as u32, bytes).ok_or_else(|| crate::Error::Other("Bildgröße passt nicht".into()))?;
    let mut out = std::io::Cursor::new(Vec::new());
    img.write_to(&mut out, image::ImageFormat::Png).map_err(|e| crate::Error::Other(format!("PNG: {e}")))?;
    Ok(out.into_inner())
}

/// An image file (PNG, JPEG or WebP) as RGB 0–1 and its size.
pub fn decode(bytes: &[u8]) -> crate::Result<(Vec<f32>, usize, usize)> {
    let img = image::load_from_memory(bytes).map_err(|e| crate::Error::Other(format!("Das zurückgegebene Bild ist nicht lesbar ({e})")))?;
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    Ok((rgb.into_raw().into_iter().map(|b| f32::from(b) / 255.0).collect(), w, h))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn resize_keeps_flat_images_flat_and_sizes_right() {
        let src = vec![0.25f32; 37 * 23 * 3];
        for (dw, dh) in [(10, 7), (37, 23), (100, 61), (1, 1)] {
            let out = resize(&src, 37, 23, 3, dw, dh);
            assert_eq!(out.len(), dw * dh * 3);
            assert!(out.iter().all(|v| (v - 0.25).abs() < 1e-5), "{dw}×{dh}");
        }
    }

    #[test]
    fn resize_follows_a_gradient() {
        let src: Vec<f32> = (0..64).map(|x| x as f32 / 63.0).collect();
        let out = resize(&src, 64, 1, 1, 16, 1);
        assert!(out.windows(2).all(|p| p[1] > p[0]));
        assert!(out[0] < 0.1 && out[15] > 0.9);
    }

    #[test]
    fn bounds_dilate_and_blur() {
        let (w, h) = (20, 10);
        let mut m = vec![0.0f32; w * h];
        m[5 * w + 7] = 1.0;
        assert_eq!(mask_bounds(&m, w, h, 0.5), Some(Rect { x: 7, y: 5, w: 1, h: 1 }));
        let d = dilate(&m, w, h, 2);
        assert_eq!(mask_bounds(&d, w, h, 0.5), Some(Rect { x: 5, y: 3, w: 5, h: 5 }));
        let b = blur(&d, w, h, 1);
        assert!(b[5 * w + 7] > 0.5 && b[5 * w + 2] == 0.0 && b[5 * w + 3] > 0.0, "{:?}", &b[5 * w..6 * w]);
        assert_eq!(mask_bounds(&vec![0.0; 4], 2, 2, 0.0), None);
    }

    #[test]
    fn png_round_trip() {
        let rgb: Vec<f32> = (0..4 * 3 * 3).map(|i| (i % 7) as f32 / 6.0).collect();
        let png = png_rgb(&rgb, 4, 3).unwrap();
        let (back, w, h) = decode(&png).unwrap();
        assert_eq!((w, h), (4, 3));
        assert!(back.iter().zip(&rgb).all(|(a, b)| (a - b).abs() < 0.003));
    }

    #[test]
    fn srgb_round_trip() {
        for v in [0.0, 0.002, 0.2, 0.5, 1.0] {
            assert!((decode_srgb(encode_srgb(v)) - v).abs() < 1e-4);
        }
    }
}
