// Derived from OpenCV 5.0 (contours_new.cpp, convhull.cpp, rotcalipers.cpp),
// https://github.com/opencv/opencv, Apache License 2.0.
// Copyright (C) 2000-2025 the OpenCV authors (see NOTICE).
// Modified: ported to Rust and reduced to the cases this crate needs.

//! `cv2.findContours(RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)`, `contourArea`,
//! `convexHull`, `minAreaRect` and `boxPoints`, ported from OpenCV 5.0
//! (`contours_new.cpp`, geometry module `convhull.cpp`, `rotcalipers.cpp`)
//! so contour order, hull start point and box corner order are the same.

use crate::image::Gray8;

/// Contour point (x, y).
pub type Point = (i32, i32);

const MASK8_RIGHT: i8 = -128; // 0x80
const MASK8_NEW: i8 = 2;
const MASK8_FLAGS: i8 = -2; // 0xFE
const MAX_SIZE: i8 = 16;
const CHAIN_DELTAS: [Point; 8] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
];

/// Border following on a 1-px zero-padded 0/1 image (`icvFetchContourEx`
/// with `CHAIN_APPROX_SIMPLE`).
struct Scanner {
    img: Vec<i8>,
    step: isize,
    width: i32,
    height: i32,
}

impl Scanner {
    #[inline]
    fn delta(&self, s: i8) -> isize {
        let (dx, dy) = CHAIN_DELTAS[(s % 8) as usize];
        dx as isize + dy as isize * self.step
    }

    #[inline]
    fn at(&self, x: i32, y: i32) -> i8 {
        self.img[(y as isize * self.step + x as isize) as usize]
    }

    /// Traces the border starting at `start` (padded coordinates); points are
    /// image coordinates (offset -1, -1).
    fn fetch(&mut self, start: Point, is_hole: bool) -> Vec<Point> {
        let mut out = Vec::new();
        let i0 = start.1 as isize * self.step + start.0 as isize;
        let mut pt = (start.0 - 1, start.1 - 1);
        let mut s_end: i8 = if is_hole { 0 } else { 4 };
        let mut s = s_end;
        let mut i1;
        loop {
            s = (s - 1) & 7;
            i1 = i0 + self.delta(s);
            if self.img[i1 as usize] != 0 || s == s_end {
                break;
            }
        }
        if s == s_end {
            self.img[i0 as usize] = MASK8_NEW | MASK8_RIGHT;
            out.push(pt);
            return out;
        }
        let mut i3 = i0;
        let mut i4;
        let mut prev_s = s ^ 4;
        loop {
            s_end = s;
            s = s.min(15);
            i4 = i3;
            while s < MAX_SIZE - 1 {
                s += 1;
                i4 = i3 + self.delta(s);
                if self.img[i4 as usize] != 0 {
                    break;
                }
            }
            s &= 7;
            if ((s as i32 - 1) as u32) < (s_end as i32 as u32) {
                self.img[i3 as usize] = MASK8_NEW | MASK8_RIGHT;
            } else if self.img[i3 as usize] == 1 {
                self.img[i3 as usize] = MASK8_NEW;
            }
            if s != prev_s {
                out.push(pt);
            }
            prev_s = s;
            pt.0 += CHAIN_DELTAS[s as usize].0;
            pt.1 += CHAIN_DELTAS[s as usize].1;
            if i4 == i0 && i3 == i1 {
                break;
            }
            i3 = i4;
            s = (s + 4) & 7;
        }
        out
    }
}

/// `cv2.findContours(bw, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE)[0]`: outer
/// borders of the non-zero regions, in OpenCV's output order (last found
/// first).
pub fn find_contours_external(bw: &Gray8) -> Vec<Vec<Point>> {
    let (w, h) = (bw.width as i32 + 2, bw.height as i32 + 2);
    let mut img = vec![0i8; (w * h) as usize];
    for y in 0..bw.height {
        let row = bw.row(y);
        let o = (y + 1) * w as usize + 1;
        for (d, &v) in img[o..o + bw.width].iter_mut().zip(row) {
            *d = (v != 0) as i8;
        }
    }
    let mut sc = Scanner {
        img,
        step: w as isize,
        width: w - 1,
        height: h - 1,
    };
    let mut found = Vec::new();
    let (mut x, mut y) = (1, 1);
    let mut last_pos: Point = (0, 1);
    let mut prev = sc.at(x - 1, y) as i32;
    while y < sc.height {
        while x < sc.width {
            // findNextX: first x whose value differs from prev
            let mut p = prev;
            while x < sc.width {
                p = sc.at(x, y) as i32;
                if p != prev {
                    break;
                }
                x += 1;
            }
            if x >= sc.width {
                break;
            }
            // contourScan (RETR_EXTERNAL: outer borders not inside another)
            let mut is_hole = false;
            let mut take = true;
            if !(prev == 0 && p == 1) {
                if p != 0 || prev < 1 {
                    take = false;
                } else {
                    if prev & MASK8_FLAGS as i32 != 0 {
                        last_pos.0 = x - 1;
                    }
                    is_hole = true;
                }
            }
            if take && (is_hole || sc.at(last_pos.0, last_pos.1) > 0) {
                take = false;
            }
            if take {
                last_pos.0 = x - is_hole as i32;
                found.push(sc.fetch((x - is_hole as i32, y), is_hole));
                // resume after the start pixel; prev is its (now marked) value
                x += 1;
                prev = sc.at(x - 1, y) as i32;
                continue;
            }
            prev = p;
            if prev & MASK8_FLAGS as i32 != 0 {
                last_pos.0 = x;
            }
            x += 1;
        }
        last_pos = (0, y + 1);
        x = 1;
        prev = 0;
        y += 1;
    }
    found.reverse();
    found
}

/// `cv2.contourArea(pts)` (not oriented): shoelace in f64 over f32 points.
pub fn contour_area(pts: &[(f32, f32)]) -> f64 {
    let Some(&last) = pts.last() else { return 0.0 };
    let mut a = 0.0f64;
    let mut prev = last;
    for &p in pts {
        a += (prev.0 as f64).mul_add(p.1 as f64, -(prev.1 as f64 * p.0 as f64));
        prev = p;
    }
    (a * 0.5).abs()
}

/// Integer contour as f32 points.
pub fn to_f32(pts: &[Point]) -> Vec<(f32, f32)> {
    pts.iter().map(|&(x, y)| (x as f32, y as f32)).collect()
}

fn sign(v: i64) -> i32 {
    (v > 0) as i32 - (v < 0) as i32
}

/// `Sklansky_<int, int64>` over sorted point indices.
fn sklansky(
    p: &[Point],
    start: usize,
    end: usize,
    stack: &mut [usize],
    nsign: i32,
    sign2: i32,
) -> usize {
    let incr: isize = if end > start { 1 } else { -1 };
    let (mut pprev, mut pcur) = (start as isize, start as isize + incr);
    let mut pnext = pcur + incr;
    let mut stacksize = 3usize;
    if start == end || p[start] == p[end] {
        stack[0] = start;
        return 1;
    }
    stack[0] = pprev as usize;
    stack[1] = pcur as usize;
    stack[2] = pnext as usize;
    let end = end as isize + incr;
    while pnext != end {
        let cury = p[pcur as usize].1 as i64;
        let nexty = p[pnext as usize].1 as i64;
        let by = nexty - cury;
        if sign(by) != nsign {
            let a = (
                p[pcur as usize].0 as i64 - p[pprev as usize].0 as i64,
                cury - p[pprev as usize].1 as i64,
            );
            let b = (p[pnext as usize].0 as i64 - p[pcur as usize].0 as i64, by);
            let convexity = a.1 * b.0 - a.0 * b.1;
            if sign(convexity) == sign2 && (a.0 != 0 || a.1 != 0) {
                pprev = pcur;
                pcur = pnext;
                pnext += incr;
                stack[stacksize] = pnext as usize;
                stacksize += 1;
            } else if pprev == start as isize {
                pcur = pnext;
                stack[1] = pcur as usize;
                pnext += incr;
                stack[2] = pnext as usize;
            } else {
                stack[stacksize - 2] = pnext as usize;
                pcur = pprev;
                pprev = stack[stacksize - 4] as isize;
                stacksize -= 1;
            }
        } else {
            pnext += incr;
            stack[stacksize - 1] = pnext as usize;
        }
    }
    stacksize - 1
}

/// `cv2.convexHull(pts, clockwise=False, returnPoints=True)` for integer
/// points (Sklansky; same start point and direction as OpenCV).
pub fn convex_hull(points: &[Point]) -> Vec<Point> {
    let total = points.len();
    if total == 0 {
        return Vec::new();
    }
    let mut order: Vec<usize> = (0..total).collect();
    order.sort_by_key(|&i| (points[i].0, points[i].1, i));
    let p: Vec<Point> = order.iter().map(|&i| points[i]).collect();
    let (mut miny, mut maxy) = (0, 0);
    for i in 1..total {
        let y = p[i].1;
        if p[miny].1 > y {
            miny = i;
        }
        if p[maxy].1 < y {
            maxy = i;
        }
    }
    let mut hull: Vec<usize> = Vec::new();
    let mut stack = vec![0usize; total + 2];
    if p[0] == p[total - 1] {
        hull.push(0);
    } else {
        // upper half (clockwise = false: tr and tl swapped)
        let tl_count = sklansky(&p, 0, maxy, &mut stack, -1, 1);
        let tl: Vec<usize> = stack[..tl_count].to_vec();
        let tr_count = sklansky(&p, total - 1, maxy, &mut stack[tl_count..], -1, -1);
        let tr: Vec<usize> = stack[tl_count..tl_count + tr_count].to_vec();
        let (tl, tr) = (tr, tl);
        for &i in &tl[..tl.len().saturating_sub(1)] {
            hull.push(i);
        }
        for i in (1..tr.len()).rev() {
            hull.push(tr[i]);
        }
        let stop_idx = if tr.len() > 2 {
            Some(tr[1])
        } else if tl.len() > 2 {
            Some(tl[tl.len() - 2])
        } else {
            None
        };
        // lower half
        let bl_count = sklansky(&p, 0, miny, &mut stack, 1, -1);
        let mut bl: Vec<usize> = stack[..bl_count].to_vec();
        let br_count = sklansky(&p, total - 1, miny, &mut stack[bl_count..], 1, 1);
        let mut br: Vec<usize> = stack[bl_count..bl_count + br_count].to_vec();
        if let Some(stop) = stop_idx {
            let check = if bl.len() > 2 {
                Some(bl[1])
            } else if bl.len() + br.len() > 2 {
                Some(br[2 - bl.len()])
            } else {
                None
            };
            if let Some(c) = check
                && (c == stop || p[c] == p[stop])
            {
                bl.truncate(2);
                br.truncate(2);
            }
        }
        for &i in &bl[..bl.len().saturating_sub(1)] {
            hull.push(i);
        }
        for i in (1..br.len()).rev() {
            hull.push(br[i]);
        }
    }
    // back to input indices, then the cyclic shift towards a monotonic
    // index sequence
    let mut idx: Vec<usize> = hull.iter().map(|&i| order[i]).collect();
    let nout = idx.len();
    if nout >= 3 {
        let (mut min_i, mut max_i, mut lt) = (0usize, 0usize, 0usize);
        for i in 1..nout {
            let v = idx[i];
            lt += (idx[i - 1] < v) as usize;
            if lt > 1 && lt + 2 <= i {
                break;
            }
            if v < idx[min_i] {
                min_i = i;
            }
            if v > idx[max_i] {
                max_i = i;
            }
        }
        let mmdist = max_i.abs_diff(min_i);
        if (mmdist == 1 || mmdist == nout - 1) && (lt <= 1 || lt + 2 >= nout) {
            let ascending = (max_i + 1) % nout == min_i;
            let i0 = if ascending { min_i } else { max_i };
            if i0 > 0 {
                let mut shifted = Vec::with_capacity(nout);
                let mut j = i0;
                let mut ok = true;
                for i in 0..nout {
                    let cur = idx[j];
                    let next_j = if j + 1 < nout { j + 1 } else { 0 };
                    shifted.push(cur);
                    if i < nout - 1 && ascending != (cur < idx[next_j]) {
                        ok = false;
                        break;
                    }
                    j = next_j;
                }
                if ok {
                    idx = shifted;
                }
            }
        }
    }
    idx.iter().map(|&i| points[i]).collect()
}

/// `cv::RotatedRect`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RotatedRect {
    pub center: (f32, f32),
    pub size: (f32, f32),
    /// Degrees, in [-90, 0).
    pub angle: f32,
}

/// `a * b + c * d` with the first product fused (GCC contraction).
#[inline]
fn dot2(a: f32, b: f32, c: f32, d: f32) -> f32 {
    a.mul_add(b, c * d)
}

/// `rotatingCalipers(CALIPERS_MINAREARECT)` on a counter-clockwise hull:
/// corner, vector 1, vector 2.
fn rotating_calipers(points: &[(f32, f32)]) -> [(f32, f32); 3] {
    let n = points.len();
    let mut vect = vec![(0f32, 0f32); n];
    let mut inv_len = vec![0f32; n];
    let (mut left, mut bottom, mut right, mut top) = (0usize, 0usize, 0usize, 0usize);
    let mut pt0 = points[0];
    let (mut left_x, mut right_x, mut top_y, mut bottom_y) = (pt0.0, pt0.0, pt0.1, pt0.1);
    for i in 0..n {
        if pt0.0 < left_x {
            left_x = pt0.0;
            left = i;
        }
        if pt0.0 > right_x {
            right_x = pt0.0;
            right = i;
        }
        if pt0.1 > top_y {
            top_y = pt0.1;
            top = i;
        }
        if pt0.1 < bottom_y {
            bottom_y = pt0.1;
            bottom = i;
        }
        let pt = points[if i + 1 < n { i + 1 } else { 0 }];
        let dx = pt.0 as f64 - pt0.0 as f64;
        let dy = pt.1 as f64 - pt0.1 as f64;
        vect[i] = (dx as f32, dy as f32);
        inv_len[i] = (1.0 / dx.mul_add(dx, dy * dy).sqrt()) as f32;
        pt0 = pt;
    }
    let mut seq = [bottom, right, top, left];
    let mut minarea = f32::MAX;
    let mut best = (0usize, 0f32, 0f32, 0f32, 0f32, 0usize);
    // firstVecIsRight(v1, v2): rotate90CW(v1) . v2 < 0
    let first_right = |v1: (f32, f32), v2: (f32, f32)| dot2(v1.1, v2.0, -v1.0, v2.1) < 0.0;
    for _ in 0..n {
        let rot = [
            vect[seq[0]],
            (vect[seq[1]].1, -vect[seq[1]].0),
            (-vect[seq[2]].0, -vect[seq[2]].1),
            (-vect[seq[3]].1, vect[seq[3]].0),
        ];
        let mut main = 0;
        for i in 1..4 {
            if first_right(rot[i], rot[main]) {
                main = i;
            }
        }
        let pi = seq[main];
        let lead_x = vect[pi].0 * inv_len[pi];
        let lead_y = vect[pi].1 * inv_len[pi];
        let (base_a, base_b) = match main {
            0 => (lead_x, lead_y),
            1 => (lead_y, -lead_x),
            2 => (-lead_x, -lead_y),
            _ => (-lead_y, lead_x),
        };
        seq[main] += 1;
        if seq[main] == n {
            seq[main] = 0;
        }
        let dx = points[seq[1]].0 - points[seq[3]].0;
        let dy = points[seq[1]].1 - points[seq[3]].1;
        let width = dot2(dx, base_a, dy, base_b);
        let dx = points[seq[2]].0 - points[seq[0]].0;
        let dy = points[seq[2]].1 - points[seq[0]].1;
        let height = dot2(-dx, base_b, dy, base_a);
        let area = width * height;
        if area <= minarea {
            minarea = area;
            best = (seq[3], base_a, width, base_b, height, seq[0]);
        }
    }
    let (l, a1, w, b1, h, b) = best;
    let (a2, b2) = (-b1, a1);
    let c1 = dot2(a1, points[l].0, points[l].1, b1);
    let c2 = dot2(a2, points[b].0, points[b].1, b2);
    let idet = 1.0 / a1.mul_add(b2, -(a2 * b1));
    let px = c1.mul_add(b2, -(c2 * b1)) * idet;
    let py = a1.mul_add(c2, -(a2 * c1)) * idet;
    [(px, py), (a1 * w, b1 * w), (a2 * h, b2 * h)]
}

/// `cv2.minAreaRect(pts)` for an integer point set.
pub fn min_area_rect(points: &[Point]) -> RotatedRect {
    let hull = to_f32(&convex_hull(points));
    let mut angle = -std::f64::consts::PI / 2.0;
    let mut r = RotatedRect {
        center: (0.0, 0.0),
        size: (0.0, 0.0),
        angle: 0.0,
    };
    let n = hull.len();
    if n > 2 {
        let out = rotating_calipers(&hull);
        r.center = (
            (out[1].0 + out[2].0).mul_add(0.5, out[0].0),
            (out[1].1 + out[2].1).mul_add(0.5, out[0].1),
        );
        let len = |v: (f32, f32)| {
            (v.0 as f64)
                .mul_add(v.0 as f64, v.1 as f64 * v.1 as f64)
                .sqrt() as f32
        };
        r.size = (len(out[2]), len(out[1]));
        if out[1].0 == 0.0 && out[1].1 > 0.0 {
            r.size = (r.size.1, r.size.0);
        } else {
            angle = -(out[1].0 as f64).atan2(out[1].1 as f64);
        }
    } else if n == 2 {
        r.center = ((hull[0].0 + hull[1].0) * 0.5, (hull[0].1 + hull[1].1) * 0.5);
        let dx = hull[0].0 as f64 - hull[1].0 as f64;
        let dy = hull[0].1 as f64 - hull[1].1 as f64;
        r.size = (0.0, dx.mul_add(dx, dy * dy).sqrt() as f32);
        if dx == 0.0 {
            r.size = (r.size.1, r.size.0);
        } else if dy < 0.0 {
            angle = dy.atan2(dx);
            r.size = (r.size.1, r.size.0);
        } else if dy > 0.0 {
            angle = -dx.atan2(dy);
        }
    } else if n == 1 {
        r.center = hull[0];
    }
    r.angle = (angle * 180.0 / std::f64::consts::PI) as f32;
    r
}

/// `cv2.boxPoints(rect)` (`RotatedRect::points`: `center +- a*h` rounded,
/// then `+- b*w` fused, as GCC compiled it in the reference build).
pub fn box_points(r: &RotatedRect) -> [(f32, f32); 4] {
    let ang = r.angle as f64 * std::f64::consts::PI / 180.0;
    let b = ang.cos() as f32 * 0.5;
    let a = ang.sin() as f32 * 0.5;
    let (w, h) = r.size;
    let (cx, cy) = r.center;
    let (ah, bh) = (a * h, b * h);
    [
        ((-b).mul_add(w, cx - ah), (-a).mul_add(w, cy + bh)),
        ((-b).mul_add(w, cx + ah), (-a).mul_add(w, cy - bh)),
        (b.mul_add(w, cx + ah), a.mul_add(w, cy - bh)),
        (b.mul_add(w, cx - ah), a.mul_add(w, cy + bh)),
    ]
}
