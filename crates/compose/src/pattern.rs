//! Procedural patterns and projections.

use texturecraft_core::{Decal, DecalFlip, Procedural, ProceduralKind, Projection};
use texturecraft_geom::glam::{Vec2, Vec3};
use texturecraft_texel::noise;

use crate::pattern_context::PatternContext;

/// Smaller pattern scales are read as this: a pattern stretched over ten thousand repeats of its projection
const MIN_PATTERN_SCALE: f32 = 1e-4;

/// Where a texel is, for pattern evaluation.
#[derive(Clone, Copy, Debug)]
pub struct Coords {
    pub uv: Vec2,
    /// World position divided by the scene radius (scale-independent).
    pub pos: Vec3,
    pub nrm: Vec3,
}

/// Raw pattern value at a 3D point, 0..1.
fn raw(kind: ProceduralKind, p: Vec3, seed: u32) -> f32 {
    let a = p.to_array();
    match kind {
        ProceduralKind::Clouds => noise::fbm(a, 6, seed, noise::value),
        ProceduralKind::Perlin => noise::fbm(a, 4, seed, noise::perlin),
        ProceduralKind::Noise => noise::value((p * 16.0).to_array(), seed),
        ProceduralKind::Cells => noise::worley(a, seed),
        ProceduralKind::Grunge => {
            let n = noise::fbm(a, 6, seed, noise::perlin) * 0.7 + noise::fbm((p * 3.7).to_array(), 3, seed ^ 0x55, noise::value) * 0.3;
            ((n - 0.45) * 4.0 + 0.5).clamp(0.0, 1.0)
        }
        ProceduralKind::Scratches => {
            // Stretch noise along one axis to make thin streaks
            let q = Vec3::new(p.x * 0.15, p.y * 6.0, p.z * 6.0);
            let n = noise::perlin(q.to_array(), seed);
            let line = 1.0 - ((n - 0.5).abs() * 40.0).min(1.0);
            let gate = noise::value((p * 0.8).to_array(), seed ^ 0x99);
            line * ((gate - 0.45) * 4.0).clamp(0.0, 1.0)
        }
        ProceduralKind::Checker => {
            let c = p.floor();
            if (c.x + c.y + c.z).rem_euclid(2.0) < 1.0 { 1.0 } else { 0.0 }
        }
        ProceduralKind::Stripes => {
            let t = p.x.rem_euclid(1.0);
            if t < 0.5 { 1.0 } else { 0.0 }
        }
        ProceduralKind::Gradient => p.x.rem_euclid(1.0),
        ProceduralKind::Wood => {
            // Warp ring boundaries and add fine fibers along X
            let warp = noise::fbm([p.x * 0.18, p.y * 0.6, p.z], 3, seed, noise::perlin);
            let t = ((p.y + (warp - 0.5) * 0.9) * 5.0).rem_euclid(1.0);
            let ring = 1.0 - smoothstep(0.45, 1.0, t);
            let fiber = noise::value([p.x * 2.0, p.y * 40.0, p.z], seed ^ 0x3C);
            let tone = noise::fbm([p.x * 0.6, p.y * 3.0, p.z], 2, seed ^ 0x71, noise::value);
            (ring * 0.6 + fiber * 0.25 + tone * 0.15).clamp(0.0, 1.0)
        }
        ProceduralKind::Bricks => {
            // Keep per-brick tone near its high value so mortar stays distinct
            let row = (p.y * 2.0).floor();
            let x = p.x + if (row as i32).rem_euclid(2) == 1 { 0.5 } else { 0.0 };
            let (col, fx, fy) = (x.floor(), x.rem_euclid(1.0), (p.y * 2.0).rem_euclid(1.0));
            let d = fx.min(1.0 - fx).min(fy.min(1.0 - fy) * 0.5);
            let tone = noise::cell_hash(col as i32, row as i32, 0, seed) * 0.6 + noise::value((p * 9.0).to_array(), seed) * 0.4;
            smoothstep(0.02, 0.045, d) * (0.85 + 0.15 * tone)
        }
        ProceduralKind::Tiles => {
            let (c, f) = (p.floor(), p - p.floor());
            let d = f.x.min(1.0 - f.x).min(f.y.min(1.0 - f.y));
            smoothstep(0.015, 0.035, d) * (0.9 + 0.1 * noise::cell_hash(c.x as i32, c.y as i32, 0, seed))
        }
        ProceduralKind::Weave | ProceduralKind::Twill => {
            // Alternate top threads for plain weave or step them into twill ribs
            let c = (p * 2.0).floor();
            let f = p * 2.0 - c;
            let pi = std::f32::consts::PI;
            let twill = kind == ProceduralKind::Twill;
            let (period, warp_cells) = if twill { (4, 2) } else { (2, 1) };
            let warp_on_top = (c.x as i32).wrapping_add(c.y as i32).rem_euclid(period) < warp_cells;
            let (across, along) = if warp_on_top { (f.y, f.x) } else { (f.x, f.y) };
            let thread = (pi * across).sin().max(0.0).sqrt() * (0.8 + 0.2 * (pi * along).sin());
            let tone = if twill && !warp_on_top { 0.55 } else { 1.0 };
            let fiber = noise::value([p.x * 40.0, p.y * 40.0, p.z], seed);
            (thread * tone * (0.88 + 0.12 * fiber)).clamp(0.0, 1.0)
        }
        ProceduralKind::Leather => {
            let w = noise::perlin((p * 0.7).to_array(), seed ^ 0x2D) - 0.5;
            let (f1, f2, _) = noise::cellular([p.x + w * 0.4, p.y - w * 0.4, p.z], seed);
            let crease = smoothstep(0.0, 0.16, f2 - f1);
            let pebble = 1.0 - f1 * 0.5;
            let fine = noise::value((p * 7.0).to_array(), seed ^ 0x13);
            (crease * pebble * (0.85 + 0.15 * fine)).clamp(0.0, 1.0)
        }
        ProceduralKind::Hammered => {
            let dent = |q: Vec3, s: u32| {
                let (f1, _, _) = noise::cellular(q.to_array(), s);
                (f1 * 1.25).min(1.0).powi(2)
            };
            (dent(p, seed) * 0.75 + dent(p * 2.3 + Vec3::splat(5.1), seed ^ 0x5A) * 0.25).clamp(0.0, 1.0)
        }
        ProceduralKind::Brushed => {
            let a = noise::value([p.x * 1.5, p.y * 60.0, p.z], seed);
            let b = noise::value([p.x * 0.4, p.y * 18.0, p.z], seed ^ 0x77);
            (a * 0.6 + b * 0.4).clamp(0.0, 1.0)
        }
        ProceduralKind::Marble => {
            let turb = noise::fbm(a, 5, seed, noise::perlin);
            let veins = ((p.x + p.y * 0.5) * 2.5 + turb * 9.0).sin().abs().powf(0.3);
            let cloud = noise::fbm((p * 2.0).to_array(), 3, seed ^ 0x19, noise::value);
            (veins * 0.85 + cloud * 0.15).clamp(0.0, 1.0)
        }
        ProceduralKind::Voronoi => noise::cellular(a, seed).2,
        ProceduralKind::Streaks => {
            // Offset column grids make sparse drips that fade downwards
            let drip = |x: f32, salt: u32| {
                let cx = x * 6.0;
                let col = cx.floor() as i32;
                let h = |k: i32| noise::cell_hash(col, k, salt as i32, seed);
                let width = 0.08 + 0.25 * h(0);
                let across = 1.0 - smoothstep(width * 0.3, width, (cx - cx.floor() - 0.5 - (h(1) - 0.5) * 0.5).abs());
                let yy = p.y * (0.35 + 0.4 * h(2)) + h(3) * 7.0;
                let t = yy - yy.floor();
                let on = if noise::cell_hash(col, yy.floor() as i32, 3 + salt as i32, seed) > 0.5 { 1.0 } else { 0.0 };
                across * on * t.powf(1.6) * (1.0 - smoothstep(0.9, 1.0, t))
            };
            let fine = noise::value([p.x * 37.0, p.y * 1.3, p.z], seed ^ 0x2B);
            (drip(p.x, 0).max(drip(p.x + 0.37, 1) * 0.8) * (0.6 + 0.4 * fine)).clamp(0.0, 1.0)
        }
        ProceduralKind::Smudges => {
            // Blend ridged elliptical prints with soft smudges
            let w = noise::perlin((p * 1.3).to_array(), seed ^ 0x61) - 0.5;
            let q = Vec3::new(p.x * 1.15 + w * 0.3, p.y * 0.85 + w * 0.3, p.z);
            let (f1, _, id) = noise::cellular(q.to_array(), seed);
            let print = (1.0 - smoothstep(0.18 + 0.1 * id, 0.3 + 0.12 * id, f1)) * if id > 0.4 { 1.0 } else { 0.0 };
            let ridges = 0.5 + 0.5 * (f1 * 190.0 + w * 12.0).sin();
            let smudge = ((noise::fbm((p * 1.7).to_array(), 4, seed ^ 0x33, noise::value) - 0.55) * 3.0).clamp(0.0, 1.0);
            (print * ridges * 0.9 + smudge * 0.4).clamp(0.0, 1.0)
        }
        ProceduralKind::Pores => {
            let (f1, _, _) = noise::cellular(a, seed);
            let (g1, _, _) = noise::cellular((p * 2.7 + Vec3::splat(13.0)).to_array(), seed ^ 0x6B);
            smoothstep(0.08, 0.16, f1).min(smoothstep(0.06, 0.12, g1))
        }
        ProceduralKind::TileGenerator => crate::tile_pattern::value(&crate::tile_pattern::DEFAULT, p.truncate(), seed, None),
        ProceduralKind::Chips => {
            let (f1, _, _) = noise::cellular((p * 2.0).to_array(), seed ^ 0x45);
            let n = noise::fbm(a, 5, seed, noise::perlin) + (f1 - 0.5) * 0.25;
            smoothstep(0.55, 0.58, n)
        }
    }
}

/// `values` raised to a triplanar `sharpness` (at least 1): whole sharpnesses up to 16 (the usual
/// ones) by repeated squaring, a few multiplies instead of three `powf` calls per texel.
#[inline]
fn sharpen(values: Vec3, sharpness: f32) -> Vec3 {
    const MAX_INTEGER_SHARPNESS: f32 = 16.0;
    let exponent = sharpness.max(1.0);
    if exponent > MAX_INTEGER_SHARPNESS || exponent.fract() != 0.0 {
        return values.powf(exponent);
    }

    let (mut remaining, mut base, mut result) = (exponent as u32, values, Vec3::ONE);
    while remaining > 0 {
        if remaining & 1 == 1 {
            result *= base;
        }
        base *= base;
        remaining >>= 1;
    }

    result
}

/// A triplanar plane whose weight, times the gain applied after blending, stays under this is
/// not evaluated: even two skipped planes change a value by less than 1/65536, under the u16
/// step layers are stored at. With sharp blends most texels need one plane, not three.
const TRIPLANAR_SKIP: f32 = 1.0 / 131_072.0;

/// Normalized triplanar blend weights (YZ, XZ, XY planes) for normal `nrm`; planes too light
/// to change a 0..1 value scaled by `gain` afterwards weigh 0 (see [`TRIPLANAR_SKIP`]).
fn triplanar_weights(nrm: Vec3, sharpness: f32, gain: f32) -> Vec3 {
    let weights = sharpen(nrm.abs(), sharpness);
    let weights = weights / (weights.x + weights.y + weights.z).max(1e-6);
    let skip_threshold = TRIPLANAR_SKIP / gain.max(1.0);
    let retained = Vec3::select(weights.cmplt(Vec3::splat(skip_threshold)), Vec3::ZERO, weights);

    // Preserve the blend if thresholding would discard every plane
    if retained == Vec3::ZERO { weights } else { retained }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn shape(pr: &Procedural, v: f32) -> f32 {
    let v = ((v - 0.5) * pr.contrast.max(0.0) + 0.5 + pr.balance).clamp(0.0, 1.0);
    if pr.invert { 1.0 - v } else { v }
}

/// Evaluate a procedural through a projection, at full resolution and without images
pub fn procedural(pattern: &Procedural, projection: &Projection, coords: &Coords) -> f32 {
    procedural_in(pattern, projection, coords, &PatternContext::default())
}

/// Evaluate a procedural through a projection with what `context` gives it: its resolution and the images it reads
pub fn procedural_in(pattern: &Procedural, projection: &Projection, coords: &Coords, context: &PatternContext) -> f32 {
    let pattern_scale = pattern.scale.max(MIN_PATTERN_SCALE);
    let sample = |point: Vec3| plane_value(pattern, point, context);
    let value = match projection {
        Projection::Uv { scale, rotation, offset } => {
            let (sn, cs) = rotation.to_radians().sin_cos();
            let uv = Vec2::new(coords.uv.x * cs - coords.uv.y * sn, coords.uv.x * sn + coords.uv.y * cs) * Vec2::from(*scale) + Vec2::from(*offset);

            // Evaluate two-dimensional patterns on a slice of the volume
            sample((uv * pattern_scale).extend(0.0))
        }
        Projection::Triplanar { scale, sharpness, offset } => {
            let p = coords.pos * *scale * pattern_scale + Vec3::from(*offset);

            // Shape scales the blend by the contrast
            let w = triplanar_weights(coords.nrm, *sharpness, pattern.contrast);
            let planes = [(w.x, Vec3::new(p.z, p.y, 0.0)), (w.y, Vec3::new(p.x, p.z, 0.0)), (w.z, Vec3::new(p.x, p.y, 0.0))];
            planes.into_iter().filter(|(w, _)| *w > 0.0).map(|(w, q)| sample(q) * w).sum()
        }
        Projection::Planar { scale, axis, offset } => {
            let p = coords.pos * *scale * pattern_scale + Vec3::from(*offset);
            let q = match axis {
                0 => Vec3::new(p.z, p.y, 0.0),
                1 => Vec3::new(p.x, p.z, 0.0),
                _ => Vec3::new(p.x, p.y, 0.0),
            };
            sample(q)
        }
        Projection::Spherical { scale } => {
            let d = coords.pos.try_normalize().unwrap_or(Vec3::Y);
            let u = d.z.atan2(d.x) / std::f32::consts::TAU + 0.5;
            let v = d.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
            sample(Vec3::new(u * 4.0, v * 2.0, 0.0) * *scale * pattern_scale)
        }
        Projection::Cylindrical { scale, axis } => {
            let p = coords.pos;
            let (a, h, b) = match axis {
                0 => (p.y, p.x, p.z),
                2 => (p.x, p.z, p.y),
                _ => (p.x, p.y, p.z),
            };
            let u = b.atan2(a) / std::f32::consts::TAU + 0.5;
            sample(Vec3::new(u * 4.0, h, 0.0) * *scale * pattern_scale)
        }

        // The decal already supplied coordinates in its local frame
        Projection::Decal(_) => sample((coords.uv * pattern_scale).extend(0.0)),
    };

    shape(pattern, value)
}

/// The texels per side `pattern`'s resolution draws it at for a UV tile of `uv_tile` texels per side in a texture set of `texture_set`, when that is coarser than the UV tile (a finer image changes nothing: patterns are evaluated per texel)
pub fn pattern_pixels(pattern: &Procedural, uv_tile: u32, texture_set: u32) -> Option<u32> {
    if !pattern.resolution.can_reduce() {
        return None;
    }

    let size = pattern.resolution.output_size(uv_tile, texture_set);
    (size < uv_tile).then_some(size)
}

/// The pattern at `point` of its plane, read from an image of `context.pixels` texels per repeat (bilinear) when given
fn plane_value(pattern: &Procedural, point: Vec3, context: &PatternContext) -> f32 {
    let cells = context.pixels.map(|pixels| pixels as f32 / pattern.scale.max(MIN_PATTERN_SCALE)).filter(|cells| cells.is_finite() && *cells > 0.0);
    let Some(cells) = cells else { return pattern_value(pattern, point, context) };

    let grid = point.truncate() * cells - 0.5;
    let base = grid.floor();
    let fraction = grid - base;
    let at = |dx: f32, dy: f32| pattern_value(pattern, ((base + Vec2::new(dx, dy) + 0.5) / cells).extend(point.z), context);
    let top_left = at(0.0, 0.0);
    let bottom_left = at(0.0, 1.0);
    let top = top_left + (at(1.0, 0.0) - top_left) * fraction.x;
    let bottom = bottom_left + (at(1.0, 1.0) - bottom_left) * fraction.x;
    top + (bottom - top) * fraction.y
}

/// The names of the images `pattern` reads from [`PatternContext::images`] (none but a Tile Generator's)
pub(crate) fn image_names(pattern: &Procedural) -> impl Iterator<Item = &str> {
    pattern.tile.as_deref().filter(|_| pattern.kind == ProceduralKind::TileGenerator).into_iter().flat_map(crate::tile_pattern::image_names)
}

/// The pattern at `point` of its plane, exactly
fn pattern_value(pattern: &Procedural, point: Vec3, context: &PatternContext) -> f32 {
    match (&pattern.tile, pattern.kind) {
        (Some(tile), ProceduralKind::TileGenerator) => crate::tile_pattern::value(tile, point.truncate(), pattern.seed, context.images),
        _ => raw(pattern.kind, point, pattern.seed),
    }
}

/// Texture coordinates (and blend weights) at which an image is sampled for a projection
/// (unused slots weigh 0; no allocation per texel).
pub fn image_samples(projection: &Projection, coords: &Coords) -> [(Vec2, f32); 3] {
    let one = |uv: Vec2| [(uv, 1.0), (Vec2::ZERO, 0.0), (Vec2::ZERO, 0.0)];
    match projection {
        Projection::Uv { scale, rotation, offset } => {
            let (sn, cs) = rotation.to_radians().sin_cos();
            let uv = Vec2::new(coords.uv.x * cs - coords.uv.y * sn, coords.uv.x * sn + coords.uv.y * cs) * Vec2::from(*scale) + Vec2::from(*offset);
            one(uv)
        }
        Projection::Triplanar { scale, sharpness, offset } => {
            let p = coords.pos * *scale + Vec3::from(*offset);

            // Float images can exceed 1, so even a tiny plane weight can carry visible detail
            let weights = coords.nrm.abs().powf(sharpness.max(1.0));
            let weights = weights / (weights.x + weights.y + weights.z).max(1e-6);
            [(Vec2::new(p.z, -p.y), weights.x), (Vec2::new(p.x, p.z), weights.y), (Vec2::new(p.x, -p.y), weights.z)]
        }
        Projection::Planar { scale, axis, offset } => {
            let p = coords.pos * *scale + Vec3::from(*offset);
            let q = match axis {
                0 => Vec2::new(p.z, -p.y),
                1 => Vec2::new(p.x, p.z),
                _ => Vec2::new(p.x, -p.y),
            };
            one(q * 0.5 + Vec2::splat(0.5))
        }
        Projection::Spherical { scale } => {
            let d = coords.pos.try_normalize().unwrap_or(Vec3::Y);
            let u = d.z.atan2(d.x) / std::f32::consts::TAU + 0.5;
            let v = d.y.clamp(-1.0, 1.0).acos() / std::f32::consts::PI;
            one(Vec2::new(u, v) * *scale)
        }
        Projection::Cylindrical { scale, axis } => {
            let p = coords.pos;
            let (a, h, b) = match axis {
                0 => (p.y, p.x, p.z),
                2 => (p.x, p.z, p.y),
                _ => (p.x, p.y, p.z),
            };
            let u = b.atan2(a) / std::f32::consts::TAU + 0.5;
            one(Vec2::new(u, 0.5 - h * 0.5) * *scale)
        }
        Projection::Decal(_) => one(coords.uv),
    }
}

/// A decal's frame: origin (center plus offset), unit right, up and normal (None when degenerate).
pub fn decal_axes(d: &Decal) -> Option<(Vec3, Vec3, Vec3, Vec3)> {
    if !d.size.iter().all(|v| v.is_finite() && *v > 0.0) {
        return None;
    }
    let n = Vec3::from(d.normal).try_normalize()?;
    let up0 = (Vec3::from(d.up) - n * n.dot(Vec3::from(d.up))).try_normalize().unwrap_or_else(|| n.any_orthonormal_vector());
    let right0 = up0.cross(n);
    let origin = Vec3::from(d.center) + right0 * d.offset[0] + up0 * d.offset[1];
    let (sn, cs) = d.rotation.to_radians().sin_cos();
    let right = right0 * cs + up0 * sn;
    let up = up0 * cs - right0 * sn;
    origin.is_finite().then_some((origin, right, up, n))
}

/// The decal's corners in world space (top left, top right, bottom right, bottom left).
pub fn decal_corners(d: &Decal) -> Option<[Vec3; 4]> {
    let (o, r, u, _) = decal_axes(d)?;
    let (hw, hh) = (d.size[0] * 0.5, d.size[1] * 0.5);
    Some([o - r * hw + u * hh, o + r * hw + u * hh, o + r * hw - u * hh, o - r * hw - u * hh])
}

/// Where world point `p` with surface normal `nrm` falls in the decal: texture coordinates
/// (v down) and how much the decal covers it, None outside the frame, behind it, beyond its
/// depth or on faces turned away. Coverage fades on grazing faces and near the depth limit.
pub fn decal_uv(d: &Decal, p: Vec3, nrm: Vec3) -> Option<(Vec2, f32)> {
    let (o, r, u, n) = decal_axes(d)?;
    let facing = nrm.dot(n);
    let q = p - o;
    let (lx, ly, lz) = (q.dot(r) / d.size[0].max(1e-6), q.dot(u) / d.size[1].max(1e-6), q.dot(n).abs());
    let depth = d.depth.max(1e-6);
    if facing <= 0.0 || lx.abs() > 0.5 || ly.abs() > 0.5 || lz > depth || !(lx.is_finite() && ly.is_finite()) {
        return None;
    }

    let w = smoothstep(0.0, 0.3, facing) * (1.0 - smoothstep(0.75 * depth, depth, lz));
    Some((flip_uv(Vec2::new(lx + 0.5, 0.5 - ly), d.flip), w))
}

/// `uv` with the image mirrored left to right (`flip[0]`) and/or top to bottom (`flip[1]`) inside the frame.
fn flip_uv(uv: Vec2, flip: [bool; 2]) -> Vec2 {
    Vec2::new(if flip[0] { 1.0 - uv.x } else { uv.x }, if flip[1] { 1.0 - uv.y } else { uv.y })
}

/// [`decal_uv`] for a decal with a mirrored copy: whichever of the decal and its copy covers the
/// point more. The copy is the decal reflected through the plane across `center` (the mesh
/// center), so a point is looked up as its reflection (the plane is `offset` away from the center) and the image flipped as asked.
pub fn decal_uv_symmetric(d: &Decal, p: Vec3, nrm: Vec3, center: Vec3) -> Option<(Vec2, f32)> {
    let direct = decal_uv(d, p, nrm);
    let Some(symmetry) = d.symmetry else { return direct };
    let axis = usize::from(symmetry.axis.min(2));
    let (mut mirrored_position, mut mirrored_normal) = (p, nrm);
    mirrored_position[axis] = 2.0 * (center[axis] + symmetry.offset) - p[axis];
    mirrored_normal[axis] = -nrm[axis];
    let mirrored = decal_uv(d, mirrored_position, mirrored_normal).map(|(uv, w)| {
        let uv = match symmetry.flip {
            DecalFlip::None => uv,
            DecalFlip::U => Vec2::new(1.0 - uv.x, uv.y),
            DecalFlip::V => Vec2::new(uv.x, 1.0 - uv.y),
        };
        (uv, w)
    });
    match (direct, mirrored) {
        (Some(a), Some(b)) => Some(if b.1 > a.1 { b } else { a }),
        (a, b) => a.or(b),
    }
}

/// World-space noise used by generators (seamless across UV islands).
pub fn world_noise(pos: Vec3, scale: f32, seed: u32) -> f32 {
    noise::fbm((pos * scale.max(1e-3)).to_array(), 4, seed, noise::perlin)
}

#[cfg(test)]
mod tests {
    use super::*;
    use texturecraft_core::{OutputResolution, ResolutionMode};

    #[test]
    fn a_pattern_is_drawn_coarser_only_below_the_uv_tiles_resolution() {
        let with = |resolution: OutputResolution| pattern_pixels(&Procedural { resolution, ..Procedural::new(ProceduralKind::Clouds) }, 1024, 2048);
        assert_eq!(with(OutputResolution::default()), None, "by default a pattern is evaluated per texel");
        assert_eq!(with(OutputResolution { factor: -1, ..OutputResolution::default() }), Some(512));
        assert_eq!(with(OutputResolution { mode: ResolutionMode::Custom, custom: 256, factor: 0 }), Some(256));
        assert_eq!(with(OutputResolution { mode: ResolutionMode::Custom, custom: 2048, factor: 0 }), None, "a finer image than the UV tile changes nothing");
    }

    #[test]
    fn a_coarser_pattern_reads_its_cells_bilinearly() {
        let pattern = Procedural { scale: 1.0, ..Procedural::new(ProceduralKind::Clouds) };
        let coarse = PatternContext { pixels: Some(4), ..PatternContext::default() };
        let at = |x: f32, context: &PatternContext| {
            procedural_in(&pattern, &Projection::default(), &Coords { uv: Vec2::new(x, 0.375), pos: Vec3::ZERO, nrm: Vec3::Y }, context)
        };
        let exact = PatternContext::default();

        // Coarse reads are exact at cell centers and blend between them
        assert!((at(0.375, &coarse) - at(0.375, &exact)).abs() < 1e-5);
        let blended = (at(0.125, &exact) + at(0.375, &exact)) * 0.5;
        assert!((at(0.25, &coarse) - blended).abs() < 1e-5, "{} vs {blended}", at(0.25, &coarse));
        assert!((at(0.25, &exact) - blended).abs() > 1e-3, "the exact pattern is not the blend, so the coarse read shows");
    }

    #[test]
    fn every_pattern_and_projection_stays_in_range() {
        let projs = [
            Projection::default(),
            Projection::Triplanar { scale: 1.0, sharpness: 4.0, offset: [0.0; 3] },
            Projection::Planar { scale: 1.0, axis: 1, offset: [0.0; 3] },
            Projection::Spherical { scale: 1.0 },
            Projection::Cylindrical { scale: 1.0, axis: 1 },
            Projection::Decal(Decal {
                center: [0.0; 3],
                normal: [0.0, 0.0, 1.0],
                up: [0.0, 1.0, 0.0],
                size: [1.0; 2],
                rotation: 0.0,
                offset: [0.0; 2],
                depth: 1.0,
                symmetry: None,
                flip: [false; 2],
            }),
        ];
        for k in ProceduralKind::ALL {
            let pr = Procedural::new(k);
            for proj in &projs {
                for i in 0..50 {
                    let t = i as f32 * 0.13;
                    let c = Coords {
                        uv: Vec2::new(t.fract(), (t * 0.7).fract()),
                        pos: Vec3::new(t.sin(), t.cos(), t * 0.1),
                        nrm: Vec3::new(0.3, 0.9, 0.1).normalize(),
                    };
                    let v = procedural(&pr, proj, &c);
                    assert!((0.0..=1.0).contains(&v), "{k:?} {proj:?} {v}");
                }
            }
        }
    }

    /// Skipping light triplanar planes stays within a u16 step of evaluating all three, at any
    /// contrast; near an axis it does skip.
    #[test]
    fn triplanar_plane_skipping_stays_under_one_u16_step() {
        let full = |pattern: &Procedural, sharpness: f32, coords: &Coords| {
            let position = coords.pos * pattern.scale;
            let weights = coords.nrm.abs().powf(sharpness);
            let weights = weights / (weights.x + weights.y + weights.z);
            shape(
                pattern,
                raw(pattern.kind, Vec3::new(position.z, position.y, 0.0), pattern.seed) * weights.x
                    + raw(pattern.kind, Vec3::new(position.x, position.z, 0.0), pattern.seed) * weights.y
                    + raw(pattern.kind, Vec3::new(position.x, position.y, 0.0), pattern.seed) * weights.z,
            )
        };
        let mut largest_difference = 0.0f32;
        for kind in ProceduralKind::ALL {
            for contrast in [1.0, 4.0, 16.0] {
                let pattern = Procedural { contrast, ..Procedural::new(kind) };
                for sharpness in [1.0, 4.0, 8.0, 16.0] {
                    let projection = Projection::Triplanar { scale: 1.0, sharpness, offset: [0.0; 3] };
                    for index in 0..400 {
                        // Cover axis aligned and oblique normals at varied positions
                        let position = index as f32 / 400.0;
                        let normal = Vec3::new(position * 1.3, 1.0, (position * 7.0).sin() * position).normalize();
                        let coords = Coords { uv: Vec2::ZERO, pos: Vec3::new((position * 11.0).sin(), (position * 5.0).cos(), position * 3.0), nrm: normal };
                        largest_difference = largest_difference.max((procedural(&pattern, &projection, &coords) - full(&pattern, sharpness, &coords)).abs());
                    }
                }
            }
        }

        assert!(largest_difference < 1.0 / 65535.0, "worst difference {largest_difference}");
        let near_axis = triplanar_weights(Vec3::new(0.2, 1.0, 0.1).normalize(), 8.0, 1.0);
        assert_eq!((near_axis.x, near_axis.z), (0.0, 0.0), "{near_axis}");

        // An oblique normal skips no plane
        let oblique = triplanar_weights(Vec3::ONE.normalize(), 8.0, 16.0);
        assert!((oblique - Vec3::splat(1.0 / 3.0)).abs().max_element() < 1e-6, "{oblique}");

        // Tiny weights must survive together when their sum exceeds a u16 step
        for sharpness in [47.0, 64.0] {
            let diagonal = triplanar_weights(Vec3::ONE.normalize(), sharpness, 1.0);
            assert!(diagonal.min_element() > 0.0, "sharpness {sharpness}: {diagonal}");
        }
    }

    #[test]
    fn triplanar_image_sampling_preserves_bright_float_pixels() {
        let image = crate::ImageRes { width: 2, height: 2, rgba: vec![[0.0; 4], [0.0; 4], [0.0; 4], [100_000.0; 4]], srgb: false };
        let coords = Coords { uv: Vec2::ZERO, pos: Vec3::new(0.25, 0.25, 0.75), nrm: Vec3::new(0.2, 1.0, 0.1).normalize() };
        let projection = Projection::Triplanar { scale: 1.0, sharpness: 8.0, offset: [0.0; 3] };
        let weights = coords.nrm.abs().powf(8.0);
        let expected = 100_000.0 * weights.x / (weights.x + weights.y + weights.z);

        let actual: f32 =
            image_samples(&projection, &coords).into_iter().filter(|(_, weight)| *weight > 0.0).map(|(uv, weight)| image.sample(uv.x, uv.y)[0] * weight).sum();

        assert!(expected > 0.25, "the light plane must make a visible contribution");
        assert!((actual - expected).abs() < 1.0 / 65535.0, "expected {expected}, got {actual}");
    }

    /// Repeated squaring agrees with `powf` to a few ulps for whole sharpnesses, and falls back
    /// to `powf` for the rest.
    #[test]
    fn sharpen_matches_powf() {
        for index in 0..=100 {
            let values = Vec3::new(index as f32 / 100.0, 1.0 - index as f32 / 100.0, (index as f32 * 0.37).fract());
            for sharpness in [0.0, 1.0, 2.0, 3.0, 4.0, 7.0, 8.0, 16.0] {
                let (actual, expected) = (sharpen(values, sharpness), values.powf(sharpness.max(1.0)));
                assert!((actual - expected).abs().max_element() <= 1e-6, "{values} ^ {sharpness}: {actual} vs {expected}");
            }

            for sharpness in [2.5, 17.0, 64.0] {
                assert_eq!(sharpen(values, sharpness), values.powf(sharpness));
            }
        }
    }

    #[test]
    fn decals_cover_their_frame_facing_them() {
        let d = Decal {
            center: [0.0; 3],
            normal: [0.0, 0.0, 1.0],
            up: [0.0, 1.0, 0.0],
            size: [2.0, 1.0],
            rotation: 0.0,
            offset: [0.0; 2],
            depth: 0.5,
            symmetry: None,
            flip: [false; 2],
        };
        let z = Vec3::Z;
        let (uv, w) = decal_uv(&d, Vec3::new(0.5, 0.25, 0.0), z).unwrap();
        assert!((uv - Vec2::new(0.75, 0.25)).length() < 1e-5 && (w - 1.0).abs() < 1e-5, "{uv} {w}");
        assert!(decal_uv(&d, Vec3::new(1.1, 0.0, 0.0), z).is_none(), "outside the frame");
        assert!(decal_uv(&d, Vec3::ZERO, -z).is_none(), "back faces");
        assert!(decal_uv(&d, Vec3::new(0.0, 0.0, 0.6), z).is_none(), "beyond the depth");

        // Rotated a quarter turn and shifted: the frame turns with it
        let r = Decal { rotation: 90.0, offset: [1.0, 0.0], ..d };
        let (uv, _) = decal_uv(&r, Vec3::new(1.0, 0.9, 0.0), z).unwrap();
        assert!((uv.x - 0.95).abs() < 1e-4 && (uv.y - 0.5).abs() < 1e-4, "{uv}");
        let c = decal_corners(&d).unwrap();
        assert!((c[0] - Vec3::new(-1.0, 0.5, 0.0)).length() < 1e-5 && (c[2] - Vec3::new(1.0, -0.5, 0.0)).length() < 1e-5);
        assert!(decal_axes(&Decal { normal: [0.0; 3], ..d }).is_none());
        assert!(decal_uv(&Decal { size: [f32::NAN; 2], ..d }, Vec3::ZERO, z).is_none());
    }

    #[test]
    fn decals_mirror_their_image_inside_the_frame() {
        let d = Decal {
            center: [0.0; 3],
            normal: [0.0, 0.0, 1.0],
            up: [0.0, 1.0, 0.0],
            size: [2.0, 1.0],
            rotation: 0.0,
            offset: [0.0; 2],
            depth: 0.5,
            symmetry: None,
            flip: [false; 2],
        };
        let (left, top) = (Vec3::new(-0.5, 0.25, 0.0), Vec3::Z);
        let uv = |flip: [bool; 2]| decal_uv(&Decal { flip, ..d }, left, top).unwrap().0;
        assert!((uv([false, false]) - Vec2::new(0.25, 0.25)).length() < 1e-5);
        assert!((uv([true, false]) - Vec2::new(0.75, 0.25)).length() < 1e-5, "Flip U mirrors left to right only");
        assert!((uv([false, true]) - Vec2::new(0.25, 0.75)).length() < 1e-5, "Flip V mirrors top to bottom only");
        assert!((uv([true, true]) - Vec2::new(0.75, 0.75)).length() < 1e-5);

        // The frame itself does not move: coverage is the same flipped or not
        let covered = |flip: [bool; 2], p: Vec3| decal_uv(&Decal { flip, ..d }, p, top).map(|(_, w)| w);
        assert_eq!(covered([true, true], left), covered([false, false], left));
        assert_eq!(covered([true, true], Vec3::new(1.1, 0.0, 0.0)), None);
    }

    #[test]
    fn twill_floats_run_in_diagonal_ribs() {
        // The value at the centre of cell (x, y); cells are half a unit.
        let at = |k, x: i32, y: i32| raw(k, Vec3::new(x as f32 * 0.5 + 0.25, y as f32 * 0.5 + 0.25, 0.0), 3);
        for y in 1..8 {
            let warp: Vec<bool> = (0..8).map(|x| at(ProceduralKind::Twill, x, y) > 0.7).collect();
            assert_eq!(warp.iter().filter(|w| **w).count(), 4, "row {y}: half the floats are warp");
            for x in 0..7 {
                assert_eq!(warp[x as usize], at(ProceduralKind::Twill, x + 1, y - 1) > 0.7, "the rib continues diagonally at ({x}, {y})");
            }
        }
        // A plain weave shows every top thread alike.
        assert!((0..8).all(|x| at(ProceduralKind::Weave, x, 0) > 0.7));
    }

    #[test]
    fn every_pattern_varies() {
        for k in ProceduralKind::ALL {
            let pr = Procedural::new(k);
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for i in 0..4096 {
                let uv = Vec2::new((i % 64) as f32 / 64.0, (i / 64) as f32 / 64.0);
                let v = procedural(&pr, &Projection::default(), &Coords { uv, pos: Vec3::ZERO, nrm: Vec3::Y });
                (lo, hi) = (lo.min(v), hi.max(v));
            }
            assert!(hi - lo > 0.3, "{k:?} is flat: {lo}..{hi}");
        }
    }
}
