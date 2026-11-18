//! The Tile Generator pattern: shapes in a grid of tiles over one repeating unit, x right and y down as a texture's rows

use std::sync::LazyLock;

use texturecraft_core::{TileBlending, TileGenerator, TileInputDistribution, TileShape, TileSizeMode, TileSymmetry};
use texturecraft_geom::glam::Vec2;

use crate::store::{ImageRes, ImageStore};
use crate::tile_random_stream::TileRandomStream;

/// The settings a Tile Generator pattern saved without its own reads
pub(crate) static DEFAULT: LazyLock<TileGenerator> = LazyLock::new(TileGenerator::default);

/// The output size the pixel size mode measures shapes against
const PIXEL_REFERENCE: f32 = 2048.0;
/// Position random moves a tile up to this many cells times its amount, either way
const POSITION_RANDOM_CELLS: f32 = 1.0;
/// Capsule shape: the radius of its round ends
const CAPSULE_RADIUS: f32 = 0.2225;
/// Half bell shape: the width of its squashed half at a specific of 0
const HALF_BELL_WIDTH: f32 = 1.24;
/// Gaussian shape: the falloff exponent, 2^(−k·r²)
const GAUSSIAN_FALLOFF: f32 = 16.0;
/// Crescent shape: scales its ring r²(1 − r²)² to a peak of 1
const CRESCENT_GAIN: f32 = 6.75;
/// Waves shape: cycles across the tile at a specific of 0, and the cycles a specific of 1 adds
const WAVES_BASE_CYCLES: f32 = 1.5;
const WAVES_SPECIFIC_CYCLES: f32 = 6.0;
/// Narrower than this, a size or a bevel is none
const EPSILON: f32 = 1e-6;
/// Valid tile settings reach at most three neighboring copies along an axis
const MAX_PERIODIC_COPY: f32 = 3.0;

fn smoothstep(t: f32) -> f32 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn bell(r: f32) -> f32 {
    (1.0 - r * r).clamp(0.0, 1.0).powi(4)
}

/// The shape's value at (u, v) of its square (−1..1 both ways; 0 outside it). `specific` changes some shapes
fn shape(kind: TileShape, u: f32, v: f32, specific: f32) -> f32 {
    if u.abs() > 1.0 || v.abs() > 1.0 {
        return 0.0;
    }

    let r = u.hypot(v);
    match kind {
        TileShape::Square => 1.0,
        TileShape::Disc => f32::from(u8::from(r <= 1.0)),
        TileShape::Paraboloid => (1.0 - r * r).clamp(0.0, 1.0),
        TileShape::Bell => bell(r),
        TileShape::Gaussian => (-GAUSSIAN_FALLOFF * r * r).exp2(),
        TileShape::Thorn => (1.0 - r).clamp(0.0, 1.0).powi(4),
        TileShape::Pyramid => (1.0 - u.abs().max(v.abs())).clamp(0.0, 1.0),
        TileShape::Cone => (1.0 - r).clamp(0.0, 1.0),
        TileShape::Brick if specific <= EPSILON => 1.0,
        TileShape::Brick => smoothstep((1.0 - u.abs()) / specific) * smoothstep((1.0 - v.abs()) / specific),
        TileShape::Gradation => (u + 1.0) * 0.5 * (1.0 - specific),
        TileShape::Waves => {
            let cycles = WAVES_BASE_CYCLES + WAVES_SPECIFIC_CYCLES * specific;
            (1.0 - r * r).clamp(0.0, 1.0) * (0.5 + 0.5 * (std::f32::consts::TAU * cycles * u).cos())
        }
        TileShape::HalfBell => {
            let width = HALF_BELL_WIDTH * (1.0 - specific);
            let cut = match (u > 0.0, width > EPSILON) {
                (false, _) => 1.0,
                (true, true) => bell(u / width),
                (true, false) => 0.0,
            };
            bell(r) * cut
        }
        TileShape::RidgedBell => bell(r) * (1.0 - u.abs()).powi(2),
        TileShape::Crescent if specific >= 1.0 => 0.0,
        TileShape::Crescent => {
            let shifted = (u + 1.0) / (1.0 - specific) - 1.0;
            let ring = (1.0 - r * r).clamp(0.0, 1.0);
            CRESCENT_GAIN * r * r * ring * ring * smoothstep((1.0 - shifted) * 0.5)
        }
        TileShape::Capsule => {
            let half_length = specific * (1.0 - CAPSULE_RADIUS);
            let distance = (u.abs() - half_length).max(0.0).hypot(v);
            (1.0 - (distance / CAPSULE_RADIUS).powi(2)).clamp(0.0, 1.0)
        }

        // The image shape draws an input image instead (see `pattern`): without one it is empty
        TileShape::Image => 0.0,
    }
}

/// `point` turned clockwise on screen by `turns` undone (y runs down)
fn unrotate(point: Vec2, turns: f32) -> Vec2 {
    if turns == 0.0 {
        return point;
    }

    let (sin, cos) = (turns * std::f32::consts::TAU).sin_cos();
    Vec2::new(cos * point.x + sin * point.y, -sin * point.x + cos * point.y)
}

/// The pattern section: tile-local (u, v) through the shape's tiling, angle, scale, size and 45° turn, then the shape (`input` for the image shape)
fn pattern(tile: &TileGenerator, mut local: Vec2, input: Option<&ImageRes>) -> f32 {
    if local.x.abs() > 1.0 || local.y.abs() > 1.0 {
        return 0.0;
    }

    if tile.shape_tiling > 1 {
        // Tiling shows the copy just above and left of center
        let count = tile.shape_tiling as f32;
        let shown = (tile.shape_tiling.div_ceil(2) - 1) as f32;
        let cell = (local + 1.0) * 0.5 * count;
        if cell.x.floor() != shown || cell.y.floor() != shown {
            return 0.0;
        }

        local = (cell - shown) * 2.0 - 1.0;
    }
    local = unrotate(local, tile.shape_angle / 360.0);
    let size = Vec2::from(tile.shape_size) * tile.shape_scale;
    if size.x <= EPSILON || size.y <= EPSILON {
        return 0.0;
    }

    local /= size;
    if tile.shape_diagonal {
        local = unrotate(local, 0.125) * std::f32::consts::SQRT_2;
    }
    if tile.shape == TileShape::Image {
        return input.map_or(0.0, |image| image_shape(image, local));
    }

    let samples = Vec2::new(tile.shape_resolution[0] as f32, tile.shape_resolution[1] as f32);
    sampled_shape(tile, local, samples)
}

/// An input image over the shape's square, as gray (the average of its color)
fn image_shape(image: &ImageRes, local: Vec2) -> f32 {
    if local.x.abs() > 1.0 || local.y.abs() > 1.0 {
        return 0.0;
    }

    let at = (local + 1.0) * 0.5;
    gray(image.sample(at.x, at.y))
}

/// The gray of an image texel: the average of its color
fn gray([red, green, blue, _]: [f32; 4]) -> f32 {
    ((red + green + blue) / 3.0).clamp(0.0, 1.0)
}

/// The shape read from an image of `samples` texels over its square, bilinearly, as the shape is drawn at its own resolution
fn sampled_shape(tile: &TileGenerator, local: Vec2, samples: Vec2) -> f32 {
    let grid = (local + 1.0) * 0.5 * samples - 0.5;
    let base = grid.floor();
    let fraction = grid - base;
    let at = |dx: f32, dy: f32| {
        let texel = (base + Vec2::new(dx, dy) + 0.5) / samples * 2.0 - 1.0;
        shape(tile.shape, texel.x, texel.y, tile.specific)
    };
    let top_left = at(0.0, 0.0);
    let bottom_left = at(0.0, 1.0);
    let top = top_left + (at(1.0, 0.0) - top_left) * fraction.x;
    let bottom = bottom_left + (at(1.0, 1.0) - bottom_left) * fraction.x;
    top + (bottom - top) * fraction.y
}

/// The start and width, in cells, of cell `index` along an axis whose middle size is `middle` (cells pair up)
fn cell_span(index: i32, middle: f32) -> (f32, f32) {
    let pair = index.div_euclid(2) as f32 * 2.0;
    if index.rem_euclid(2) == 0 { (pair, 2.0 * middle) } else { (pair + 2.0 * middle, 2.0 * (1.0 - middle)) }
}

/// `delta` wrapped into −span/2..span/2 (the pattern repeats every `span`)
fn wrap(delta: f32, span: f32) -> f32 {
    delta - (delta / span).round() * span
}

/// What one tile gives a texel (its shape times its luminance), or None when hidden or out of reach; `inputs` are its input images
fn tile_value(
    tile: &TileGenerator,
    seed: u32,
    (column, row): (i32, i32),
    (columns, rows): (i32, i32),
    (point, inputs): (Vec2, &[Option<&ImageRes>]),
) -> Option<f32> {
    let total = columns * rows;
    let number = column * rows + row;
    let index = (if tile.reverse_order { total - 1 - number } else { number }) as u32;
    if hidden(tile, seed, (column, row), index) {
        return None;
    }

    let (start_x, width) = cell_span(column, tile.middle_size[0]);
    let (start_y, height) = cell_span(row, tile.middle_size[1]);
    let mut center = Vec2::new(start_x + width * 0.5, start_y + height * 0.5);
    if tile.vertical_offset {
        center.y += line_shift(tile, column);
    } else {
        center.x += line_shift(tile, row);
    }
    center.x += (TileRandomStream::PositionX.draw(seed, index) - 0.5) * 2.0 * POSITION_RANDOM_CELLS * tile.position_random[0];
    center.y += (TileRandomStream::PositionY.draw(seed, index) - 0.5) * 2.0 * POSITION_RANDOM_CELLS * tile.position_random[1];
    let counts = Vec2::new(columns as f32, rows as f32);
    let center = center / counts + Vec2::from(tile.global_offset);

    let scale_factor = 1.0 - tile.scale_random * TileRandomStream::Scale.draw(tile.scale_random_seed, index);
    let size = tile_size(tile, seed, index, Vec2::new(width, height) / counts) * tile.scale * scale_factor;
    if size.x <= EPSILON || size.y <= EPSILON {
        return None;
    }

    let delta = Vec2::new(wrap(point.x - center.x, 1.0), wrap(point.y - center.y, 1.0));
    let turns = (tile.rotation + tile.rotation_random * TileRandomStream::Rotation.draw(seed, index) * 360.0) / 360.0;
    let (sin, cos) = (turns * std::f32::consts::TAU).sin_cos();
    let radius = Vec2::new(cos.abs() * size.x + sin.abs() * size.y, sin.abs() * size.x + cos.abs() * size.y) * 0.5;
    let first = (-radius - delta).ceil().max(Vec2::splat(-MAX_PERIODIC_COPY));
    let last = (radius - delta).floor().min(Vec2::splat(MAX_PERIODIC_COPY));
    let quarter_turns = quarter_turns(tile, seed, (column, row), index);
    let mirrors = tile.symmetry_random * 0.5;
    let mirror_x = tile.symmetry != TileSymmetry::Vertical && TileRandomStream::MirrorX.draw(seed, index) < mirrors;
    let mirror_y = tile.symmetry != TileSymmetry::Horizontal && TileRandomStream::MirrorY.draw(seed, index) < mirrors;

    let input = match tile.input_distribution {
        _ if inputs.is_empty() => None,
        TileInputDistribution::Random => inputs.get((TileRandomStream::Input.draw(seed, index) * inputs.len() as f32) as usize).copied().flatten(),
        TileInputDistribution::ByNumber => inputs.get(number as usize % inputs.len()).copied().flatten(),
    };
    let mut value = 0.0f32;
    // A rotated shape may cross a repeat boundary even when the nearest copy misses it
    for copy_y in first.y as i32..=last.y as i32 {
        let y = delta.y + copy_y as f32;
        if y.abs() > radius.y {
            continue;
        }
        for copy_x in first.x as i32..=last.x as i32 {
            let x = delta.x + copy_x as f32;
            if x.abs() > radius.x {
                continue;
            }
            let mut local = Vec2::new(cos * x + sin * y, -sin * x + cos * y) * 2.0 / size;
            for _ in 0..quarter_turns {
                local = Vec2::new(local.y, -local.x);
            }
            if mirror_x {
                local.x = -local.x;
            }
            if mirror_y {
                local.y = -local.y;
            }
            value = value.max(pattern(tile, local, input));
        }
    }
    if value <= 0.0 {
        return None;
    }

    let mut luminance = 1.0 - tile.luminance_random * TileRandomStream::Luminance.draw(seed, index);
    if tile.luminance_by_number {
        luminance *= (number + 1) as f32 / total as f32;
    }
    if tile.luminance_by_scale {
        luminance *= scale_factor;
    }
    Some(value * luminance)
}

/// Whether the masks hide tile (column, row), whose randoms are at `index`
fn hidden(tile: &TileGenerator, seed: u32, (column, row): (i32, i32), index: u32) -> bool {
    let masked = (tile.checker_mask && (column + row).rem_euclid(2) == 1)
        || (tile.horizontal_mask && row.rem_euclid(2) == 1)
        || (tile.vertical_mask && column.rem_euclid(2) == 0)
        || (tile.random_mask > 0.0 && TileRandomStream::Mask.draw(seed, index) < tile.random_mask);
    masked != tile.invert_mask
}

/// The tile's width and height on the plane before its scale, for a cell `cell` wide and high
fn tile_size(tile: &TileGenerator, seed: u32, index: u32, cell: Vec2) -> Vec2 {
    match tile.size_mode {
        TileSizeMode::NormalInterstice => {
            let gap_x = tile.interstice[0] * (1.0 - tile.interstice_random[0] * TileRandomStream::IntersticeX.draw(seed, index));
            let gap_y = tile.interstice[1] * (1.0 - tile.interstice_random[1] * TileRandomStream::IntersticeY.draw(seed, index));
            cell * Vec2::new(1.0 - gap_x, 1.0 - gap_y)
        }
        TileSizeMode::NormalSize => cell * Vec2::from(tile.size),
        TileSizeMode::KeepRatio => Vec2::splat(cell.x.min(cell.y)),
        TileSizeMode::Absolute | TileSizeMode::Pixel => {
            let size = if tile.size_mode == TileSizeMode::Absolute { Vec2::from(tile.size_absolute) } else { Vec2::from(tile.size_pixel) / PIXEL_REFERENCE };
            let random_x = 1.0 - tile.size_random[0] * TileRandomStream::SizeX.draw(seed, index);
            let random_y = 1.0 - tile.size_random[1] * TileRandomStream::SizeY.draw(seed, index);
            size * Vec2::new(random_x, random_y)
        }
    }
}

/// The quarter turns clockwise of tile (column, row): the set ones, the quincunx pinwheel and the random ones
fn quarter_turns(tile: &TileGenerator, seed: u32, (column, row): (i32, i32), index: u32) -> u32 {
    let mut turns = u32::from(tile.quarter_turns);
    if tile.quincunx_flip {
        turns += match (column.rem_euclid(2), row.rem_euclid(2)) {
            (0, 0) => 0,
            (1, 0) => 3,
            (0, _) => 1,
            _ => 2,
        };
    }
    if tile.quarter_turn_random > 0.0 && TileRandomStream::QuarterTurnApplies.draw(seed, index) < tile.quarter_turn_random {
        turns += (TileRandomStream::QuarterTurn.draw(seed, index) * 4.0) as u32;
    }
    turns % 4
}

/// How far, in cells along each axis, a tile's center can be from a texel its shape reaches: half its largest size turned, plus its random move and the shift of its center by the middle size
fn reach(tile: &TileGenerator, counts: Vec2) -> Vec2 {
    let widest = Vec2::new(tile.middle_size[0].max(1.0 - tile.middle_size[0]), tile.middle_size[1].max(1.0 - tile.middle_size[1])) * 2.0;
    let largest = match tile.size_mode {
        TileSizeMode::NormalInterstice | TileSizeMode::KeepRatio => widest / counts,
        TileSizeMode::NormalSize => widest / counts * Vec2::from(tile.size),
        TileSizeMode::Absolute => Vec2::from(tile.size_absolute),
        TileSizeMode::Pixel => Vec2::from(tile.size_pixel) / PIXEL_REFERENCE,
    } * tile.scale;
    let radius = largest.length() * 0.5 * counts;
    let moved = Vec2::from(tile.position_random) * POSITION_RANDOM_CELLS;
    let middle = Vec2::new((tile.middle_size[0] - 0.5).abs(), (tile.middle_size[1] - 0.5).abs());
    radius + moved + middle
}

/// The cells along an axis of `count` whose centers can be within `reach` of `position` (in cells): each once, at most `count`
fn near(position: f32, reach: f32, count: i32) -> impl Iterator<Item = i32> {
    let first = (position - reach - 0.5).floor();
    let last = (position + reach - 0.5).ceil();

    // A reach past the whole grid visits each cell once; float math keeps a hostile reach from overflowing
    let span = (last - first + 1.0).min(count as f32).max(1.0) as i32;
    let first = first as i32;
    (first..first.saturating_add(span)).map(move |index| index.rem_euclid(count))
}

/// The shift along its line of line `line` (a row, or a column with a vertical offset), in cells
fn line_shift(tile: &TileGenerator, line: i32) -> f32 {
    tile.offset * line as f32 + tile.offset_random * TileRandomStream::Offset.draw(tile.offset_random_seed, line as u32)
}

/// Every image name [`value`] may read: its inputs (whatever the shape) and its background. The
/// pattern cache keys values by these images, so a name `value` reads must be listed here.
pub(crate) fn image_names(tile: &TileGenerator) -> impl Iterator<Item = &str> {
    tile.inputs.iter().take(TileGenerator::MAX_INPUTS).map(String::as_str).chain(tile.background_image.as_deref())
}

/// The pattern's value at `point` of its plane (one unit = the whole grid, repeating); `images` holds the images its inputs name
pub(crate) fn value(tile: &TileGenerator, point: Vec2, seed: u32, images: Option<&ImageStore>) -> f32 {
    let image = |name: &str| images.and_then(|images| images.get(name)).map(|image| &**image);
    let mut slots: [Option<&ImageRes>; TileGenerator::MAX_INPUTS] = [None; TileGenerator::MAX_INPUTS];
    let count = tile.inputs.len().min(TileGenerator::MAX_INPUTS);
    if tile.shape == TileShape::Image {
        for (slot, name) in slots.iter_mut().zip(&tile.inputs) {
            *slot = image(name);
        }
    }
    let inputs = &slots[..count];
    let background = tile.background_image.as_deref().and_then(image);

    // Commands and saved projects keep the settings in range; the counts are guarded as they size the loops
    let (columns, rows) = (tile.amount[0].clamp(1, *TileGenerator::AMOUNT.end()) as i32, tile.amount[1].clamp(1, *TileGenerator::AMOUNT.end()) as i32);
    let counts = Vec2::new(columns as f32, rows as f32);
    let point = Vec2::new(point.x.rem_euclid(1.0), point.y.rem_euclid(1.0));
    let cell = (point - Vec2::from(tile.global_offset)) * counts;
    let reach = reach(tile, counts);

    let mut sum = 0.0f32;
    let mut colored = 0.0f32;
    let mut add = |value: f32| match tile.blending {
        TileBlending::Max => {
            sum = sum.max(value);
            colored = colored.max(value * tile.color);
        }
        TileBlending::Add => {
            sum += value;
            colored += value * tile.color;
        }
        TileBlending::AddSub => {
            sum += value;
            colored += value * (2.0 * tile.color - 1.0);
        }
    };

    // Each shifted line needs its own nearby tile search
    if tile.vertical_offset {
        for column in near(cell.x, reach.x, columns) {
            for row in near(cell.y - line_shift(tile, column), reach.y, rows) {
                if let Some(value) = tile_value(tile, seed, (column, row), (columns, rows), (point, inputs)) {
                    add(value);
                }
            }
        }
    } else {
        for row in near(cell.y, reach.y, rows) {
            for column in near(cell.x - line_shift(tile, row), reach.x, columns) {
                if let Some(value) = tile_value(tile, seed, (column, row), (columns, rows), (point, inputs)) {
                    add(value);
                }
            }
        }
    }
    let background = background.map_or(tile.background, |image| gray(image.sample(point.x, point.y)));
    finish(tile, sum, colored, background)
}

/// The combined tiles over `background`, through the pattern's opacity and its position/contrast window
fn finish(tile: &TileGenerator, sum: f32, colored: f32, background: f32) -> f32 {
    let alpha = sum.clamp(0.0, 1.0) * tile.opacity;
    let color = colored.clamp(0.0, 1.0) * tile.opacity;
    let value = (background * (1.0 - alpha) + color).clamp(0.0, 1.0);
    let windowed = window(value, tile.position, tile.contrast);

    // Out-of-range settings drawn as they are must not leave a non-number in the texture
    if windowed.is_finite() { windowed } else { 0.0 }
}

/// The position/contrast window: position picks the input range stretched to 0..1 (0.5 keeps it), contrast narrows it around its middle
fn window(value: f32, position: f32, contrast: f32) -> f32 {
    let low = (1.0 - 2.0 * position).clamp(0.0, 1.0);
    let high = (2.0 - 2.0 * position).clamp(0.0, 1.0);
    let middle = (low + high) * 0.5;
    let half = (high - low) * 0.5 * (1.0 - contrast);
    if half <= EPSILON {
        return f32::from(u8::from(value >= middle));
    }

    ((value - (middle - half)) / (2.0 * half)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn at(tile: &TileGenerator, x: f32, y: f32) -> f32 {
        value(tile, Vec2::new(x, y), 1, None)
    }

    #[test]
    fn rotated_tiles_cover_the_repeat_boundary_once() {
        let tile = TileGenerator {
            shape: TileShape::Square,
            amount: [1, 1],
            size_mode: TileSizeMode::NormalSize,
            size: [1.0, 0.1],
            scale: 2.0,
            rotation: 45.0,
            blending: TileBlending::Add,
            ..TileGenerator::default()
        };

        assert!(at(&tile, 0.99, 0.01) > 0.99);
        assert!(at(&tile, 0.99, 0.01) <= 1.0);
    }

    #[test]
    fn overlapping_periodic_copies_contribute_one_tile_value() {
        let tile = TileGenerator {
            shape: TileShape::Square,
            amount: [1, 1],
            size_mode: TileSizeMode::NormalSize,
            size: [1.0, 1.0],
            scale: 2.0,
            color: 0.4,
            blending: TileBlending::Add,
            ..TileGenerator::default()
        };

        assert!((at(&tile, 0.5, 0.5) - 0.4).abs() < EPSILON);
    }

    /// Bell shapes peak only at their tile's center, so the brightest texel of a cell shows where its tile is
    fn bells(amount: [u32; 2]) -> TileGenerator {
        TileGenerator { shape: TileShape::Bell, amount, ..TileGenerator::default() }
    }

    #[test]
    fn tile_centers_follow_the_amounts() {
        let tile = bells([4, 2]);

        // Cell (i, j) of a 4 × 2 grid is centered at ((i + 0.5) / 4, (j + 0.5) / 2)
        for (x, y) in [(0.125, 0.25), (0.375, 0.75), (0.875, 0.25)] {
            assert!(at(&tile, x, y) > 0.99, "a tile center at ({x}, {y})");
        }
        assert!(at(&tile, 0.25, 0.25) < 0.01, "the bells meet at zero between cells");
    }

    #[test]
    fn the_offset_shifts_each_row_along_by_its_share_of_a_cell() {
        let tile = TileGenerator { offset: 0.5, ..bells([4, 4]) };

        // Row 0 stays; row 1 (y 0.375) moves half a cell (0.125) to the right
        assert!(at(&tile, 0.125, 0.125) > 0.99);
        assert!(at(&tile, 0.25, 0.375) > 0.99, "row 1's first tile is centered at x 0.25");
        assert!(at(&tile, 0.125, 0.375) < 0.5);

        let columns = TileGenerator { vertical_offset: true, ..tile };
        assert!(at(&columns, 0.375, 0.25) > 0.99, "a vertical offset moves column 1 down instead");
    }

    #[test]
    fn a_free_rotation_turns_the_shape_clockwise() {
        // The half bell's bright side is to the left (−x); turned a quarter clockwise it is at the top
        let tile = TileGenerator { shape: TileShape::HalfBell, specific: 1.0, amount: [1, 1], ..TileGenerator::default() };
        assert!(at(&tile, 0.3, 0.5) > at(&tile, 0.7, 0.5));
        let turned = TileGenerator { rotation: 90.0, ..tile };
        assert!(at(&turned, 0.5, 0.3) > at(&turned, 0.5, 0.7), "the bright side moved from the left to the top");
    }

    /// Whether a disc of a 2 × 1 grid's first tile, centered at (0.25, 0.5), covers the point `right` and `down` of its center
    fn first_disc_covers(tile: &TileGenerator, right: f32, down: f32) -> bool {
        at(tile, 0.25 + right, 0.5 + down) > 0.5
    }

    #[test]
    fn each_size_mode_sizes_the_shape_its_own_way() {
        let discs = TileGenerator { shape: TileShape::Disc, amount: [2, 1], ..TileGenerator::default() };
        assert!(first_disc_covers(&discs, 0.2, 0.0) && first_disc_covers(&discs, 0.0, 0.4), "without a gap the disc fills its 0.5 × 1 cell");

        // Half the cell: 0.25 wide and 0.5 high, so the disc ends 0.125 right and 0.25 below its center
        let halved = TileGenerator { size_mode: TileSizeMode::NormalSize, size: [0.5, 0.5], ..discs.clone() };
        assert!(first_disc_covers(&halved, 0.1, 0.0) && !first_disc_covers(&halved, 0.15, 0.0));
        assert!(first_disc_covers(&halved, 0.0, 0.2) && !first_disc_covers(&halved, 0.0, 0.3));

        // The largest square of the cell is 0.5 wide and high, whatever the size is set to
        let square = TileGenerator { size_mode: TileSizeMode::KeepRatio, ..halved.clone() };
        assert!(first_disc_covers(&square, 0.2, 0.0) && first_disc_covers(&square, 0.0, 0.2) && !first_disc_covers(&square, 0.0, 0.3));

        // A fifth of the pattern has the same extent in absolute and reference pixels
        let absolute = TileGenerator { size_mode: TileSizeMode::Absolute, size_absolute: [0.2, 0.2], ..discs.clone() };
        let pixel = TileGenerator { size_mode: TileSizeMode::Pixel, size_pixel: [0.2 * PIXEL_REFERENCE; 2], ..discs.clone() };
        for sized in [&absolute, &pixel] {
            assert!(first_disc_covers(sized, 0.08, 0.0) && !first_disc_covers(sized, 0.12, 0.0), "{:?}", sized.size_mode);
            assert!(first_disc_covers(sized, 0.0, 0.08) && !first_disc_covers(sized, 0.0, 0.12), "{:?}", sized.size_mode);
        }
    }

    #[test]
    fn size_randomness_shrinks_shapes_sized_absolutely_or_in_pixels_only() {
        // Sixteen texels across each of the 4 × 4 cells: enough for the covered count to move when the discs shrink
        const SAMPLES_PER_SIDE: usize = 64;
        let center = |sample: usize| (sample as f32 + 0.5) / SAMPLES_PER_SIDE as f32;
        let covered = |tile: &TileGenerator| {
            (0..SAMPLES_PER_SIDE).flat_map(|y| (0..SAMPLES_PER_SIDE).map(move |x| (x, y))).filter(|(x, y)| at(tile, center(*x), center(*y)) > 0.5).count()
        };
        let fifth = [0.2, 0.2];
        let discs = TileGenerator {
            shape: TileShape::Disc,
            amount: [4, 4],
            size: [0.8, 0.8],
            size_absolute: fifth,
            size_pixel: fifth.map(|share| share * PIXEL_REFERENCE),
            ..TileGenerator::default()
        };
        let random = TileGenerator { size_random: [1.0, 1.0], ..discs.clone() };

        for mode in [TileSizeMode::NormalInterstice, TileSizeMode::NormalSize, TileSizeMode::KeepRatio] {
            let (plain, random) = (TileGenerator { size_mode: mode, ..discs.clone() }, TileGenerator { size_mode: mode, ..random.clone() });
            assert_eq!(covered(&plain), covered(&random), "{mode:?} ignores the size randomness");
        }
        for mode in [TileSizeMode::Absolute, TileSizeMode::Pixel] {
            let (plain, random) = (TileGenerator { size_mode: mode, ..discs.clone() }, TileGenerator { size_mode: mode, ..random.clone() });
            assert!(covered(&random) < covered(&plain), "{mode:?}: random sizes shrink the discs");
        }
    }

    #[test]
    fn masks_hide_their_rows_columns_and_checkers() {
        let checker = TileGenerator { checker_mask: true, ..bells([2, 2]) };
        assert!(at(&checker, 0.25, 0.25) > 0.99 && at(&checker, 0.75, 0.25) < 0.01, "odd (i + j) is hidden");
        let rows = TileGenerator { horizontal_mask: true, ..bells([2, 2]) };
        assert!(at(&rows, 0.25, 0.25) > 0.99 && at(&rows, 0.25, 0.75) < 0.01, "the horizontal mask hides odd rows");
        let columns = TileGenerator { vertical_mask: true, ..bells([2, 2]) };
        assert!(at(&columns, 0.25, 0.25) < 0.01 && at(&columns, 0.75, 0.25) > 0.99, "the vertical mask hides even columns");
        let inverted = TileGenerator { invert_mask: true, ..columns };
        assert!(at(&inverted, 0.25, 0.25) > 0.99, "invert mask shows what the masks hid");
    }

    #[test]
    fn the_position_window_stretches_a_range_and_contrast_makes_a_step() {
        assert_eq!(window(0.3, 0.5, 0.0), 0.3, "the middle position leaves the value");
        assert_eq!(window(0.75, 0.25, 0.0), 0.5, "position 0.25 stretches 0.5..1 to 0..1");
        assert_eq!(window(0.49, 0.5, 1.0), 0.0);
        assert_eq!(window(0.51, 0.5, 1.0), 1.0, "full contrast cuts at the middle of the range");
    }

    #[test]
    fn every_shape_stays_in_range_and_outside_its_square_is_empty() {
        for shape_kind in TileShape::ALL {
            for specific in [0.0, 0.5, 1.0] {
                for step in 0..41 {
                    let u = step as f32 / 20.0 - 1.0;
                    let value = shape(shape_kind, u, u * 0.7, specific);
                    assert!((0.0..=1.0 + 1e-5).contains(&value), "{shape_kind:?} at {u}: {value}");
                }
                assert_eq!(shape(shape_kind, 1.2, 0.0, specific), 0.0);
            }
        }
    }

    /// An image `size` texels wide whose left half is `left` and right half `right`
    fn halves(left: f32, right: f32) -> std::sync::Arc<ImageRes> {
        let size = 8;
        let rgba = (0..size * size).map(|texel| if texel % size < size / 2 { [left, left, left, 1.0] } else { [right, right, right, 1.0] }).collect();
        std::sync::Arc::new(ImageRes { width: size as u32, height: size as u32, rgba, srgb: false })
    }

    #[test]
    fn the_image_shape_draws_its_inputs_in_turn_by_pattern_number() {
        let images: ImageStore = [("dark".to_string(), halves(0.2, 0.2)), ("bright".to_string(), halves(0.9, 0.9))].into();
        let tile = TileGenerator {
            shape: TileShape::Image,
            amount: [2, 1],
            inputs: vec!["dark".into(), "bright".into()],
            input_distribution: TileInputDistribution::ByNumber,
            ..TileGenerator::default()
        };

        // Tile 0 (left) draws the first input, tile 1 the second
        let left = value(&tile, Vec2::new(0.25, 0.5), 1, Some(&images));
        let right = value(&tile, Vec2::new(0.75, 0.5), 1, Some(&images));
        assert!((left - 0.2).abs() < 0.01 && (right - 0.9).abs() < 0.01, "{left} {right}");
        assert_eq!(value(&tile, Vec2::new(0.25, 0.5), 1, None), 0.0, "missing images draw nothing");
    }

    #[test]
    fn an_image_input_keeps_its_orientation_in_the_tile() {
        let images: ImageStore = [("halves".to_string(), halves(0.0, 1.0))].into();
        let tile = TileGenerator { shape: TileShape::Image, amount: [1, 1], inputs: vec!["halves".into()], ..TileGenerator::default() };
        assert!(value(&tile, Vec2::new(0.3, 0.5), 1, Some(&images)) < 0.05, "the image's left half is on the tile's left");
        assert!(value(&tile, Vec2::new(0.7, 0.5), 1, Some(&images)) > 0.95);
    }

    #[test]
    fn the_background_image_shows_between_the_shapes() {
        let images: ImageStore = [("ground".to_string(), halves(0.4, 0.4))].into();
        let tile = TileGenerator { background_image: Some("ground".into()), interstice: [0.5, 0.5], ..bells([2, 2]) };

        // Between the bells only the background is left
        assert!((value(&tile, Vec2::new(0.0, 0.0), 1, Some(&images)) - 0.4).abs() < 0.01);
        assert_eq!(value(&tile, Vec2::new(0.0, 0.0), 1, None), 0.0, "without the image the background value stays");
    }

    #[test]
    fn hostile_settings_neither_crash_nor_leave_the_range() {
        let hostile = TileGenerator {
            amount: [0, u32::MAX],
            scale: f32::INFINITY,
            size_mode: TileSizeMode::Pixel,
            size_pixel: [f32::NAN, 1e30],
            position_random: [1.0, 1.0],
            offset: f32::NAN,
            middle_size: [0.0, 1.0],
            ..TileGenerator::default()
        };
        for step in 0..20 {
            let value = at(&hostile, step as f32 * 0.37, -(step as f32) * 1.3);
            assert!((0.0..=1.0).contains(&value));
        }
    }
}
