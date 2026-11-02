//! What the user is looking at, so background composites can do those tiles first: the tiles
//! of a texture set the 3D view shows (in the view and facing the camera; occlusion is not
//! tested, as this only orders work) and those inside the 2D view's UV rectangle.

use std::collections::BTreeSet;

use texturecraft_geom::glam::Vec3;
use texturecraft_geom::{Mesh, UvGBuffer};
use texturecraft_texel::TILE;

use crate::Session;

/// What the views show, for ordering background composites.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Focus {
    /// The 3D view is on screen (with the session's camera and viewport size).
    pub view3d: bool,
    /// The UV rectangle `[u0, v0, u1, v1]` the 2D view shows of the active texture set
    /// (v down, as the G-buffer's rows), when the 2D view is on screen.
    pub uv: Option<[f32; 4]>,
}

impl Focus {
    /// Nothing to put first (no view described).
    pub fn is_none(&self) -> bool {
        !self.view3d && self.uv.is_none()
    }
}

/// A covered texel's world position and shading normal.
type Probe = (Vec3, Vec3);

/// Sample points of each covered tile of a layout: covered texels on a 4 × 4 grid, or the
/// first covered texel of a sliver the grid misses.
#[derive(Debug, Default)]
pub struct TileProbes {
    tiles: Vec<((u32, u32), Vec<Probe>)>,
}

/// Grid step of the probes inside a tile (texels).
const STEP: u32 = 16;

impl TileProbes {
    pub fn new(mesh: &Mesh, gbuf: &UvGBuffer, covered: &[(u32, u32)]) -> TileProbes {
        let (w, h) = (gbuf.width, gbuf.height);
        let at = |x: u32, y: u32| if x < w && y < h { gbuf.surface(mesh, (y * w + x) as usize).map(|s| (s.pos, s.nrm)) } else { None };
        let tiles = covered
            .iter()
            .map(|&(tx, ty)| {
                let (x0, y0) = (tx * TILE, ty * TILE);
                let grid = (0..TILE / STEP).flat_map(|j| (0..TILE / STEP).map(move |i| (x0 + STEP / 2 + i * STEP, y0 + STEP / 2 + j * STEP)));
                let mut pts: Vec<Probe> = grid.filter_map(|(x, y)| at(x, y)).collect();
                if pts.is_empty() {
                    // A sliver between the grid points: its first covered texel.
                    pts.extend((0..TILE * TILE).find_map(|k| at(x0 + k % TILE, y0 + k / TILE)));
                }
                ((tx, ty), pts)
            })
            .collect();
        TileProbes { tiles }
    }
}

impl Session {
    /// The covered tiles of texture set `ts` that `focus` shows: facing the camera inside the
    /// 3D view, or (the active set only) inside the 2D view's rectangle. A 2D view showing the
    /// whole set beside the 3D view puts nothing first, so it is left out then.
    pub fn visible_tiles(&self, ts: usize, focus: &Focus) -> BTreeSet<(u32, u32)> {
        let mut out = BTreeSet::new();
        let Some(c) = self.caches.get(ts) else { return out };
        let (w, h) = (c.size.0.max(1) as f32, c.size.1.max(1) as f32);
        let uv = focus.uv.filter(|_| ts == self.ts()).filter(|r| !(focus.view3d && r[0] <= 0.0 && r[1] <= 0.0 && r[2] >= 1.0 && r[3] >= 1.0));
        let probes = c.probes.get_or_init(|| TileProbes::new(&self.mesh, &c.gbuf, &c.covered_tiles));
        let cam = &self.camera;
        let aspect = self.viewport.x / self.viewport.y.max(1.0);
        let (eye, forward) = (cam.eye(), cam.forward());
        let seen = |p: Vec3, n: Vec3| {
            let to_eye = if cam.ortho { -forward } else { eye - p };
            n.dot(to_eye) > 0.0 && cam.project(p, aspect).is_some_and(|q| q.x.abs() <= 1.0 && q.y.abs() <= 1.0)
        };
        for ((tx, ty), pts) in &probes.tiles {
            let in_uv = uv.is_some_and(|r| {
                let t = TILE as f32;
                let (u0, v0, u1, v1) = (*tx as f32 * t / w, *ty as f32 * t / h, (*tx + 1) as f32 * t / w, (*ty + 1) as f32 * t / h);
                u1 > r[0] && u0 < r[2] && v1 > r[1] && v0 < r[3]
            });
            if in_uv || (focus.view3d && pts.iter().any(|(p, n)| seen(*p, *n))) {
                out.insert((*tx, *ty));
            }
        }
        out
    }
}
