use std::sync::Arc;

use texturecraft_compose::{BakedMaps, ImageStore, Inputs, PaintTarget, PixelStore};
use texturecraft_core::Project;
use texturecraft_geom::{Mesh, UvGBuffer};

/// Compositor inputs frozen at stroke start. A mask target identifies the layer that owns it.
#[derive(Clone, Debug)]
pub struct PickupSnapshot {
    pub project: Project,
    pub ts: usize,
    pub target: PaintTarget,
    pub mesh: Arc<Mesh>,
    pub bounds: (texturecraft_geom::glam::Vec3, texturecraft_geom::glam::Vec3),
    pub pixels: PixelStore,
    pub baked: BakedMaps,
    pub images: ImageStore,
}

impl PickupSnapshot {
    pub(super) fn inputs<'a>(&'a self, gbuf: &'a UvGBuffer) -> Option<Inputs<'a>> {
        let set = self.project.texture_sets.get(self.ts)?;
        Some(Inputs {
            project: &self.project,
            ts: self.ts,
            mesh: &self.mesh,
            gbuf,
            pixels: &self.pixels,
            baked: &self.baked,
            block_scales: set.block_scales(),
            preview: None,
            images: &self.images,
            bounds: self.bounds,
            patterns: None,
        })
    }
}
