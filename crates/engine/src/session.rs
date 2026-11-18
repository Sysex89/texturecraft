//! The editing session: document, mesh, per-texture-set caches, history, tool state.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use texturecraft_compose::{
    BakedMaps, BelowCache, Composite, PATTERN_BUDGET, PaintTarget, PatternCache, PixelStore, StackKey, StrokePreview, covered_tiles, pad_tiles,
    stack_reads_neighbours,
};
use texturecraft_core::{Channel, Layer, LayerId, LayerKind, Project, TextureSet};
use texturecraft_geom::glam::Vec2;
use texturecraft_geom::{Bvh, Mesh, OrbitCamera, UvGBuffer};
use texturecraft_paint::{BrushSettings, EyeDepth, Stroke, StrokeCanvas};
use texturecraft_texel::DilationMap;

use crate::fill_mode::FillMode;
pub use crate::stencil_look::StencilLook;
use crate::{EngineError, Result};

/// Padding (texels) applied to composites for display.
pub const DISPLAY_PADDING: u32 = 8;
/// Undo steps kept.
pub const MAX_HISTORY: usize = 200;

/// Everything that undo/redo restores. Pixel data is copy-on-write, so snapshots are cheap.
#[derive(Clone, Debug)]
pub struct Document {
    pub project: Project,
    pub pixels: PixelStore,
    pub baked: BTreeMap<usize, BakedMaps>,
    pub bake_paint: BTreeMap<(usize, texturecraft_core::BakePaintTarget), Arc<texturecraft_bake::BakePaintMap>>,
    /// Imported image resources by name: original file bytes.
    pub resources: BTreeMap<String, Arc<Vec<u8>>>,
}

/// Derived per-texture-set data (rebuilt when the mesh or resolution changes).
pub struct SetCache {
    /// Texels across and down (`TextureSet::size`).
    pub size: (u32, u32),
    pub gbuf: Arc<UvGBuffer>,
    pub dilation: Arc<DilationMap>,
    pub covered_tiles: Arc<Vec<(u32, u32)>>,
    pub composite: Composite,
    /// Tiles to recomposite.
    pub dirty: BTreeSet<(u32, u32)>,
    /// Tiles to recomposite at low priority: what they show is already right (a committed
    /// stroke's live preview), but it was composited from the stroke, not the layer pixels.
    pub settle: BTreeSet<(u32, u32)>,
    /// Tiles whose display textures must be re-uploaded.
    pub upload: BTreeSet<(u32, u32)>,
    /// Bumps on every composite change (thumbnails, exports, caches).
    pub revision: u64,
    /// Everything but the layer being edited, per tile (repaints evaluate only that layer).
    pub below: BelowCache,
    /// Tiles per composite chunk that fit the last frame's budget (0: not measured yet).
    pub chunk: usize,
    /// Procedural pattern values of this layout, kept across composites (bakes and stack
    /// edits leave them valid); a new layout gets a new cache with the new `SetCache`.
    pub patterns: PatternCache,
}

impl SetCache {
    pub(crate) fn new(mesh: &Mesh, ts: &TextureSet) -> SetCache {
        let (w, h) = ts.size();
        let gbuf = UvGBuffer::new(mesh, ts.material, w, h);
        // Padding stays inside each UV tile.
        let dilation = DilationMap::in_blocks(&gbuf.coverage_mask(), w, h, Some(DISPLAY_PADDING), ts.block_resolution());
        let covered = covered_tiles(&gbuf);
        let all: BTreeSet<(u32, u32)> = texturecraft_compose::all_tiles(w, h).into_iter().collect();
        SetCache {
            size: (w, h),
            gbuf: Arc::new(gbuf),
            dilation: Arc::new(dilation),
            covered_tiles: Arc::new(covered),
            composite: Composite { block_scales: ts.block_scales(), ..Composite::with_size(&ts.channel_list(), w, h) },
            dirty: all.clone(),
            settle: BTreeSet::new(),
            upload: all,
            revision: 0,
            below: BelowCache::default(),
            chunk: 0,
            patterns: PatternCache::default(),
        }
    }

    pub fn invalidate_all(&mut self) {
        self.dirty.extend(self.covered_tiles.iter().copied());
    }
}

/// Active tool.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum Tool {
    #[default]
    Paint,
    Eraser,
    Projection,
    PolygonFill,
    Smudge,
    Clone,
    ColorPicker,
    /// Paint along an editable curve (or fill a closed one).
    Path,
    /// Click mesh parts to show or hide the selected layer on them.
    GeometryMask,
}

impl Tool {
    pub const ALL: [Tool; 9] =
        [Tool::Paint, Tool::Eraser, Tool::Projection, Tool::PolygonFill, Tool::Smudge, Tool::Clone, Tool::ColorPicker, Tool::Path, Tool::GeometryMask];
    pub fn label(self) -> &'static str {
        match self {
            Tool::Paint => "Paint",
            Tool::Eraser => "Eraser",
            Tool::Projection => "Projection",
            Tool::PolygonFill => "Polygon fill",
            Tool::Smudge => "Smudge",
            Tool::Clone => "Clone",
            Tool::ColorPicker => "Material picker",
            Tool::Path => "Path",
            Tool::GeometryMask => "Geometry mask",
        }
    }
    pub fn id(self) -> &'static str {
        match self {
            Tool::Paint => "paint",
            Tool::Eraser => "eraser",
            Tool::Projection => "projection",
            Tool::PolygonFill => "polygonFill",
            Tool::Smudge => "smudge",
            Tool::Clone => "clone",
            Tool::ColorPicker => "colorPicker",
            Tool::Path => "path",
            Tool::GeometryMask => "geometryMask",
        }
    }
    pub fn parse(s: &str) -> Option<Tool> {
        let n = |x: &str| x.to_ascii_lowercase().replace([' ', '_', '-'], "");
        Tool::ALL.into_iter().find(|t| n(t.id()) == n(s) || n(t.label()) == n(s))
    }
}

/// What the viewport displays.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum ViewMode {
    /// The lit material.
    #[default]
    Material,
    /// One channel, unlit.
    Channel(Channel),
    /// A baked mesh map.
    MeshMap(texturecraft_core::BakedMap),
    /// UV checker (texel density and seams).
    UvChecker,
    /// One layer's mask, in grayscale (Alt+click a mask).
    Mask(LayerId),
}

impl ViewMode {
    pub fn label(&self) -> String {
        match self {
            ViewMode::Material => "Material".into(),
            ViewMode::Channel(c) => c.label().into(),
            ViewMode::MeshMap(m) => m.id().into(),
            ViewMode::UvChecker => "UV checker".into(),
            ViewMode::Mask(_) => "Layer mask".into(),
        }
    }
}

/// A screen-pinned image for the Projection tool (a stencil).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StencilState {
    /// Image resource name.
    pub image: String,
    /// Center in normalized device coordinates (-1..1, y up).
    pub center: [f32; 2],
    /// Height of the image as a fraction of the viewport height (1 = full height).
    pub scale: f32,
    /// Rotation in degrees.
    pub rotation: f32,
    /// Repeat the image across the screen.
    pub tile: bool,
    /// Overlay opacity in the viewport (display only).
    pub opacity: f32,
}

impl StencilState {
    pub fn new(image: &str) -> StencilState {
        StencilState::with_look(image, StencilLook::default())
    }

    /// A stencil of `image` placed as `look` says.
    pub fn with_look(image: &str, look: StencilLook) -> StencilState {
        StencilState { image: image.to_string(), center: look.center, scale: look.scale, rotation: look.rotation, tile: look.tile, opacity: look.opacity }
    }

    pub fn look(&self) -> StencilLook {
        StencilLook { center: self.center, scale: self.scale, rotation: self.rotation, tile: self.tile, opacity: self.opacity }
    }

    /// Back to the placement a new stencil starts with (center, scale and rotation); the image, tiling and opacity stay.
    pub fn reset_placement(&mut self) {
        let start = StencilState::new(&self.image);
        self.center = start.center;
        self.scale = start.scale;
        self.rotation = start.rotation;
    }

    /// Placement for a viewport aspect (width / height): center, half extents in NDC, radians.
    pub fn placement(&self, image_aspect: f32, viewport_aspect: f32) -> ([f32; 2], [f32; 2], f32) {
        let hy = self.scale.max(0.01);
        let hx = hy * image_aspect / viewport_aspect.max(1e-3);
        (self.center, [hx, hy], self.rotation.to_radians())
    }
}

/// Tool settings shared by the UI, CLI and agents.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolState {
    pub tool: Tool,
    pub brush: BrushSettings,
    /// Material painted per channel (linear values) and whether each channel is painted.
    pub values: BTreeMap<Channel, [f32; 3]>,
    pub enabled: BTreeSet<Channel>,
    /// The simple material (shelf id) the brush paints the pattern of, with `values` as its plain
    /// stand-in; `None` for plain values (any edit of a value clears it).
    #[serde(default)]
    pub material: Option<String>,
    /// Gray painted into masks.
    pub mask_value: f32,
    pub fill_mode: FillMode,
    /// How a polygon fill box decides which polygons it selects
    #[serde(default)]
    pub fill_hit_test: crate::fill_hit_test::FillHitTest,
    /// Polygon fill boxes in the 3D view select only the polygons the camera sees
    #[serde(default = "default_fill_visible_only")]
    pub fill_visible_only: bool,
    /// Stencil used by the Projection tool.
    #[serde(default)]
    pub stencil: Option<StencilState>,
    /// Placement the next stencil starts with: the last stencil's, kept when the stencil is
    /// cleared or the project changes (its image belongs to the project, the placement does not).
    #[serde(default)]
    pub stencil_look: StencilLook,
    /// Smudge tool: how much of the picked-up values each stamp deposits (0..1).
    #[serde(default = "default_smudge_strength")]
    pub smudge_strength: f32,
    /// Clone tool source (V + click). It belongs to the active texture set: switching the set
    /// (`textureSet.select`, `layer.select` into another set) clears it.
    #[serde(default)]
    pub clone_source: Option<CloneSource>,
    /// Clone tool: keep the offset of the first stroke for the following strokes (aligned, the
    /// default). Off, every stroke starts sampling at the source point.
    #[serde(default = "default_clone_aligned")]
    pub clone_aligned: bool,
    /// What the Clone tool copies: the composite or the selected layer alone.
    #[serde(default)]
    pub clone_from: CloneFrom,
    /// The Path tool's curve being edited.
    #[serde(default)]
    pub path: crate::path_tool::PathState,
}

pub(crate) fn default_smudge_strength() -> f32 {
    0.5
}

fn default_clone_aligned() -> bool {
    true
}

fn default_fill_visible_only() -> bool {
    true
}

/// What the Clone tool copies (the clone source modes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum CloneFrom {
    /// Everything the texture set shows.
    #[default]
    Composite,
    /// The selected layer only (its content and mask; empty parts copy nothing).
    Layer,
}

/// Where the Clone tool copies from, in texture coordinates.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct CloneSource {
    pub uv: [f32; 2],
    /// Source minus destination, fixed by the first stroke after the source was set and kept
    /// for the following strokes (aligned cloning; never set when the tool is not aligned).
    pub offset: Option<[f32; 2]>,
}

impl Default for ToolState {
    fn default() -> Self {
        let values = Channel::ALL.iter().map(|c| (*c, c.default_value())).collect();
        let mut t = ToolState {
            tool: Tool::Paint,
            brush: BrushSettings::default(),
            values,
            enabled: [Channel::BaseColor, Channel::Metallic, Channel::Roughness, Channel::Height, Channel::Normal].into_iter().collect(),
            material: None,
            mask_value: 1.0,
            fill_mode: FillMode::Triangle,
            fill_hit_test: Default::default(),
            fill_visible_only: default_fill_visible_only(),
            stencil: None,
            stencil_look: StencilLook::default(),
            smudge_strength: default_smudge_strength(),
            clone_source: None,
            clone_aligned: true,
            clone_from: CloneFrom::Composite,
            path: Default::default(),
        };
        t.values.insert(Channel::BaseColor, [0.8, 0.8, 0.8]);
        t.values.insert(Channel::Roughness, [0.3; 3]);
        t
    }
}

/// What is selected in the layer stack.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum StackTarget {
    #[default]
    Content,
    /// The layer's mask (painting adds/uses a paint effect).
    Mask,
    /// A specific effect.
    Effect(LayerId),
}

pub(crate) struct ActiveStroke {
    pub ts: usize,
    /// None: the stroke edits the quick mask.
    pub target: Option<PaintTarget>,
    pub stroke: Stroke,
    pub canvas: StrokeCanvas,
    /// Camera visibility for 3D strokes (None for strokes in the 2D view).
    pub depth: Option<Arc<EyeDepth>>,
    /// 2D view strokes: where the stroke is between samples.
    pub uv: Option<crate::uv_stroke_state::UvStrokeState>,
    pub camera: OrbitCamera,
    pub viewport: Vec2,
    pub values: BTreeMap<Channel, [f32; 3]>,
    pub erase: bool,
    /// Projection tool: the stencil captured with the stroke's camera.
    pub stencil: Option<texturecraft_compose::Stencil>,
    /// History snapshot taken when the stroke began.
    pub before: Document,
    pub label: String,
    /// The commit's destination tiles, copied as the stroke reaches them.
    pub staging: texturecraft_paint::Staging,
}

/// A stroke visibility buffer and what it was rendered for: camera, size, mesh id.
pub(crate) type DepthCache = (OrbitCamera, (u32, u32), u64, Arc<EyeDepth>);

/// History labels that name a layer, a path number or a smart mask: the UI translates the
/// template and fills the value in (the control channel and `edit.history` keep the English label).
/// `{layer}` is a layer name, `{n}` a path number, `{name}` a smart mask name.
pub const HISTORY_TEMPLATES: &[&str] = &[
    "Paint on {layer}",
    "Erase on {layer}",
    "Clone on {layer}",
    "Smudge on {layer}",
    "Projection on {layer}",
    "Paint mask of {layer}",
    "Erase mask of {layer}",
    "Projection mask of {layer}",
    EDIT_PATH_TEMPLATE,
    APPLY_SMART_MASK_TEMPLATE,
];

pub(crate) const EDIT_PATH_TEMPLATE: &str = "Edit path {n}";
pub(crate) const APPLY_SMART_MASK_TEMPLATE: &str = "Apply smart mask {name}";
pub(crate) const QUICK_MASK_STROKE_LABEL: &str = "Quick mask stroke";

/// `template` with its one `{...}` placeholder replaced by `value`
pub fn history_label(template: &str, value: &str) -> String {
    match (template.find('{'), template.find('}')) {
        (Some(open), Some(close)) if open < close => format!("{}{}{}", &template[..open], value, &template[close + 1..]),
        _ => template.to_string(),
    }
}

/// The history template of a stroke by `tool`, on a layer or (`on_mask`) on its mask
pub fn stroke_template(tool: Tool, on_mask: bool) -> &'static str {
    match (tool, on_mask) {
        (Tool::Eraser, false) => "Erase on {layer}",
        (Tool::Eraser, true) => "Erase mask of {layer}",
        (Tool::Clone, _) => "Clone on {layer}",
        (Tool::Smudge, _) => "Smudge on {layer}",
        (Tool::Projection, false) => "Projection on {layer}",
        (Tool::Projection, true) => "Projection mask of {layer}",
        (_, false) => "Paint on {layer}",
        (_, true) => "Paint mask of {layer}",
    }
}

pub struct Session {
    pub doc: Document,
    pub mesh: Arc<Mesh>,
    /// Unique per loaded mesh and UV layout (GPU caches compare it).
    pub mesh_id: u64,
    /// Unique per open project: a mesh reimport or UV layout change keeps it, and opening, closing
    /// or creating a project replaces it.
    pub project_id: u64,
    /// The mesh with its own UVs (`mesh` maps UV-tiled sets into their atlas).
    pub(crate) source: Arc<Mesh>,
    pub bvh: Arc<Bvh>,
    /// Original mesh file (saved inside the project); None for built-in meshes.
    pub mesh_source: Option<(String, Vec<u8>)>,
    /// High-definition meshes and cage for baking (saved inside the project).
    pub bake_meshes: crate::BakeMeshes,
    pub caches: Vec<SetCache>,
    pub(crate) undo: Vec<(String, Document)>,
    pub(crate) redo: Vec<(String, Document)>,
    /// Set by [`crate::execute`] from a command's `merge` param: its edits continue the previous
    /// undo step with the same label (a slider drag is one step however many values it applies).
    pub(crate) merge_edits: bool,
    pub tool: ToolState,
    pub active_set: usize,
    /// Selected layer per texture set.
    pub selected: BTreeMap<usize, LayerId>,
    /// Layers selected together with the selected one (Shift+click, Ctrl+click), per texture set.
    /// Counts only while the selected layer is the one they were picked with (see [`Session::selected_layers`]).
    pub(crate) multi: BTreeMap<usize, crate::layer_selection::MultiSelection>,
    pub target: StackTarget,
    pub camera: OrbitCamera,
    pub view: ViewMode,
    /// The layer shown alone in the viewport (not saved, not in the history; see [`crate::solo`]).
    pub solo: Option<crate::Solo>,
    /// The live bake preview while the app bakes (see [`crate::bake_preview`]).
    pub bake_preview: Option<crate::bake_preview::BakePreview>,
    /// Viewport size (pixels) used by scripted strokes.
    pub viewport: Vec2,
    pub(crate) stroke: Option<ActiveStroke>,
    /// Layer clipboard: a document snapshot and the copied layer (or effect).
    pub clipboard: Option<(Document, LayerId)>,
    /// Decoded image resources (rebuilt from `doc.resources` by `Session::sync_images`).
    pub images: texturecraft_compose::ImageStore,
    /// Temporary texel mask limiting paint strokes (not saved, not in the history).
    pub quick_mask: crate::quick_mask::QuickMask,
    /// A path sublayer is being repainted from its paths (the only way to paint into one).
    pub(crate) painting_paths: bool,
    /// Pixels of the paths below the one being edited, so live edits skip repainting them.
    pub(crate) path_cache: Option<crate::path_tool::PathCache>,
    /// A new path's live preview: the path becomes an undo step only when it is applied
    pub(crate) path_draft: Option<crate::path_draft::PathDraft>,
    /// Point changes of the path being edited: (earlier states for undo, undone states for redo)
    pub(crate) path_steps: (Vec<crate::path_tool::PathState>, Vec<crate::path_tool::PathState>),
    pub path: Option<String>,
    /// False in the empty state (the app at start, or after File > Close): the document is a
    /// tiny placeholder that panels do not show. Project-independent commands remain available
    /// (see [`crate::commands::check_enabled`]).
    pub project_open: bool,
    pub modified: bool,
    /// Bumps on every document change.
    pub revision: u64,
    /// Messages for the UI (status bar / toasts).
    pub log: Vec<String>,
    /// The last stroke's camera visibility buffer: reused while the camera, view size and mesh
    /// are unchanged, so back-to-back dabs start instantly.
    pub(crate) depth_cache: Option<DepthCache>,
    /// The polygon fill box being dragged
    pub(crate) polygon_fill_drag: Option<crate::polygon_fill_drag::PolygonFillDrag>,
    /// The user's export presets (kept when another project opens).
    pub export_presets: crate::preset_cmds::UserPresets,
    /// The user's brush presets (kept when another project opens).
    pub brush_presets: crate::brush_preset_cmds::UserBrushes,
    /// The user's project templates (kept when another project opens).
    pub templates: crate::template_cmds::UserTemplates,
}

impl Session {
    /// A session for `project` on `mesh` (its sets must reference mesh materials), with UV-tiled sets' UVs mapped into their atlas ([`Session::source_mesh`])
    pub fn new(doc: Document, mesh: Mesh, mesh_source: Option<(String, Vec<u8>)>) -> Result<Session> {
        mesh.validate().map_err(|error| EngineError::Other(error.to_string()))?;

        let source = Arc::new(mesh);
        let mesh = tiled_mesh(&source, &doc.project);
        let bvh = Bvh::new(&mesh);
        let mut camera = OrbitCamera { yaw: 0.6, pitch: 0.25, ..OrbitCamera::default() };
        let (bounds_min, bounds_max) = mesh.bounds();
        camera.frame(bounds_min, bounds_max);
        let caches: Vec<SetCache> = doc.project.texture_sets.iter().map(|texture_set| SetCache::new(&mesh, texture_set)).collect();
        share_pattern_budget(&caches);
        let selected = doc.project.texture_sets.iter().enumerate().filter_map(|(index, texture_set)| texture_set.stack.last().map(|id| (index, *id))).collect();

        Ok(Session {
            doc,
            mesh_id: next_mesh_id(),
            project_id: next_mesh_id(),
            source,
            mesh,
            bvh: Arc::new(bvh),
            mesh_source,
            bake_meshes: Default::default(),
            caches,
            undo: Vec::new(),
            redo: Vec::new(),
            merge_edits: false,
            tool: ToolState::default(),
            active_set: 0,
            selected,
            multi: BTreeMap::new(),
            target: StackTarget::Content,
            camera,
            view: ViewMode::Material,
            solo: None,
            bake_preview: None,
            viewport: Vec2::new(1280.0, 800.0),
            stroke: None,
            clipboard: None,
            images: Default::default(),
            quick_mask: Default::default(),
            painting_paths: false,
            path_cache: None,
            path_draft: None,
            path_steps: Default::default(),
            path: None,
            project_open: true,
            modified: false,
            revision: 1,
            log: Vec::new(),
            depth_cache: None,
            polygon_fill_drag: None,
            export_presets: Default::default(),
            brush_presets: Default::default(),
            templates: Default::default(),
        })
    }

    /// The empty state: no project is open. Holds a tiny placeholder document that the UI never
    /// shows; project-dependent commands are refused.
    pub fn closed() -> Result<Session> {
        // The hidden placeholder uses the minimum supported texture resolution
        let mut session = Session::new_project("", texturecraft_geom::primitives::plane(1.0, 1), "builtin:plane", None, texturecraft_core::MIN_RESOLUTION)?;
        session.project_open = false;

        Ok(session)
    }

    /// Replace this session with `next` (another project), keeping the viewport size, the
    /// user's brush presets, export presets and project templates and the tool settings.
    pub fn switch_to(&mut self, mut next: Session) {
        // Invalidate caches even when both projects were loaded with the same revision
        next.revision = self.revision.wrapping_add(1);

        next.viewport = self.viewport;
        next.export_presets = std::mem::take(&mut self.export_presets);
        next.brush_presets = std::mem::take(&mut self.brush_presets);
        next.templates = std::mem::take(&mut self.templates);
        crate::tool_settings::ToolSettings::of(&self.tool).apply(&mut next.tool);
        *self = next;
    }

    /// A new project: one texture set per mesh material, each with an empty paint layer. Sets
    /// whose UVs leave tile 1001 paint UV tiles (see [`Session::new_project_tiled`]).
    pub fn new_project(name: &str, mesh: Mesh, mesh_ref: &str, mesh_source: Option<(String, Vec<u8>)>, resolution: u32) -> Result<Session> {
        Session::new_project_tiled(name, mesh, mesh_ref, mesh_source, resolution, None)
    }

    /// [`Session::new_project`] choosing the UV tile (UDIM) mode: `Some(true)` for every
    /// set, `Some(false)` for none, `None` for the sets whose UVs leave tile 1001.
    pub fn new_project_tiled(
        name: &str,
        mesh: Mesh,
        mesh_ref: &str,
        mesh_source: Option<(String, Vec<u8>)>,
        resolution: u32,
        uv_tiles: Option<bool>,
    ) -> Result<Session> {
        let res = check_resolution(resolution)?;
        let mut project = Project::new(name, mesh_ref);
        let mats = if mesh.materials.is_empty() { vec!["Default".to_string()] } else { mesh.materials.clone() };
        let mut log = Vec::new();
        for (i, m) in mats.iter().enumerate() {
            let mut ts = TextureSet::new(m.clone(), i, res);
            ts.uv_tiles = detect_uv_tiles(&mesh, i, uv_tiles, &mut log).map_err(|e| EngineError::Other(format!("texture set \"{m}\": {e}")))?;
            check_size(&ts)?;
            let id = project.alloc_id();
            let layer = Layer::new(id, "Layer 1", LayerKind::Paint, &ts.channel_list());
            project.layers.insert(id, layer);
            ts.stack.push(id);
            project.texture_sets.push(ts);
        }
        let mut s = Session::new(
            Document { bake_paint: BTreeMap::new(), project, pixels: PixelStore::default(), baked: BTreeMap::new(), resources: BTreeMap::new() },
            mesh,
            mesh_source,
        )?;
        s.log.extend(log);
        Ok(s)
    }

    /// The mesh with its own UVs (UV-tiled sets' UVs outside their atlas), for mesh export.
    pub fn source_mesh(&self) -> Mesh {
        (*self.source).clone()
    }

    pub fn project(&self) -> &Project {
        &self.doc.project
    }

    pub fn set_count(&self) -> usize {
        self.doc.project.texture_sets.len()
    }

    /// The active texture set index, clamped.
    pub fn ts(&self) -> usize {
        self.active_set.min(self.set_count().saturating_sub(1))
    }

    pub fn selected_layer(&self) -> Option<LayerId> {
        let ts = self.ts();
        self.selected.get(&ts).copied().filter(|id| self.doc.project.locate(*id).is_some_and(|(t, _, _)| t == ts))
    }

    // -----------------------------------------------------------------------------------------
    // History

    /// Run an undoable edit. On error the document is left untouched.
    pub fn edit<T>(&mut self, label: &str, f: impl FnOnce(&mut Session) -> Result<T>) -> Result<T> {
        self.edit_merging(label, self.merge_edits, f)
    }

    /// [`Session::edit`] that, with `merge`, continues the previous undo step when it has the same label (a viewport drag is one step however many updates it makes).
    pub fn edit_merging<T>(&mut self, label: &str, merge: bool, change: impl FnOnce(&mut Session) -> Result<T>) -> Result<T> {
        self.edit_if_changed(label, merge, |s| change(s).map(|value| (value, true)))
    }

    /// A conditional edit shares the usual rollback and history snapshot with layer reorders.
    pub(crate) fn edit_if_changed<T>(&mut self, label: &str, merge: bool, change: impl FnOnce(&mut Session) -> Result<(T, bool)>) -> Result<T> {
        // A box fill under way becomes its own step first, so this edit isn't written over or folded into it
        self.end_polygon_fill_box();

        let before = self.doc.clone();
        let result = change(self).and_then(|(value, changed)| {
            if changed && let Some(id) = self.doc.project.instance_cycle() {
                return Err(EngineError::Other(format!("instance layer {} would draw itself", id.0)));
            }

            Ok((value, changed))
        });
        match result {
            Ok((value, changed)) => {
                if changed {
                    self.record(label, before, merge);
                }

                Ok(value)
            }
            Err(error) => {
                self.doc = before;

                // A failed repaint may have cached tiles from the discarded document
                self.path_cache = None;
                Err(error)
            }
        }
    }

    pub(crate) fn push_history(&mut self, label: &str, before: Document) {
        self.record(label, before, false);
    }

    /// A change from `before`: a new undo step `label`, or with `merge` part of the previous one when it has the same label
    fn record(&mut self, label: &str, before: Document, merge: bool) {
        self.invalidate_changed(&before);
        self.push_step(label, before, merge);
    }

    /// [`Session::record`] for a change whose tiles the caller already marked for recompositing
    pub(crate) fn push_step(&mut self, label: &str, before: Document, merge: bool) {
        // Another change during a new path's preview makes the path its own step first
        if !self.painting_paths
            && let Some(draft) = self.path_draft.take()
        {
            self.undo.push((crate::path_tool::NEW_PATH_LABEL.to_string(), draft.base));
        }

        // Point changes of an edited path undo only until another step lands
        if !self.painting_paths {
            self.path_steps = Default::default();
        }

        self.modified = true;
        self.revision = self.revision.wrapping_add(1);
        if !(merge && self.undo.last().is_some_and(|(l, _)| l == label)) {
            self.undo.push((label.to_string(), before));
        }
        let over = self.undo.len().saturating_sub(MAX_HISTORY);
        self.undo.drain(..over);
        self.redo.clear();
    }

    /// A change from `other` to the current document that is not an undo step (a path preview)
    pub(crate) fn changed_without_history(&mut self, other: &Document) {
        self.modified = true;
        self.revision = self.revision.wrapping_add(1);
        self.invalidate_changed(other);
    }

    /// Mark dirty every texture set whose composite differs between the previous and current
    /// document (after an edit, undo or redo).
    pub(crate) fn invalidate_changed(&mut self, previous_document: &Document) {
        self.follow_uv_layouts(&previous_document.project);
        self.sync_caches();

        // Removed path effects must release their cached painted tiles
        if self.path_cache.as_ref().is_some_and(|cache| self.paths(cache.eid).is_none()) {
            self.path_cache = None;
        }

        // Removed or replaced resources may change any set's rendered content
        let removed_or_replaced_resource =
            previous_document.resources.iter().any(|(name, previous)| !self.doc.resources.get(name).is_some_and(|current| Arc::ptr_eq(previous, current)));
        let added_resource_names: BTreeSet<_> = self.doc.resources.keys().filter(|name| !previous_document.resources.contains_key(*name)).collect();
        let uses_added_resource = |texture_set_index: usize| {
            use texturecraft_core::{EffectKind, Fill, Source};

            let uses_source = |source: &Source| source.images().into_iter().any(|name| added_resource_names.iter().any(|added| added.as_str() == name));
            let uses_fill =
                |fill: &Fill| fill.sources.values().any(uses_source) || fill.link.as_ref().is_some_and(|link| added_resource_names.contains(&link.package));
            let project = &self.doc.project;
            !added_resource_names.is_empty()
                && project.texture_sets.get(texture_set_index).is_some_and(|set| {
                    project.render_tree(&set.stack).iter().filter_map(|id| project.layers.get(id)).any(|layer| {
                        matches!(&layer.kind, LayerKind::Fill(fill) if uses_fill(fill))
                            || layer.mask.iter().flat_map(|mask| &mask.effects).chain(&layer.content_effects).any(|effect| match &effect.kind {
                                EffectKind::Fill { source, .. } => uses_source(source),
                                EffectKind::Material(fill) => uses_fill(fill),
                                EffectKind::Generator(generator) => generator.inputs.values().any(uses_source),
                                _ => false,
                            })
                    })
                })
        };
        let (empty_baked_maps, empty_images) = (BakedMaps::default(), Default::default());
        let stack_key = |document: &Document, texture_set_index: usize| {
            let stack = document.project.texture_sets.get(texture_set_index).map(|set| set.stack.as_slice()).unwrap_or_default();
            StackKey::new(
                &document.project,
                &document.pixels,
                document.baked.get(&texture_set_index).unwrap_or(&empty_baked_maps),
                &empty_images,
                texture_set_index,
                stack,
            )
        };
        for texture_set_index in 0..self.caches.len() {
            let before = stack_key(previous_document, texture_set_index);
            let after = stack_key(&self.doc, texture_set_index);
            let resource_changed = removed_or_replaced_resource || uses_added_resource(texture_set_index);
            let Some(cache) = self.caches.get_mut(texture_set_index) else {
                continue;
            };

            if resource_changed {
                cache.invalidate_all();
                continue;
            }

            if before == after {
                continue;
            }

            // Neighbourhood readers require a full recomposite after painted pixels change
            let changed_tiles = match (&before, &after) {
                (Some(before), Some(after))
                    if !stack_reads_neighbours(&self.doc.project, texture_set_index)
                        && !stack_reads_neighbours(&previous_document.project, texture_set_index) =>
                {
                    before.changed_tiles(after)
                }
                _ => None,
            };
            match changed_tiles {
                Some(changed_tiles) => {
                    // covered_tiles is emitted in row order
                    cache
                        .dirty
                        .extend(changed_tiles.into_iter().filter(|tile| cache.covered_tiles.binary_search_by_key(&(tile.1, tile.0), |&(x, y)| (y, x)).is_ok()));
                }
                None => cache.invalidate_all(),
            }
        }
    }

    /// After the UV tile layout of a texture set changed (UV tiles on or off, tile
    /// resolutions) between `before` and the current project, map the mesh's UVs into the new
    /// atlas and rebuild those sets' caches.
    fn follow_uv_layouts(&mut self, before: &Project) {
        let layout = |p: &Project, ts: usize| p.texture_sets.get(ts).map(uv_layout);
        let changed: Vec<usize> = (0..self.doc.project.texture_sets.len()).filter(|ts| layout(before, *ts) != layout(&self.doc.project, *ts)).collect();
        if changed.is_empty() && before.texture_sets.len() == self.doc.project.texture_sets.len() {
            return;
        }
        self.mesh = tiled_mesh(&self.source, &self.doc.project);
        self.mesh_id = next_mesh_id();
        self.depth_cache = None;
        for ts in changed {
            if let (Some(c), Some(set)) = (self.caches.get_mut(ts), self.doc.project.texture_sets.get(ts)) {
                *c = SetCache::new(&self.mesh, set);
            }
        }
        share_pattern_budget(&self.caches);
    }

    /// Mark the document changed and every texture set dirty.
    pub fn touched(&mut self) {
        self.modified = true;
        self.revision = self.revision.wrapping_add(1);
        self.sync_caches();
        for c in &mut self.caches {
            c.invalidate_all();
        }
    }

    /// Make caches agree with the project's texture sets (resolution or channel changes).
    pub fn sync_caches(&mut self) {
        let sets = self.doc.project.texture_sets.clone();
        self.caches.truncate(sets.len());
        for (i, ts) in sets.iter().enumerate() {
            match self.caches.get_mut(i) {
                Some(c) if c.size == ts.size() => {
                    let (want, scales) = (ts.channel_list(), ts.block_scales());
                    if c.composite.channels.keys().copied().collect::<Vec<_>>() != want || c.composite.block_scales != scales {
                        c.composite = Composite { block_scales: scales, ..Composite::with_size(&want, c.size.0, c.size.1) };
                        c.upload.extend(texturecraft_compose::all_tiles(c.size.0, c.size.1));
                        c.invalidate_all();
                    }
                }
                _ => {
                    let cache = SetCache::new(&self.mesh, ts);
                    if i < self.caches.len() {
                        self.caches[i] = cache;
                    } else {
                        self.caches.push(cache);
                    }
                }
            }
        }
        share_pattern_budget(&self.caches);
    }

    /// Whether a point edit, an uncommitted path preview, or a document edit can be undone
    pub fn can_undo(&self) -> bool {
        !self.undo.is_empty() || self.path_draft.is_some() || self.path_step_pending(true)
    }

    /// Whether a point edit or document edit can be restored
    pub fn can_redo(&self) -> bool {
        !self.redo.is_empty() || self.path_step_pending(false)
    }

    /// Label of the next undo action, giving an active path's point edits and preview priority
    pub fn undo_label(&self) -> Option<&str> {
        if self.path_draft.is_some() || self.path_step_pending(true) {
            return Some(crate::path_tool::NEW_PATH_LABEL);
        }
        self.undo.last().map(|(l, _)| l.as_str())
    }

    /// Label of the next redo action, giving an active path's point history priority
    pub fn redo_label(&self) -> Option<&str> {
        if self.path_step_pending(false) {
            return Some(crate::path_tool::NEW_PATH_LABEL);
        }
        self.redo.last().map(|(l, _)| l.as_str())
    }

    /// The undone steps, the next one to redo first.
    pub fn redo_labels(&self) -> Vec<String> {
        self.redo.iter().rev().map(|(l, _)| l.clone()).collect()
    }
    pub fn history_labels(&self) -> Vec<String> {
        self.undo.iter().map(|(l, _)| l.clone()).collect()
    }

    /// Undo a point edit, cancel an uncommitted path, or restore a document snapshot, in that order
    pub fn undo(&mut self) -> Result<String> {
        // A stroke, a box fill or an edited path under way would otherwise write over the restored document
        self.cancel_stroke();
        self.end_polygon_fill_box();

        // While a path is edited, undo steps back through its point changes first
        if self.step_path(true)? {
            return Ok(crate::path_tool::NEW_PATH_LABEL.to_string());
        }

        // A path being placed isn't a step yet: undo removes it
        if self.path_draft.is_some() {
            self.stop_path_edit();
            return Ok(crate::path_tool::NEW_PATH_LABEL.to_string());
        }

        self.stop_path_edit();

        let (label, restored) = self.undo.pop().ok_or_else(|| EngineError::Other("nothing to undo".into()))?;
        let current = std::mem::replace(&mut self.doc, restored);
        self.after_history(&current);
        self.redo.push((label.clone(), current));
        self.sync_images();
        Ok(label)
    }

    /// Restore a point edit and its pixels before considering document redo
    pub fn redo(&mut self) -> Result<String> {
        // A stroke, a box fill or an edited path under way would otherwise write over the restored document
        self.cancel_stroke();
        self.end_polygon_fill_box();

        if self.step_path(false)? {
            return Ok(crate::path_tool::NEW_PATH_LABEL.to_string());
        }

        self.stop_path_edit();

        let (label, restored) = self.redo.pop().ok_or_else(|| EngineError::Other("nothing to redo".into()))?;
        let current = std::mem::replace(&mut self.doc, restored);
        self.after_history(&current);
        self.undo.push((label.clone(), current));
        self.sync_images();
        Ok(label)
    }

    fn after_history(&mut self, other: &Document) {
        self.modified = true;

        // Layers picked together belong to the stack they were picked in
        self.multi.clear();
        self.revision = self.revision.wrapping_add(1);
        self.invalidate_changed(other);

        // An undone add or duplicate takes the selected effect away
        if let StackTarget::Effect(e) = self.target
            && self.doc.project.effect(e).is_none()
        {
            let p = &other.project;
            let content = p.effect(e).and_then(|(owner, _)| p.layers.get(&owner)).is_some_and(|l| l.content_effects.iter().any(|x| x.id == e));
            self.target = if content { StackTarget::Content } else { StackTarget::Mask };
        }
        let has_mask = self.selected_layer().and_then(|lid| self.doc.project.layer(self.doc.project.mask_owner(lid)).ok()).is_some_and(|l| l.mask.is_some());
        if self.target == StackTarget::Mask && !has_mask {
            self.target = StackTarget::Content;
        }
    }

    // -----------------------------------------------------------------------------------------
    // Compositing

    /// Recomposite dirty tiles of every texture set (with the live stroke preview). Returns
    /// whether anything changed.
    pub fn update(&mut self) -> bool {
        self.update_for(f64::INFINITY, || 0.0)
    }

    /// [`Session::update`] within about `budget_ms` of `clock` (milliseconds): tiles go in
    /// chunks sized from the time the previous chunk took, and what is left stays dirty for the
    /// next call (at least one chunk runs). The UI calls this every frame so a full recomposite
    /// of a large texture set spreads over frames instead of freezing. The active texture set
    /// goes first; settling tiles (see [`SetCache::settle`]) get at most [`SETTLE_BUDGET_MS`]
    /// of a finite budget, after the dirty ones.
    pub fn update_for(&mut self, budget_ms: f64, clock: impl Fn() -> f64) -> bool {
        let start = clock();
        let initial_chunk_size = if budget_ms.is_finite() { rayon::current_num_threads().clamp(1, CHUNK) } else { CHUNK };

        // The layer being edited: everything else is cached per tile
        let hot = self.stroke.as_ref().and_then(|stroke| stroke.target).map(|target| self.paint_owner(target)).or_else(|| self.selected_layer());
        let stroke_size = self.stroke.as_ref().and_then(|stroke| self.caches.get(stroke.ts)).map(|cache| cache.size);
        let preview = self.stroke.as_ref().and_then(|stroke| {
            Some(StrokePreview {
                target: stroke.target?,
                alpha: stroke.canvas.alpha.clone(),
                opacity: stroke.stroke.settings.opacity,
                erase: stroke.erase,
                values: stroke.values.clone(),
                mask_value: self.tool.mask_value,
                stencil: stroke.stencil.clone(),
                carried_mask: stroke.canvas.pickup.as_ref().and_then(|pickup| pickup.mask_values()).cloned(),
                carried: stroke.canvas.pickup.as_ref().map(|pickup| pickup.values.clone()).unwrap_or_default(),
            })
        });

        // A soloed layer's view of the project; a removed layer ends the solo
        if self.solo.is_some_and(|solo| self.doc.project.locate(solo.layer).is_none()) {
            self.set_solo(None);
        }
        let solo_view = if self.pending_tiles() > 0 { self.solo_view() } else { None };
        let active = self.ts();
        let order = std::iter::once(active).chain((0..self.caches.len()).filter(|ts| *ts != active));
        let mut changed = false;
        for ts in order {
            let Some(cache) = self.caches.get_mut(ts) else { continue };
            cache.below.set_hot(hot);
            cache.below.pass = cache.dirty.len() + cache.settle.len();

            // Instances in sets of the stroke's size show the live stroke too
            let preview = preview.as_ref().filter(|_| Some(cache.size) == stroke_size);

            // Reuse the last affordable chunk size for heavy stacks
            let mut chunk = if budget_ms.is_finite() && cache.chunk > 0 { cache.chunk } else { initial_chunk_size };
            loop {
                if cache.dirty.is_empty() {
                    if cache.settle.is_empty() || clock() - start >= budget_ms.min(SETTLE_BUDGET_MS) {
                        break;
                    }

                    // Small chunks: the settle slice of the budget is short
                    cache.dirty.extend(next_chunk(&mut cache.settle, chunk.min(initial_chunk_size)));
                }
                let chunk_started = clock();
                let tiles = next_chunk(&mut cache.dirty, chunk);
                for tile in &tiles {
                    cache.settle.remove(tile);
                }

                let empty = BakedMaps::default();
                let inputs = texturecraft_compose::Inputs {
                    project: solo_view.as_ref().unwrap_or(&self.doc.project),
                    ts,
                    mesh: &self.mesh,
                    gbuf: &cache.gbuf,
                    pixels: &self.doc.pixels,
                    baked: self.doc.baked.get(&ts).unwrap_or(&empty),
                    block_scales: cache.composite.block_scales.clone(),
                    preview,
                    images: &self.images,
                    bounds: self.mesh.bounds(),
                    patterns: Some(&cache.patterns),
                };

                // Look ahead so sliced updates filter a run of tiles once
                cache.below.ahead = if budget_ms.is_finite() { tiles_ahead(&cache.dirty, &tiles) } else { Vec::new() };
                texturecraft_compose::composite_tiles_cached(&inputs, &mut cache.composite, &tiles, Some(&mut cache.below));

                // The composited tiles and the neighbours their padding reached change
                let padded = pad_tiles(&mut cache.composite, &cache.dilation, &tiles);
                cache.upload.extend(tiles.iter().copied().chain(padded));
                cache.revision = cache.revision.wrapping_add(1);
                changed = true;

                let now = clock();
                let remaining_ms = budget_ms - (now - start);

                // Estimate how many tiles fit in the remaining frame budget
                let ms_per_tile = (now - chunk_started) / tiles.len().max(1) as f64;
                if budget_ms.is_finite() && ms_per_tile > 0.0 {
                    // Never above one tile per thread: work can turn heavy from one frame to the next
                    cache.chunk = ((budget_ms / ms_per_tile) as usize).clamp(1, initial_chunk_size);
                }
                if remaining_ms <= 0.0 {
                    return changed;
                }

                let fitting_tiles = if ms_per_tile > 0.0 { (remaining_ms / ms_per_tile) as usize } else { CHUNK };
                if fitting_tiles == 0 {
                    return changed;
                }

                chunk = fitting_tiles.min(CHUNK);
            }
        }

        changed
    }

    /// Tiles still waiting to be composited, over every texture set (settling ones included).
    pub fn pending_tiles(&self) -> usize {
        self.caches.iter().map(|c| c.dirty.len() + c.settle.len()).sum()
    }

    /// The layer a stroke target belongs to (an effect's owner).
    pub(crate) fn paint_owner(&self, t: PaintTarget) -> LayerId {
        let (PaintTarget::Layer(id) | PaintTarget::Mask(id)) = t;
        self.doc.project.effect(id).map_or(id, |(owner, _)| owner)
    }

    /// Mark `tiles` of texture set `ts` dirty after painting `target`, and the same tiles of
    /// other sets of that size drawing it through an instance.
    pub(crate) fn mark_painted(&mut self, ts: usize, target: Option<PaintTarget>, tiles: impl IntoIterator<Item = (u32, u32)>) {
        self.mark_tiles(ts, target, tiles, |c| &mut c.dirty);
    }

    /// [`Session::mark_painted`] for the tiles of a committed stroke: they settle at low
    /// priority (see [`SetCache::settle`]).
    pub(crate) fn mark_settle(&mut self, ts: usize, target: Option<PaintTarget>, tiles: impl IntoIterator<Item = (u32, u32)>) {
        self.mark_tiles(ts, target, tiles, |c| &mut c.settle);
    }

    fn mark_tiles(
        &mut self,
        ts: usize,
        target: Option<PaintTarget>,
        tiles: impl IntoIterator<Item = (u32, u32)>,
        set: impl Fn(&mut SetCache) -> &mut BTreeSet<(u32, u32)>,
    ) {
        let size = self.caches.get(ts).map(|c| c.size);
        let sets = target.map(|t| self.doc.project.sets_drawing(self.paint_owner(t))).unwrap_or_default();
        let tiles: Vec<(u32, u32)> = tiles.into_iter().collect();
        for (i, c) in self.caches.iter_mut().enumerate() {
            if i == ts || (sets.contains(&i) && Some(c.size) == size) {
                set(c).extend(tiles.iter().copied());
            }
        }
    }

    /// Drain tiles needing upload for texture set `ts`.
    pub fn take_uploads(&mut self, ts: usize) -> Vec<(u32, u32)> {
        self.caches.get_mut(ts).map(|c| std::mem::take(&mut c.upload).into_iter().collect()).unwrap_or_default()
    }

    /// `f` on the compositor inputs of texture set `ts` (without a live stroke; None for a set
    /// not ready).
    pub(crate) fn with_inputs<R>(&self, ts: usize, f: impl FnOnce(&texturecraft_compose::Inputs) -> Option<R>) -> Option<R> {
        let cache = self.caches.get(ts)?;
        let empty = BakedMaps::default();
        f(&texturecraft_compose::Inputs {
            project: &self.doc.project,
            ts,
            mesh: &self.mesh,
            gbuf: &cache.gbuf,
            pixels: &self.doc.pixels,
            baked: self.doc.baked.get(&ts).unwrap_or(&empty),
            block_scales: cache.composite.block_scales.clone(),
            preview: None,
            images: &self.images,
            bounds: self.mesh.bounds(),
            patterns: Some(&cache.patterns),
        })
    }

    /// Grayscale RGBA8 of layer `id`'s mask at `size²` for the mask view (None without a mask).
    pub fn mask_view_rgba(&self, id: LayerId, size: u32) -> Option<Vec<u8>> {
        let (ts, _, _) = self.doc.project.locate(id)?;
        let gray = self.with_inputs(ts, |inp| texturecraft_compose::mask_image(inp, id, size))?;
        Some(gray.iter().flat_map(|g| [*g, *g, *g, 255]).collect())
    }

    /// Mask values at texture-set coordinates in request order; None without an enabled mask
    /// Uncovered and out-of-range texels read zero; neighbourhood filters also read surrounding texels
    pub fn mask_at(&self, id: LayerId, texels: &[(u32, u32)]) -> Option<Vec<f32>> {
        let (texture_set, _, _) = self.doc.project.locate(id)?;
        self.with_inputs(texture_set, |inputs| texturecraft_compose::mask_at(inputs, id, texels))
    }

    /// Histogram (`bins` buckets over 0..1) of the values entering effect `effect`: its layer's
    /// mask, or for a content effect the layer's `channel` (None for an unknown effect).
    pub fn effect_histogram(&self, effect: LayerId, channel: Channel, bins: usize) -> Option<Vec<u32>> {
        let (owner, _) = self.doc.project.effect(effect)?;
        let (ts, _, _) = self.doc.project.locate(owner)?;
        self.with_inputs(ts, |inp| texturecraft_compose::effect_histogram(inp, effect, channel, bins))
    }

    pub fn baked(&self, ts: usize) -> Option<&BakedMaps> {
        self.doc.baked.get(&ts)
    }

    /// Bytes held by painted layer pixels, the composites of every texture set and their
    /// hot-layer caches.
    pub fn memory(&self) -> Memory {
        let composites = self.caches.iter().flat_map(|c| c.composite.channels.values()).map(texturecraft_texel::TileImage::bytes).sum();
        Memory { pixels: self.doc.pixels.bytes(), composites, cache: self.caches.iter().map(|c| c.below.bytes() + c.patterns.bytes()).sum() }
    }
}

/// Memory use reported by [`Session::memory`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Memory {
    pub pixels: usize,
    pub composites: usize,
    pub cache: usize,
}

impl Memory {
    pub fn total(self) -> usize {
        self.pixels + self.composites + self.cache
    }
}

/// Most tiles per composite chunk (a 1024² region).
const CHUNK: usize = 256;
/// The most a budgeted update spends on settling tiles (what they show is already right).
pub const SETTLE_BUDGET_MS: f64 = 6.0;

/// Take up to `n` dirty tiles, block by block (16 × 16 tiles) so neighbourhood filters
/// evaluate compact regions.
fn next_chunk(dirty: &mut BTreeSet<(u32, u32)>, max_tiles: usize) -> Vec<(u32, u32)> {
    // A zero-sized request cannot select a pivot
    if max_tiles == 0 {
        return Vec::new();
    }

    if dirty.len() <= max_tiles {
        return std::mem::take(dirty).into_iter().collect();
    }

    let mut tiles: Vec<(u32, u32)> = dirty.iter().copied().collect();

    // Partition by block order so only the requested chunk needs sorting
    tiles.select_nth_unstable_by_key(max_tiles - 1, |&(x, y)| (y / texturecraft_compose::BLOCK, x / texturecraft_compose::BLOCK, y, x));
    tiles.truncate(max_tiles);
    tiles.sort_by_key(|&(x, y)| (y / texturecraft_compose::BLOCK, x / texturecraft_compose::BLOCK, y, x));
    for tile in &tiles {
        dirty.remove(tile);
    }

    tiles
}

/// Dirty tiles a neighbourhood filter runs over ahead of the chunk being composited.
const FILTER_AHEAD: usize = 64;

/// The dirty tiles composited after `chunk` in its 16 × 16 tile blocks, in the order [`next_chunk`] takes them, at most [`FILTER_AHEAD`].
fn tiles_ahead(dirty: &BTreeSet<(u32, u32)>, chunk: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let block = |&(x, y): &(u32, u32)| (y / texturecraft_compose::BLOCK, x / texturecraft_compose::BLOCK);
    let blocks: BTreeSet<(u32, u32)> = chunk.iter().map(block).collect();
    let mut ahead: Vec<(u32, u32)> = dirty.iter().copied().filter(|tile| blocks.contains(&block(tile))).collect();
    ahead.sort_by_key(|tile| (block(tile), tile.1, tile.0));
    ahead.truncate(FILTER_AHEAD);
    ahead
}

/// Valid texture set resolutions: powers of two from 128 to 8192.
pub fn check_resolution(r: u32) -> Result<u32> {
    if (128..=8192).contains(&r) && r.is_power_of_two() {
        Ok(r)
    } else {
        Err(EngineError::Other(format!("resolution {r} must be a power of two between 128 and 8192")))
    }
}

/// A texture set's texels over all its UV tiles must stay within what bakes and caches hold.
pub fn check_size(ts: &TextureSet) -> Result<()> {
    let (w, h) = ts.size();
    if u64::from(w) * u64::from(h) > texturecraft_bake::MAX_TEXELS {
        return Err(EngineError::Other(format!(
            "{} UV tiles at {} exceed {} texels: lower the resolution",
            ts.uv_tiles.len(),
            ts.resolution,
            texturecraft_bake::MAX_TEXELS
        )));
    }
    Ok(())
}

/// UDIM numbers of the UV tiles material `m` of `mesh` paints, ascending: `mode` as in
/// [`Session::new_project_tiled`]. Tiles outside 1001..=2000 are an error in UV tile mode and
/// leave an automatic set classic (with a note in `log`).
pub(crate) fn detect_uv_tiles(mesh: &Mesh, m: usize, mode: Option<bool>, log: &mut Vec<String>) -> std::result::Result<Vec<u32>, String> {
    if mode == Some(false) {
        return Ok(Vec::new());
    }
    let cells = mesh.uv_tiles(m);
    // Mesh v points down: UDIM row r is v in -r..1-r.
    let udims: Option<Vec<u32>> =
        cells.iter().map(|&(u, v)| ((0..=9).contains(&u) && (-99..=0).contains(&v)).then(|| 1001 + u.unsigned_abs() + 10 * v.unsigned_abs())).collect();
    match (mode, udims) {
        (Some(true), Some(t)) if t.is_empty() => Ok(vec![1001]),
        (None, Some(t)) if t == [1001] || t.is_empty() => Ok(Vec::new()),
        (_, Some(mut t)) => {
            t.sort_unstable();
            Ok(t)
        }
        (Some(true), None) => Err("UVs lie outside UDIM tiles 1001..2000".into()),
        (None, None) => {
            log.push(format!("material {m}: UVs outside UDIM tiles 1001..2000; painting tile 1001 only"));
            Ok(Vec::new())
        }
        (Some(false), _) => Ok(Vec::new()),
    }
}

/// One pattern-cache budget for the whole session: every texture set may hold its share, so a
/// project with many sets cannot hold [`PATTERN_BUDGET`] per set.
pub(crate) fn share_pattern_budget(caches: &[SetCache]) {
    let share = PATTERN_BUDGET / caches.len().max(1);
    for c in caches {
        c.patterns.set_budget(share);
    }
}

/// Where texture set `t`'s UV tiles sit in its atlas (the mesh maps its UVs by this).
pub(crate) fn uv_layout(t: &TextureSet) -> (usize, Vec<(u32, f32)>) {
    (t.material, t.uv_tiles.iter().map(|u| (*u, t.tile_scale(*u))).collect())
}

fn next_mesh_id() -> u64 {
    static NEXT_MESH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
    NEXT_MESH.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
}

/// `source` with the UVs of each UV-tiled set's material mapped into its atlas (shared when no
/// set is UV-tiled).
fn tiled_mesh(source: &Arc<Mesh>, project: &Project) -> Arc<Mesh> {
    let tiled: Vec<_> = project.texture_sets.iter().filter_map(|ts| Some((ts, ts.tile_grid()?))).collect();
    if tiled.is_empty() {
        return source.clone();
    }
    let mut mesh = (**source).clone();
    let mut done = BTreeSet::new();
    for (ts, grid) in tiled {
        if done.insert(ts.material) {
            mesh.map_material_uvs_with(ts.material, |c, uv| Vec2::from(ts.to_atlas(&grid, c.to_array(), uv.to_array())));
        }
    }
    Arc::new(mesh)
}

#[cfg(test)]
mod pattern_budget_tests {
    use super::*;

    #[test]
    fn texture_sets_share_one_pattern_budget() {
        let mesh = texturecraft_geom::primitives::plane(2.0, 4);
        let caches: Vec<SetCache> = (0..4).map(|i| SetCache::new(&mesh, &TextureSet::new(format!("set{i}"), 0, 64))).collect();
        assert!(caches.iter().all(|c| c.patterns.budget() == PATTERN_BUDGET), "a lone cache starts with the whole budget");
        share_pattern_budget(&caches);
        let total: usize = caches.iter().map(|c| c.patterns.budget()).sum();
        assert!(total <= PATTERN_BUDGET && total > PATTERN_BUDGET - 4, "{total}");
        share_pattern_budget(caches.get(..1).unwrap_or_default());
        assert_eq!(caches[0].patterns.budget(), PATTERN_BUDGET);
        share_pattern_budget(&[]);
    }

    #[test]
    fn a_session_with_many_texture_sets_keeps_to_one_pattern_budget() {
        let mut s = Session::new_project("t", texturecraft_geom::primitives::plane(2.0, 4), "builtin:test", None, 128).unwrap();
        for i in 1..10 {
            s.doc.project.texture_sets.push(TextureSet::new(format!("set{i}"), 0, 64));
        }
        s.sync_caches();
        assert_eq!(s.caches.len(), 10);
        let total = |s: &Session| s.caches.iter().map(|c| c.patterns.budget()).sum::<usize>();
        assert!(total(&s) <= PATTERN_BUDGET && total(&s) > PATTERN_BUDGET - 10, "ten sets hold {} of {PATTERN_BUDGET}", total(&s));
        // Fewer sets, a larger share each; a set added later takes its share from the others.
        s.doc.project.texture_sets.truncate(2);
        s.sync_caches();
        assert!(s.caches.iter().all(|c| c.patterns.budget() == PATTERN_BUDGET / 2));
        s.doc.project.texture_sets.push(TextureSet::new("again", 0, 128));
        s.sync_caches();
        assert!(total(&s) <= PATTERN_BUDGET && s.caches.iter().all(|c| c.patterns.budget() == PATTERN_BUDGET / 3));
    }
}
