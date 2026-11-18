//! The Tile Generator pattern and pattern resolutions through engine commands

use serde_json::{Value, json};
use texturecraft_engine::core::{
    Channel, EffectKind, LayerId, LayerKind, Procedural, ProceduralKind, Projection, ResolutionMode, Source, TileGenerator, TileShape,
};
use texturecraft_engine::geom::primitives;
use texturecraft_engine::{Session, execute};

fn session() -> Session {
    Session::new_project("t", primitives::sphere(1.0, 16, 8), "builtin:sphere", None, 128).unwrap()
}

/// A fill layer whose base color reads the Tile Generator, and its id
fn tile_fill(session: &mut Session) -> u64 {
    let layer = execute(session, "layer.add.fill", &Value::Null).unwrap()["layer"].as_u64().unwrap();
    execute(session, "layer.fill.set", &json!({"channel": "baseColor", "procedural": "Tile Generator"})).unwrap();
    layer
}

fn base_color_pattern(session: &Session, layer: u64) -> Procedural {
    let LayerKind::Fill(fill) = &session.doc.project.layer(LayerId(layer)).unwrap().kind else { panic!("not a fill layer") };
    let Source::Procedural { pattern, .. } = fill.source(Channel::BaseColor) else { panic!("the base color is not a pattern") };
    pattern
}

fn tile(pattern: &Procedural) -> TileGenerator {
    pattern.tile.as_deref().cloned().unwrap()
}

/// A mask fill effect reading a uniform value, on a fresh fill layer's black mask
fn mask_fill(session: &mut Session) -> u64 {
    execute(session, "layer.add.fill", &Value::Null).unwrap();
    execute(session, "mask.add.black", &Value::Null).unwrap();
    execute(session, "effect.add.fill", &json!({"value": 1.0})).unwrap()["effect"].as_u64().unwrap()
}

fn mask_pattern(session: &Session, effect: u64) -> Procedural {
    let Some((_, effect)) = session.doc.project.effect(LayerId(effect)) else { panic!("no effect {effect}") };
    let EffectKind::Fill { source: Source::Procedural { pattern, .. }, .. } = &effect.kind else { panic!("the mask fill reads no pattern") };
    pattern.clone()
}

#[test]
fn a_new_tile_generator_starts_from_the_reference_defaults() {
    let mut session = session();
    let layer = tile_fill(&mut session);

    let pattern = base_color_pattern(&session, layer);
    assert_eq!(pattern.kind, ProceduralKind::TileGenerator);
    assert_eq!(tile(&pattern), TileGenerator::default());
    assert_eq!(pattern.scale, 1.0, "the tile counts set its frequency: the grid covers one repeat of the projection");
}

#[test]
fn tile_settings_alone_edit_the_channels_pattern() {
    let mut session = session();
    let layer = tile_fill(&mut session);

    execute(&mut session, "layer.fill.set", &json!({"channel": "baseColor", "tile": {"amount": [4, 2], "shape": "Half bell", "offset": 0.5}})).unwrap();

    let tile = tile(&base_color_pattern(&session, layer));
    assert_eq!(tile.amount, [4, 2]);
    assert_eq!(tile.shape, TileShape::HalfBell, "a menu choice is taken by its label");
    assert_eq!(tile.offset, 0.5);
    assert_eq!(tile.specific, TileGenerator::default().specific, "settings not named keep their value");
}

#[test]
fn tile_settings_with_a_seed_edit_both() {
    let mut session = session();
    let layer = tile_fill(&mut session);

    execute(&mut session, "layer.fill.set", &json!({"channel": "baseColor", "tile": {"amount": [3, 3]}, "seed": 9, "invert": true})).unwrap();

    let pattern = base_color_pattern(&session, layer);
    assert_eq!(tile(&pattern).amount, [3, 3]);
    assert_eq!((pattern.seed, pattern.invert), (9, true), "the seed and invert beside a tile setting are applied, not dropped");
}

#[test]
fn picking_the_tile_generator_again_keeps_its_settings() {
    let mut session = session();
    let layer = tile_fill(&mut session);
    execute(&mut session, "layer.fill.set", &json!({"channel": "baseColor", "tile": {"amount": [3, 3]}, "resolution": "custom", "customResolution": 128}))
        .unwrap();

    // The panel's scale and seed edits send the pattern again with its kind
    execute(&mut session, "layer.fill.set", &json!({"channel": "baseColor", "procedural": "Tile Generator", "seed": 9})).unwrap();

    let pattern = base_color_pattern(&session, layer);
    assert_eq!(pattern.seed, 9);
    assert_eq!(tile(&pattern).amount, [3, 3]);
    assert_eq!((pattern.resolution.mode, pattern.resolution.custom), (ResolutionMode::Custom, 128));
}

#[test]
fn hostile_tile_settings_are_brought_into_range() {
    let mut session = session();
    let layer = tile_fill(&mut session);

    execute(
        &mut session,
        "layer.fill.set",
        &json!({"channel": "baseColor", "tile": {"amount": [0, 999], "specific": -5.0, "shapeTiling": 400, "rotation": 450.0, "shapeResolution": [3, 100000]}}),
    )
    .unwrap();

    let tile = tile(&base_color_pattern(&session, layer));
    assert_eq!(tile.amount, [1, 64]);
    assert_eq!(tile.specific, 0.0);
    assert_eq!(tile.shape_tiling, 16);
    assert_eq!(tile.rotation, 90.0, "angles wrap around the turn");
    assert_eq!(tile.shape_resolution, [256, 4096]);
}

#[test]
fn unknown_or_misplaced_tile_settings_are_errors() {
    let mut session = session();
    tile_fill(&mut session);

    let unknown = execute(&mut session, "layer.fill.set", &json!({"channel": "baseColor", "tile": {"colour": 0.5}}));
    assert!(unknown.unwrap_err().to_string().contains("colour"), "the unknown setting is named");
    let bad_shape = execute(&mut session, "layer.fill.set", &json!({"channel": "baseColor", "tile": {"shape": "Hexagon"}}));
    assert!(bad_shape.is_err());

    execute(&mut session, "layer.fill.set", &json!({"channel": "roughness", "procedural": "Clouds"})).unwrap();
    let on_clouds = execute(&mut session, "layer.fill.set", &json!({"channel": "roughness", "tile": {"offset": 0.5}}));
    assert!(on_clouds.is_err(), "tile settings go with the Tile Generator only");
    let no_pattern = execute(&mut session, "layer.fill.set", &json!({"channel": "metallic", "tile": {"offset": 0.5}}));
    assert!(no_pattern.is_err(), "a uniform channel has no pattern to edit");
}

#[test]
fn a_mask_fill_picks_and_edits_a_pattern_with_effect_set() {
    let mut session = session();
    let effect = mask_fill(&mut session);

    execute(&mut session, "effect.set", &json!({"effect": effect, "procedural": "Tile Generator"})).unwrap();
    execute(&mut session, "effect.set", &json!({"effect": effect, "tile": {"checkerMask": true}, "seed": 4})).unwrap();

    let pattern = mask_pattern(&session, effect);
    assert!(tile(&pattern).checker_mask);
    assert_eq!(pattern.seed, 4);

    execute(&mut session, "effect.set", &json!({"effect": effect, "tileDefaults": true})).unwrap();
    assert_eq!(tile(&mask_pattern(&session, effect)), TileGenerator::default(), "Restore defaults puts every tile setting back");

    // Only an explicit uniform pick replaces an active pattern
    execute(&mut session, "effect.set", &json!({"effect": effect, "value": 0.3})).unwrap();
    assert_eq!(mask_pattern(&session, effect).kind, ProceduralKind::TileGenerator);

    execute(&mut session, "effect.set", &json!({"effect": effect, "uniform": 0.3})).unwrap();
    let Some((_, edited)) = session.doc.project.effect(LayerId(effect)) else { panic!("the mask fill is gone") };
    assert!(matches!(&edited.kind, EffectKind::Fill { source: Source::Uniform(value), .. } if *value == [0.3; 3]), "a uniform pick reads the value again");
}

#[test]
fn a_uniform_pick_outranks_pattern_edits_on_a_mask_fill() {
    let mut session = session();
    let effect = mask_fill(&mut session);
    execute(&mut session, "effect.set", &json!({"effect": effect, "procedural": "Tile Generator"})).unwrap();

    execute(&mut session, "effect.set", &json!({"effect": effect, "uniform": 0.3, "seed": 4})).unwrap();

    let Some((_, edited)) = session.doc.project.effect(LayerId(effect)) else { panic!("the mask fill is gone") };
    assert!(matches!(&edited.kind, EffectKind::Fill { source: Source::Uniform(value), .. } if *value == [0.3; 3]), "the fill reads the value, not the pattern");
}

/// A fill effect's `scale` is also its projection's: while it reads a pattern, the pattern takes it
#[test]
fn a_pattern_scale_edit_leaves_the_mask_fills_projection_alone() {
    let mut session = session();
    let effect = mask_fill(&mut session);
    execute(&mut session, "effect.set", &json!({"effect": effect, "procedural": "Tile Generator"})).unwrap();

    execute(&mut session, "effect.set", &json!({"effect": effect, "scale": 3.0})).unwrap();

    let Some((_, edited)) = session.doc.project.effect(LayerId(effect)) else { panic!("the mask fill is gone") };
    let EffectKind::Fill { source: Source::Procedural { pattern, .. }, projection } = &edited.kind else { panic!("the fill reads a pattern") };
    assert_eq!(pattern.scale, 3.0);
    assert_eq!(*projection, Projection::default(), "the projection keeps its scale");
}

#[test]
fn a_mask_fill_added_with_tile_settings_keeps_them() {
    let mut session = session();
    execute(&mut session, "layer.add.fill", &Value::Null).unwrap();
    execute(&mut session, "mask.add.black", &Value::Null).unwrap();

    let added = execute(&mut session, "effect.add.fill", &json!({"procedural": "Tile Generator", "tile": {"amount": [6, 6]}, "resolutionFactor": -2})).unwrap();

    let pattern = mask_pattern(&session, added["effect"].as_u64().unwrap());
    assert_eq!(tile(&pattern).amount, [6, 6]);
    assert_eq!(pattern.resolution.factor, -2);
}

#[test]
fn inspect_reports_the_tile_settings_and_a_patterns_resolution() {
    let mut session = session();
    tile_fill(&mut session);
    execute(&mut session, "layer.fill.set", &json!({"channel": "baseColor", "tile": {"amount": [5, 7]}, "resolution": "textureSet"})).unwrap();

    let inspected = execute(&mut session, "project.inspect", &Value::Null).unwrap().to_string();
    assert!(inspected.contains("\"amount\":[5,7]"), "the settings are reported: {inspected}");
    assert!(inspected.contains("\"randomMask\""), "under the camelCase names the commands take");
    assert!(inspected.contains("\"mode\":\"TextureSet\""), "the pattern's resolution is reported");
}

/// Imports a small gray PNG as resource `name` and returns its resource name
fn import_gray(session: &mut Session, name: &str) -> String {
    import_shade(session, name, 128)
}

/// Imports a small PNG of gray `shade` as resource `name` and returns its resource name
fn import_shade(session: &mut Session, name: &str, shade: u8) -> String {
    let image = image::RgbaImage::from_pixel(4, 4, image::Rgba([shade, shade, shade, 255]));
    let mut bytes = Vec::new();
    image.write_to(&mut std::io::Cursor::new(&mut bytes), image::ImageFormat::Png).unwrap();
    session.import_resource_bytes(name, bytes).unwrap()
}

#[test]
fn removing_unused_resources_keeps_the_images_a_tile_generator_reads() {
    let mut session = session();
    let input = import_gray(&mut session, "vent.png");
    let background = import_gray(&mut session, "plate.png");
    let unused = import_gray(&mut session, "unused.png");
    tile_fill(&mut session);
    execute(&mut session, "layer.fill.set", &json!({"channel": "baseColor", "tile": {"shape": "Image", "inputs": [input], "backgroundImage": background}}))
        .unwrap();

    execute(&mut session, "file.removeUnused", &Value::Null).unwrap();
    assert!(session.doc.resources.contains_key(&input), "the pattern input is still read");
    assert!(session.doc.resources.contains_key(&background), "the background image is still read");
    assert!(!session.doc.resources.contains_key(&unused));
}

/// The composited base color of every texel of the texture set
fn base_colors(session: &mut Session) -> Vec<f32> {
    session.update();
    let composite = &session.caches[0].composite;
    let size = composite.resolution;
    (0..size * size).map(|texel| composite.value(Channel::BaseColor, texel % size, texel / size)[0]).collect()
}

#[test]
fn the_compositor_draws_a_tile_generators_image_input() {
    let mut session = session();
    let input = import_gray(&mut session, "plate.png");
    tile_fill(&mut session);

    // One tile with no gap: the image covers the whole texture
    execute(&mut session, "layer.fill.set", &json!({"channel": "baseColor", "tile": {"shape": "Image", "inputs": [input], "amount": [1, 1]}})).unwrap();

    let colors = base_colors(&mut session);
    let mid_gray = colors.iter().filter(|value| (**value - 0.5).abs() < 0.05).count();
    assert!(mid_gray > colors.len() / 2, "the image's gray reaches the texture, not an empty shape: {mid_gray} of {}", colors.len());
}

#[test]
fn importing_the_image_a_tile_generator_names_redraws_it() {
    let mut session = session();
    tile_fill(&mut session);
    execute(&mut session, "layer.fill.set", &json!({"channel": "baseColor", "tile": {"shape": "Image", "inputs": ["plate.png"], "amount": [1, 1]}})).unwrap();
    let before = base_colors(&mut session);

    assert_eq!(import_gray(&mut session, "plate.png"), "plate.png");
    let after = base_colors(&mut session);

    assert_ne!(before, after, "the image that arrived later reaches the texture");
}

#[test]
fn an_image_replaced_under_the_same_name_redraws_the_tile_generator() {
    // A session that only ever saw the second image
    let mut fresh = session();
    tile_fill(&mut fresh);
    execute(&mut fresh, "layer.fill.set", &json!({"channel": "baseColor", "tile": {"shape": "Image", "inputs": ["plate.png"], "amount": [1, 1]}})).unwrap();
    assert_eq!(import_shade(&mut fresh, "plate.png", 230), "plate.png");
    let want = base_colors(&mut fresh);

    // The first image is composited (and its pattern values cached), then removed, and another
    // file is imported under the same name
    let mut session = session();
    import_shade(&mut session, "plate.png", 40);
    tile_fill(&mut session);
    execute(&mut session, "layer.fill.set", &json!({"channel": "baseColor", "tile": {"shape": "Image", "inputs": ["plate.png"], "amount": [1, 1]}})).unwrap();
    let first = base_colors(&mut session);
    assert!(session.memory().cache > 0, "the pattern is cached");
    execute(&mut session, "resource.remove", &json!({"name": "plate.png"})).unwrap();
    assert_eq!(import_shade(&mut session, "plate.png", 230), "plate.png");
    let after = base_colors(&mut session);

    assert_ne!(first, want, "the two images draw differently");
    assert_eq!(after, want, "the replaced image reaches the texture, not the first image's cached pattern");
}

#[test]
fn a_coarser_pattern_resolution_changes_the_composite() {
    let mut session = session();
    tile_fill(&mut session);
    let exact = base_colors(&mut session);

    execute(&mut session, "layer.fill.set", &json!({"channel": "baseColor", "resolutionFactor": -3})).unwrap();
    let coarse = base_colors(&mut session);
    let changed = exact.iter().zip(&coarse).filter(|(exact, coarse)| (**exact - **coarse).abs() > 0.05).count();
    assert!(changed > 0, "an eighth of the resolution blurs the bricks' bevels");
}

#[test]
fn a_tile_generator_gradient_map_reads_the_image_input() {
    let mut session = session();
    let input = import_gray(&mut session, "plate.png");
    tile_fill(&mut session);
    let colors = [[0.0; 3], [1.0, 0.0, 0.0]];
    execute(
        &mut session,
        "layer.fill.set",
        &json!({"channel": "baseColor", "tile": {"shape": "Image", "inputs": [input], "amount": [1, 1]}, "colors": colors}),
    )
    .unwrap();

    let red = base_colors(&mut session);

    let mid_red = red.iter().filter(|value| (**value - 0.5).abs() < 0.05).count();
    assert!(mid_red > red.len() / 2, "the gradient map recolors the input image's gray: {mid_red} of {}", red.len());
    assert_eq!(session.caches[0].composite.value(Channel::BaseColor, 64, 64)[1], 0.0);
}

#[test]
fn a_tile_generator_gradient_map_keeps_the_selected_resolution() {
    let mut session = session();
    let layer = tile_fill(&mut session);
    let colors = [[0.0; 3], [1.0, 0.0, 0.0]];
    execute(&mut session, "layer.fill.set", &json!({"channel": "baseColor", "colors": colors})).unwrap();
    let exact = base_colors(&mut session);

    execute(&mut session, "layer.fill.set", &json!({"channel": "baseColor", "resolution": "uvTile", "resolutionFactor": -3})).unwrap();
    let coarse = base_colors(&mut session);
    let inspected = execute(&mut session, "project.inspect", &Value::Null).unwrap();
    let row = inspected["layers"].as_array().unwrap().iter().find(|row| row["id"] == layer).unwrap();
    let source = &row["fill"]["sources"]["baseColor"];

    assert_ne!(exact, coarse, "the gradient map samples the coarser pattern");
    assert_eq!(source["colors"], json!(colors));
    assert_eq!(source["resolution"]["mode"], "UvTile");
    assert_eq!(source["resolution"]["factor"], -3);
    assert_eq!(source["tile"]["amount"], json!(TileGenerator::default().amount));
}
