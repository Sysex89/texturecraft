//! Background compositing ends in exactly the composite of compositing on the session's thread,
//! whatever happens while a job runs: edits, composites on the session's thread, strokes,
//! layout changes.
#![cfg(not(target_arch = "wasm32"))]

use serde_json::{Value, json};
use texturecraft_engine::geom::glam::Vec2;
use texturecraft_engine::geom::primitives;
use texturecraft_engine::{BackgroundCompositor, Focus, Session, execute};

/// Procedural fills, a masked layer with a generator and a blur, a folder, a painted layer.
fn build() -> Session {
    let mut s = Session::new_project("bg", primitives::sphere(1.0, 32, 16), "builtin:sphere", None, 512).unwrap();
    s.viewport = Vec2::new(600.0, 600.0);
    let run = |s: &mut Session, id: &str, p: Value| execute(s, id, &p).unwrap();
    run(&mut s, "layer.add.fill", json!({"color": [0.6, 0.5, 0.4], "roughness": 0.6}));
    run(&mut s, "layer.fill.set", json!({"channel": "baseColor", "procedural": "clouds", "low": [0.2, 0.2, 0.2], "high": [0.8, 0.7, 0.6]}));
    run(&mut s, "layer.fill.projection", json!({"projection": "triplanar", "scale": 2}));
    run(&mut s, "layer.add.fill", json!({"color": [0.1, 0.3, 0.6]}));
    run(&mut s, "layer.fill.set", json!({"channel": "roughness", "procedural": "grunge"}));
    run(&mut s, "mask.add.black", Value::Null);
    run(&mut s, "effect.add.generator", json!({"generator": "position", "level": 0.5}));
    run(&mut s, "effect.add.filter", json!({"filter": "blur", "radius": 8}));
    run(&mut s, "layer.add.group", Value::Null);
    run(&mut s, "layer.add.fill", json!({"color": [0.9, 0.9, 0.2]}));
    let top = s.doc.project.texture_sets[0].stack.last().copied().unwrap();
    run(&mut s, "layer.select", json!({"layer": top.0}));
    run(&mut s, "layer.add.paint", Value::Null);
    run(&mut s, "tool.stroke", json!({"points": [[200, 300], [400, 350]]}));
    s
}

/// Everything dirty, nothing cached below the selected layer: a full recomposite to do.
fn dirty_all(s: &mut Session) {
    for c in &mut s.caches {
        c.invalidate_all();
        c.below.clear();
    }
}

/// `s` composited on its own thread.
fn reference(mut s: Session) -> Session {
    dirty_all(&mut s);
    s.update();
    s
}

fn same_composites(a: &Session, b: &Session) -> bool {
    a.caches.len() == b.caches.len() && a.caches.iter().zip(&b.caches).all(|(x, y)| x.composite == y.composite)
}

#[test]
fn background_composites_match_compositing_on_the_session_thread() {
    let want = reference(build());
    let mut s = build();
    dirty_all(&mut s);
    let mut bg = BackgroundCompositor::new();
    bg.finish(&mut s);
    assert_eq!(s.pending_tiles(), 0);
    assert!(same_composites(&s, &want), "a background composite differs");
    // A second pass reads the pattern values the first one cached: still the same.
    dirty_all(&mut s);
    bg.finish(&mut s);
    assert!(same_composites(&s, &want), "a background composite from cached patterns differs");
}

#[test]
fn tiles_in_flight_count_as_pending() {
    let mut s = build();
    dirty_all(&mut s);
    let mut bg = BackgroundCompositor::new();
    bg.drive(&mut s, 30.0, || 0.0);
    assert!(bg.busy());
    assert!(s.caches[0].dirty.is_empty() && !s.caches[0].in_flight.is_empty());
    assert_eq!(s.pending_tiles(), s.caches.iter().map(|c| c.dirty.len() + c.settle.len() + c.in_flight.len()).sum::<usize>());
    assert!(s.pending_tiles() > 0);
    bg.cancel(&mut s);
    assert!(!bg.busy() && s.caches[0].in_flight.is_empty() && !s.caches[0].dirty.is_empty(), "cancelled tiles are dirty again");
}

#[test]
fn edits_and_session_composites_during_a_job_end_in_the_edited_composite() {
    let edit = |s: &mut Session, k: u32| {
        let v = 0.1 + 0.2 * k as f32;
        execute(s, "layer.fill.set", &json!({"layer": s.doc.project.texture_sets[0].stack[1].0, "channel": "baseColor", "color": [v, 0.4, 0.3]})).unwrap();
    };
    // Different moments in the job: at once, after some chunks, with a composite on the
    // session's thread in between.
    for (k, wait_ms, sync) in [(0, 0, false), (1, 5, false), (2, 0, true), (3, 15, true)] {
        let mut want = build();
        edit(&mut want, k);
        let want = reference(want);
        let mut s = build();
        dirty_all(&mut s);
        let mut bg = BackgroundCompositor::new();
        bg.drive(&mut s, 30.0, || 0.0);
        std::thread::sleep(std::time::Duration::from_millis(wait_ms));
        edit(&mut s, k);
        if sync {
            s.update();
        }
        bg.finish(&mut s);
        assert!(same_composites(&s, &want), "edit {k} after {wait_ms} ms (session composite: {sync}) left a stale composite");
    }
}

#[test]
fn a_stroke_cancels_the_job_and_composites_on_the_session_thread() {
    let mut s = build();
    dirty_all(&mut s);
    let mut bg = BackgroundCompositor::new();
    bg.drive(&mut s, 30.0, || 0.0);
    assert!(bg.busy());
    s.begin_stroke(Vec2::new(300.0, 300.0), s.viewport, 1.0).unwrap();
    bg.drive(&mut s, f64::INFINITY, || 0.0);
    assert!(!bg.busy() && s.caches.iter().all(|c| c.in_flight.is_empty()), "the stroke cancelled the job");
    s.continue_stroke(Vec2::new(360.0, 320.0), 1.0);
    bg.drive(&mut s, f64::INFINITY, || 0.0);
    s.end_stroke().unwrap();
    bg.finish(&mut s);
    // The same stroke, composited on the session's thread throughout.
    let mut want = build();
    want.begin_stroke(Vec2::new(300.0, 300.0), want.viewport, 1.0).unwrap();
    want.continue_stroke(Vec2::new(360.0, 320.0), 1.0);
    want.end_stroke().unwrap();
    assert!(same_composites(&s, &reference(want)), "a stroke during a job left a stale composite");
}

#[test]
fn a_layout_change_during_a_job_drops_its_results() {
    let mut want = build();
    execute(&mut want, "textureSet.resolution", &json!({"resolution": 256})).unwrap();
    let want = reference(want);
    let mut s = build();
    dirty_all(&mut s);
    let mut bg = BackgroundCompositor::new();
    bg.drive(&mut s, 30.0, || 0.0);
    execute(&mut s, "textureSet.resolution", &json!({"resolution": 256})).unwrap();
    bg.finish(&mut s);
    assert_eq!(s.caches[0].size, (256, 256));
    assert!(same_composites(&s, &want), "results for the old resolution were applied");
}

/// Tiles inside the UV rectangle `[u0, v0, u1, v1]` of set 0.
fn tiles_in(s: &Session, r: [f32; 4]) -> std::collections::BTreeSet<(u32, u32)> {
    let (w, h) = (s.caches[0].size.0 as f32, s.caches[0].size.1 as f32);
    s.caches[0]
        .covered_tiles
        .iter()
        .copied()
        .filter(|&(tx, ty)| {
            let (u0, v0, u1, v1) = (tx as f32 * 64.0 / w, ty as f32 * 64.0 / h, (tx + 1) as f32 * 64.0 / w, (ty + 1) as f32 * 64.0 / h);
            u1 > r[0] && u0 < r[2] && v1 > r[1] && v0 < r[3]
        })
        .collect()
}

#[test]
fn the_tiles_the_views_show_go_first() {
    let want = reference(build());
    let mut s = build();
    s.active_set = 0;
    dirty_all(&mut s);
    let quarter = [0.0, 0.0, 0.5, 0.5];
    let shown = tiles_in(&s, quarter);
    assert!(!shown.is_empty() && shown.len() < s.caches[0].covered_tiles.len());
    let mut bg = BackgroundCompositor::new();
    bg.set_focus(Focus { view3d: false, uv: Some(quarter) });
    bg.drive(&mut s, 30.0, || 0.0);
    // The first job took exactly the tiles the 2D view shows; the rest wait.
    let dirty = &s.caches[0].dirty;
    assert!(shown.iter().all(|t| !dirty.contains(t)), "a shown tile was left for later");
    assert!(s.caches[0].covered_tiles.iter().any(|t| !shown.contains(t) && dirty.contains(t)), "the first job took tiles not shown");
    bg.finish(&mut s);
    assert!(same_composites(&s, &want), "a composite in visibility order differs");
}

#[test]
fn the_3d_view_shows_the_tiles_facing_the_camera() {
    let mut s = build();
    let view = Focus { view3d: true, uv: None };
    let front = s.visible_tiles(0, &view);
    let covered = s.caches[0].covered_tiles.len();
    assert!(!front.is_empty() && front.len() < covered, "{} of {covered}", front.len());
    // A 2D view showing the whole set beside the 3D view puts nothing more first.
    assert_eq!(s.visible_tiles(0, &Focus { view3d: true, uv: Some([0.0, 0.0, 1.0, 1.0]) }), front);
    // Half way round the sphere, mostly other tiles face the camera.
    s.camera.yaw += std::f32::consts::PI;
    let back = s.visible_tiles(0, &view);
    let both = front.intersection(&back).count();
    assert!(!back.is_empty() && both * 2 < front.len().min(back.len()), "front {}, back {}, both {both}", front.len(), back.len());
}
