//! Headless performance tests for baking and compositing, through the same [`Headless`] backend
//! the CLI and the MCP server drive: the GPU ray tracer is installed as the app installs it (CPU
//! when there is no adapter), and every stage the app runs is timed separately.
//!
//! ```sh
//! cargo test -p texturecraft-automation --release --test perf -- --ignored --nocapture --test-threads=1
//! cargo test -p texturecraft-automation --release --test perf -- --ignored --nocapture bake_   # bakes only
//! ```
//!
//! Each stage has a budget; a release run fails when a stage blows its budget. The budgets are
//! regression guards, about 1.5× what each stage took on a 20-thread laptop with an Iris Xe GPU
//! (2026-11); `TC_PERF_SCALE=0.5` turns them into targets. Environment:
//! - `TC_PERF_SCALE=<f>` multiplies every budget (a slower machine: `2`; tightening: `0.5`).
//! - `TC_PERF_BUDGETS=off` reports without failing.
//! - `TC_PERF_OUT=<dir>` writes one `<test>.json` per test (stage, ms, budget ms) for comparing runs.
//!
//! Debug builds report only: their timings say nothing about release speed. Run one test at a
//! time (`--test-threads=1`): the stages use every core.

use std::time::{Duration, Instant};

use rayon::prelude::*;
use serde_json::{Value, json};
use texturecraft_automation::headless::{Filesystem, Headless};
use texturecraft_engine::Session;
use texturecraft_engine::bake::Progress;
use texturecraft_engine::compose::{DisplayKind, tile_bytes};
use texturecraft_engine::core::{BakeResolution, BakedMap};
use texturecraft_engine::geom::glam::Vec2;
use texturecraft_engine::project_cmds::sample_session;

#[path = "perf/report.rs"]
mod report;

use report::{Report, ms};

/// The startup sample project (material ball, two texture sets with smart materials whose
/// generators read the baked maps) at `resolution`, in a headless backend with the GPU tracer installed.
/// The sample's own bake is a cheap 256² one: tests rebake at the size they measure.
fn sample(resolution: u32) -> Headless {
    // Install the GPU tracer before the sample session performs its bake
    let mut backend = Headless::new(Headless::default_session().unwrap(), Filesystem::Denied);
    let mut session = sample_session(resolution, 256).unwrap();
    session.viewport = Vec2::new(1000.0, 880.0);
    backend.session = session;
    backend.session.update();

    backend
}

/// Every tile the viewport would upload, converted to every display format as the UI does
/// (in parallel); returns the bytes.
fn display_all(session: &mut Session) -> usize {
    let mut bytes = 0;

    for texture_set in 0..session.set_count() {
        let tiles = session.take_uploads(texture_set);
        let baked = session.baked(texture_set).cloned().unwrap_or_default();
        let composite = &session.caches[texture_set].composite;
        bytes += tiles
            .par_iter()
            .flat_map_iter(|&(tile_x, tile_y)| DisplayKind::ALL.map(|kind| tile_bytes(composite, &baked, kind, tile_x, tile_y).len()))
            .sum::<usize>();
    }

    bytes
}

fn tiles_of(session: &Session) -> usize {
    session.caches.iter().map(|cache| cache.dirty.len()).sum()
}

/// What a recomposite finds cached.
#[derive(Clone, Copy, PartialEq)]
enum Caches {
    /// Nothing: the first composite after opening a project.
    None,
    /// Pattern values, not the stack below the selected layer: what a bake or a stack edit
    /// leaves (what users wait on).
    AfterEdit,
    /// Both, as far as their budgets hold them.
    Warm,
}

impl Caches {
    fn label(self) -> &'static str {
        match self {
            Caches::None => "nothing cached",
            Caches::AfterEdit => "after an edit",
            Caches::Warm => "warm",
        }
    }
}

/// Every tile of every set dirty, finding `caches`.
fn invalidate(session: &mut Session, caches: Caches) {
    for texture_set in 0..session.set_count() {
        invalidate_set(session, texture_set, caches);
    }
}

fn invalidate_set(session: &mut Session, texture_set: usize, caches: Caches) {
    let cache = &mut session.caches[texture_set];
    cache.invalidate_all();
    if caches != Caches::Warm {
        cache.below.clear();
    }
    if caches == Caches::None {
        cache.patterns.clear();
    }
}

/// Bake every texture set at `bake_resolution` the way the app does: the ray-casting task (the worker
/// thread's part), then `apply_bake` and the recomposite of the layers that read the maps (the
/// frame's part). `cpu` skips the GPU tracer. Budgets are per whole-project stage.
fn bake_project(report: &mut Report, backend: &mut Headless, label: &str, bake_resolution: u32, cpu: bool, budgets: [Option<f64>; 3]) {
    let session = &mut backend.session;
    let resolution = BakeResolution::from_pixels(bake_resolution).unwrap();

    for texture_set in 0..session.set_count() {
        session.doc.project.texture_set_mut(texture_set).unwrap().bake.common.resolution = resolution;
    }

    let (mut rays, mut apply, mut tracer) = (Duration::ZERO, Duration::ZERO, String::new());
    let mut outputs = Vec::new();

    for texture_set in 0..session.set_count() {
        let mut task = session.bake_task(texture_set).unwrap();
        if cpu {
            task.tracer = None;
        }

        let start = Instant::now();
        let output = task.run(&Progress::new()).unwrap();
        rays += start.elapsed();
        tracer = output.tracer.clone();
        outputs.push((texture_set, output));
    }

    for (texture_set, output) in outputs {
        let start = Instant::now();
        session.apply_bake(texture_set, output).unwrap();
        apply += start.elapsed();
    }

    let maps = session.doc.project.texture_sets[0].bake.maps.enabled().count();
    report.add(format!("{label}: bake {} sets × {maps} maps at {bake_resolution}²", session.set_count()), rays, budgets[0], format!("[{tracer}]"));
    report.add(format!("{label}: apply_bake (undo snapshot + store)"), apply, budgets[1], "");

    let dirty = tiles_of(session);
    report.time(format!("{label}: recomposite after bake ({dirty} tiles)"), budgets[2], || session.update());
}

fn sample_for_project_bake(resolution: u32) -> Headless {
    const PROJECT_BAKE_RAYS: u32 = 32;

    let mut backend = sample(resolution);
    for texture_set in 0..backend.session.set_count() {
        let maps = &mut backend.session.doc.project.texture_set_mut(texture_set).unwrap().bake.maps;
        for ray_settings in [&mut maps.ao.parameters.rays, &mut maps.thickness.parameters.rays, &mut maps.bent_normal.parameters.rays] {
            ray_settings.rays = PROJECT_BAKE_RAYS;
        }
    }

    backend
}

#[test]
fn project_bake_uses_documented_ray_count() {
    let backend = sample_for_project_bake(256);

    for texture_set in 0..backend.session.set_count() {
        let task = backend.session.bake_task(texture_set).unwrap();
        assert_eq!(task.settings.maps.ao.parameters.rays.rays, 32);
        assert_eq!(task.settings.maps.thickness.parameters.rays.rays, 32);
        assert_eq!(task.settings.maps.bent_normal.parameters.rays.rays, 32);
    }
}

/// The whole-project bake at benchmark settings (1024², 32 rays, AA 1), then 2048².
#[test]
#[ignore]
fn bake_sample_project() {
    let mut report = Report::new("bake_sample_project");
    let mut backend = sample_for_project_bake(2048);

    bake_project(&mut report, &mut backend, "1K", 1024, false, [Some(5_000.0), Some(200.0), Some(6_500.0)]);
    bake_project(&mut report, &mut backend, "2K", 2048, false, [Some(23_000.0), Some(400.0), Some(6_500.0)]);

    // Measure the agent command with bake and recomposite in one call
    report.time("bake.run (command, all sets, 1K)", Some(12_000.0), || backend.command_run("bake.run", json!({"ts": "all", "resolution": 1024})).unwrap());

    report.finish();
}

/// Each mesh map alone at 2048² on texture set 0: which baker is the slow one.
#[test]
#[ignore]
fn bake_each_map() {
    let mut report = Report::new("bake_each_map");
    let backend = sample(1024);
    let session = &backend.session;
    let maps: Vec<_> = session.doc.project.texture_sets[0].bake.maps.enabled().collect();

    for map in maps {
        let mut task = session.bake_task(0).unwrap();
        task.settings.maps.select([map]);
        task.settings.common.resolution = BakeResolution::R2048;

        let start = Instant::now();
        let output = task.run(&Progress::new()).unwrap();

        // Hemisphere maps cast many rays per texel
        let budget = if matches!(map, BakedMap::AmbientOcclusion | BakedMap::Thickness | BakedMap::BentNormal) { 6_000.0 } else { 300.0 };
        report.add(format!("set 0, {} at 2048²", map.id()), start.elapsed(), Some(budget), format!("[{}]", output.tracer));
    }

    report.finish();
}

/// GPU against CPU rays on the same 1K bake (no budget on the CPU row: it is the reference).
#[test]
#[ignore]
fn bake_gpu_vs_cpu() {
    let mut report = Report::new("bake_gpu_vs_cpu");
    let mut backend = sample(1024);

    bake_project(&mut report, &mut backend, "GPU", 1024, false, [Some(5_000.0), None, None]);
    bake_project(&mut report, &mut backend, "CPU", 1024, true, [None, None, None]);

    let rays: Vec<f64> = report.rows.iter().filter(|row| row.stage.contains(": bake ")).map(|row| ms(row.took)).collect();
    if let [gpu, cpu] = rays[..] {
        println!("  GPU speed-up over CPU: {:.1}×", cpu / gpu.max(1e-3));
    }

    report.finish();
}

/// Full recomposite of the sample project (every texture set) at 2K and 4K with nothing
/// cached (opening a project), after an edit (a bake or stack edit: pattern values kept) and
/// warm, then the display conversion the viewport uploads and the UI's budgeted path (30 ms
/// frames) after an edit.
fn composite_full(report: &mut Report, resolution: u32, budgets: [f64; 3]) {
    let mut backend = sample(resolution);
    let session = &mut backend.session;

    for caches in [Caches::None, Caches::AfterEdit, Caches::Warm] {
        invalidate(session, caches);
        let tiles = tiles_of(session);
        let start = Instant::now();
        session.update();
        let took = start.elapsed();
        let per_tile = format!("{tiles} tiles, {:.0} µs/tile", took.as_secs_f64() * 1e6 / tiles.max(1) as f64);
        let budget = (caches == Caches::None).then_some(budgets[0]);
        report.add(format!("{resolution}²: full recomposite, {} sets, {}", session.set_count(), caches.label()), took, budget, per_tile);
    }

    let cached = session.caches.iter().map(|cache| cache.below.len()).sum::<usize>();
    let covered = session.caches.iter().map(|cache| cache.covered_tiles.len()).sum::<usize>();
    const BYTES_PER_MIB: f64 = (1 << 20) as f64;
    let below_mib = session.caches.iter().map(|cache| cache.below.bytes()).sum::<usize>() as f64 / BYTES_PER_MIB;
    println!("  (below cache holds {cached} of {covered} tiles, {below_mib:.0} MiB)");
    let pattern_mib = session.caches.iter().map(|cache| cache.patterns.bytes()).sum::<usize>() as f64 / BYTES_PER_MIB;
    println!("  (pattern cache {pattern_mib:.0} MiB)");

    let start = Instant::now();
    let bytes = display_all(session);
    report.add(
        format!("{resolution}²: display conversion of every tile"),
        start.elapsed(),
        Some(budgets[1]),
        format!("{:.0} MiB", bytes as f64 / BYTES_PER_MIB),
    );

    invalidate(session, Caches::AfterEdit);
    let clock = Instant::now();
    let now = || clock.elapsed().as_secs_f64() * 1000.0;

    // Bound the benchmark if the scheduler stops making progress
    const MAX_FRAMES: usize = 100_000;
    let (mut frames, mut worst) = (0, Duration::ZERO);

    while session.pending_tiles() > 0 && frames < MAX_FRAMES {
        let start = Instant::now();
        session.update_for(30.0, now);
        worst = worst.max(start.elapsed());
        frames += 1;
    }

    report.add(
        format!("{resolution}²: budgeted recomposite, worst frame of {frames}"),
        worst,
        Some(budgets[2]),
        format!("total {:.0} ms", ms(clock.elapsed())),
    );
}

#[test]
#[ignore]
fn composite_sample_2k() {
    let mut report = Report::new("composite_sample_2k");

    composite_full(&mut report, 2048, [6_500.0, 150.0, 120.0]);

    report.finish();
}

#[test]
#[ignore]
fn composite_sample_4k() {
    let mut report = Report::new("composite_sample_4k");

    composite_full(&mut report, 4096, [24_000.0, 400.0, 120.0]);

    report.finish();
}

/// Editing the stack through the commands the UI runs: selecting the bottom layer (the
/// below-cache now caches everything but it) and toggling its visibility (every layer above it
/// recomposites).
#[test]
#[ignore]
fn composite_after_edits() {
    let mut report = Report::new("composite_after_edits");
    let mut backend = sample(2048);
    let bottom = backend.session.doc.project.texture_sets[0].stack[0];

    for (label, id, params) in [
        ("select bottom layer", "layer.select", json!({"layer": bottom.0})),
        ("toggle bottom layer visibility", "layer.visible", json!({"layer": bottom.0, "visible": false})),
        ("show it again", "layer.visible", json!({"layer": bottom.0, "visible": true})),
    ] {
        report.time(format!("{label} ({id})"), None, || backend.command_run(id, params).unwrap());
        let dirty = tiles_of(&backend.session);
        report.time(format!("  then recomposite ({dirty} tiles)"), Some(3_000.0), || backend.session.update());
    }

    report.finish();
}

/// What each layer of the sample costs, per texture set: the full recomposite of the set with
/// that layer hidden, against all visible (a group's row includes its children), and with every
/// layer hidden (the fixed per-tile cost). Report only: it points at the layer to optimize.
#[test]
#[ignore]
fn composite_per_layer() {
    let mut report = Report::new("composite_per_layer");
    let mut backend = sample(1024);
    let set_visibility = |backend: &mut Headless, ids: &[texturecraft_engine::core::LayerId], visible: bool| {
        for id in ids {
            backend.command_run("layer.visible", json!({"layer": id.0, "visible": visible})).unwrap();
        }
    };

    for texture_set in 0..backend.session.set_count() {
        backend.session.active_set = texture_set;

        // Invalidate only this set to measure its full composite
        let full = |backend: &mut Headless| {
            invalidate_set(&mut backend.session, texture_set, Caches::None);
            let start = Instant::now();
            backend.session.update();
            start.elapsed()
        };

        // Warm the thread pool and lazily built inputs before timing
        full(&mut backend);
        let all = full(&mut backend);
        let set = &backend.session.doc.project.texture_sets[texture_set];
        let tiles = backend.session.caches[texture_set].covered_tiles.len();
        let percentage = |d: Duration| format!("{:.0}% of the set", 100.0 * ms(d) / ms(all).max(1e-3));
        report.add(
            format!("set {texture_set} ({}), every layer visible", set.name),
            all,
            None,
            format!("{tiles} tiles, {:.0} µs/tile", all.as_secs_f64() * 1e6 / tiles.max(1) as f64),
        );

        let top = set.stack.clone();
        set_visibility(&mut backend, &top, false);
        let none = full(&mut backend);
        set_visibility(&mut backend, &top, true);
        report.add(format!("set {texture_set}, every layer hidden (fixed per-tile cost)"), none, None, percentage(none));

        for (id, depth) in backend.session.doc.project.walk(texture_set) {
            let layer = backend.session.doc.project.layer(id).unwrap();
            if !layer.visible {
                continue;
            }

            let name = format!("{}{}", "  ".repeat(depth), layer.name);
            set_visibility(&mut backend, &[id], false);
            let hidden = full(&mut backend);
            set_visibility(&mut backend, &[id], true);
            let saved = all.saturating_sub(hidden);
            report.add(format!("  hide {name}"), saved, None, format!("saves {}", percentage(saved)));
        }
    }

    report.finish();
}

/// Median of three cold composites of set `texture_set` (the other sets stay clean).
fn cold_median(backend: &mut Headless, texture_set: usize) -> Duration {
    let mut runs: Vec<Duration> = (0..3)
        .map(|_| {
            invalidate_set(&mut backend.session, texture_set, Caches::None);
            let start = Instant::now();
            backend.session.update();
            start.elapsed()
        })
        .collect();
    runs.sort();

    runs[1]
}

/// Inside each smart material layer (`TC_PERF_LAYERS=Paint,Rust` picks layers by name; default
/// every fill layer): what its mask costs (mask off), each mask effect (hidden one at a time) and
/// each procedural channel of its content (replaced by a uniform value). Every row is one edit
/// through the commands the UI runs, a cold composite of the set and `edit.undo`. Report only.
#[test]
#[ignore]
fn composite_per_effect() {
    use texturecraft_engine::core::{LayerKind, Source};

    let mut report = Report::new("composite_per_effect");
    let mut backend = sample(1024);
    let selected_layers: Option<Vec<String>> = std::env::var("TC_PERF_LAYERS").ok().map(|v| v.split(',').map(|name| name.trim().to_string()).collect());

    for texture_set in 0..backend.session.set_count() {
        backend.session.active_set = texture_set;

        // Warm the thread pool and lazily built inputs before timing
        cold_median(&mut backend, texture_set);
        let all = cold_median(&mut backend, texture_set);
        report.add(format!("set {texture_set} ({}), cold", backend.session.doc.project.texture_sets[texture_set].name), all, None, "");
        let percentage = |d: Duration| format!("{:.0}% of the set", 100.0 * ms(d) / ms(all).max(1e-3));

        for (id, _) in backend.session.doc.project.walk(texture_set) {
            let layer = backend.session.doc.project.layer(id).unwrap().clone();
            let LayerKind::Fill(fill) = &layer.kind else { continue };
            if selected_layers.as_ref().is_some_and(|names| !names.contains(&layer.name)) {
                continue;
            }

            // Undo each edit before measuring the next
            let mut edits: Vec<(String, &str, Value)> = Vec::new();
            if let Some(mask) = &layer.mask {
                edits.push(("mask off".into(), "mask.toggle", json!({"layer": id.0, "enabled": false})));
                for (index, effect) in mask.effects.iter().enumerate() {
                    let description = match &effect.kind {
                        texturecraft_engine::core::EffectKind::Generator(generator) => format!("{:?} generator", generator.kind),
                        texturecraft_engine::core::EffectKind::Fill { source: Source::Procedural { pattern, .. }, .. } => {
                            format!("{:?} pattern (scale {})", pattern.kind, pattern.scale)
                        }
                        kind => format!("{kind:?}").chars().take(40).collect(),
                    };
                    edits.push((
                        format!("mask effect {index}: {description}, {:?}", effect.blend),
                        "effect.set",
                        json!({"effect": effect.id.0, "visible": false}),
                    ));
                }
            }

            for (channel, source) in &fill.sources {
                if let Source::Procedural { pattern, low, .. } = source {
                    let value = if channel.components() == 3 { json!({"color": low}) } else { json!({"value": low[0]}) };
                    let mut params = json!({"layer": id.0, "channel": channel});
                    params.as_object_mut().unwrap().extend(value.as_object().unwrap().clone());
                    edits.push((format!("{channel:?}: {:?} (scale {}) → uniform", pattern.kind, pattern.scale), "layer.fill.set", params));
                }
            }

            for (label, command, params) in edits {
                backend.command_run(command, params).unwrap();
                let took = cold_median(&mut backend, texture_set);
                backend.command_run("edit.undo", Value::Null).unwrap();
                let saved = all.saturating_sub(took);
                report.add(format!("{} / {label}", layer.name), saved, None, format!("saves {}", percentage(saved)));
            }
        }
    }

    report.finish();
}

/// The smart materials project every pattern triplanar (sharpness 8), which evaluates the
/// pattern on all three planes per texel. How many of those evaluations weigh too little to
/// change a texel: under 1/65535 (the u16 step layers are stored at) or under 1/255 (8-bit
/// export). Report only: the size of the prize for skipping them.
#[test]
#[ignore]
fn triplanar_skippable_planes() {
    let backend = sample(1024);
    let session = &backend.session;
    println!("\n=== triplanar_skippable_planes (sharpness 8)");

    for texture_set in 0..session.set_count() {
        let gbuffer = &session.caches[texture_set].gbuf;
        let (mut planes, mut under_u16, mut under_u8, mut one_plane) = (0usize, 0usize, 0usize, 0usize);

        for index in 0..(gbuffer.width * gbuffer.height) as usize {
            let Some(surface) = gbuffer.surface(&session.mesh, index) else { continue };
            let weights = surface.nrm.abs().powf(8.0);
            let weights = weights / (weights.x + weights.y + weights.z).max(1e-6);
            let small = weights.to_array().map(|weight| (weight < 1.0 / 65535.0, weight < 1.0 / 255.0));
            planes += 3;
            under_u16 += small.iter().filter(|weight| weight.0).count();
            under_u8 += small.iter().filter(|weight| weight.1).count();
            one_plane += usize::from(small.iter().filter(|weight| weight.0).count() == 2);
        }

        let percentage = |count: usize, total: usize| 100.0 * count as f64 / total.max(1) as f64;
        println!(
            "  set {texture_set}: {:.0}% of plane evaluations weigh < 1/65535, {:.0}% < 1/255; {:.0}% of texels need one plane only",
            percentage(under_u16, planes),
            percentage(under_u8, planes),
            percentage(one_plane, planes / 3)
        );
    }
}

/// One interactive painting frame on the sample project at 2K: stamping, compositing the dirty
/// tiles, display conversion; then the commit at release. Budgets are per frame (60 fps is 16 ms).
#[test]
#[ignore]
fn composite_painting_frames() {
    let mut report = Report::new("composite_painting_frames");
    let mut backend = sample(2048);
    let session = &mut backend.session;
    display_all(session);

    for (size, brush) in [(0.02, "default brush"), (0.08, "large brush")] {
        session.tool.brush.size = size;
        let (mut stamping, mut compositing, mut conversion, mut worst, mut tiles) = (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, 0);
        session.begin_stroke(Vec2::new(350.0, 400.0), session.viewport, 1.0).unwrap();
        let frames = 60;
        for i in 0..frames {
            let position = Vec2::new(350.0 + i as f32 * 5.0, 400.0 + (i as f32 * 0.2).sin() * 60.0);
            let start = Instant::now();
            session.continue_stroke(position, 1.0);
            let stamp_time = start.elapsed();
            tiles += tiles_of(session);

            let start = Instant::now();
            session.update();
            let composite_time = start.elapsed();

            let start = Instant::now();
            display_all(session);
            let conversion_time = start.elapsed();
            stamping += stamp_time;
            compositing += composite_time;
            conversion += conversion_time;
            worst = worst.max(stamp_time + composite_time + conversion_time);
        }

        let frame_count = frames as u32;
        let note = format!("{:.0} dirty tiles/frame", tiles as f64 / frames as f64);
        report.add(format!("{brush}: stamp per frame"), stamping / frame_count, Some(8.0), "");
        report.add(format!("{brush}: composite per frame"), compositing / frame_count, Some(16.0), note);
        report.add(format!("{brush}: display per frame"), conversion / frame_count, Some(8.0), "");
        report.add(format!("{brush}: worst frame"), worst, Some(50.0), "");
        report.time(format!("{brush}: end_stroke"), Some(100.0), || session.end_stroke().unwrap());
        report.time(format!("{brush}: recomposite after commit"), Some(200.0), || session.update());
    }

    report.finish();
}
