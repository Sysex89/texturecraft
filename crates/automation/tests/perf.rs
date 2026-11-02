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
use texturecraft_engine::core::BakedMap;
use texturecraft_engine::geom::glam::Vec2;
use texturecraft_engine::project_cmds::sample_session;

// ---------------------------------------------------------------------------------------------
// Reporting

struct Row {
    stage: String,
    took: Duration,
    budget: Option<Duration>,
    note: String,
}

struct Report {
    test: &'static str,
    rows: Vec<Row>,
}

fn scale() -> f64 {
    std::env::var("TC_PERF_SCALE").ok().and_then(|s| s.parse::<f64>().ok()).filter(|s| s.is_finite() && *s > 0.0).unwrap_or(1.0)
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

impl Report {
    fn new(test: &'static str) -> Report {
        println!("\n=== {test} ({} threads, {} build)", rayon::current_num_threads(), if cfg!(debug_assertions) { "debug" } else { "release" });
        Report { test, rows: Vec::new() }
    }

    /// Record a stage; `budget_ms` is scaled by `TC_PERF_SCALE`.
    fn add(&mut self, stage: impl Into<String>, took: Duration, budget_ms: Option<f64>, note: impl Into<String>) {
        let row = Row { stage: stage.into(), took, budget: budget_ms.map(|b| Duration::from_secs_f64(b * scale() / 1000.0)), note: note.into() };
        let budget = row.budget.map_or(String::new(), |b| format!(" / {:>9.1} ms{}", ms(b), if row.took > b { "  OVER" } else { "" }));
        println!("  {:<58} {:>9.1} ms{budget}  {}", row.stage, ms(row.took), row.note);
        self.rows.push(row);
    }

    /// Time `f` as one stage.
    fn time<T>(&mut self, stage: impl Into<String>, budget_ms: Option<f64>, f: impl FnOnce() -> T) -> T {
        let t = Instant::now();
        let out = f();
        self.add(stage, t.elapsed(), budget_ms, "");
        out
    }

    /// Write the JSON report and fail on blown budgets (release builds only).
    fn finish(self) {
        if let Ok(dir) = std::env::var("TC_PERF_OUT") {
            let rows: Vec<Value> =
                self.rows.iter().map(|r| json!({"stage": r.stage, "ms": ms(r.took), "budgetMs": r.budget.map(ms), "note": r.note})).collect();
            let doc = json!({"test": self.test, "threads": rayon::current_num_threads(), "release": !cfg!(debug_assertions), "stages": rows});
            // Tests run in the crate's directory: a relative path is taken from the workspace root.
            let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../..").join(dir);
            std::fs::create_dir_all(&dir).unwrap();
            let path = dir.join(format!("{}.json", self.test));
            std::fs::write(&path, serde_json::to_string_pretty(&doc).unwrap()).unwrap();
            println!("  report: {}", path.display());
        }
        let over: Vec<String> =
            self.rows.iter().filter_map(|r| r.budget.filter(|b| r.took > *b).map(|b| format!("{}: {:.1} ms > {:.1} ms", r.stage, ms(r.took), ms(b)))).collect();
        if over.is_empty() {
            return;
        }
        if cfg!(debug_assertions) || std::env::var("TC_PERF_BUDGETS").is_ok_and(|v| v == "off") {
            println!("  ({} stages over budget; not failing: {})", over.len(), if cfg!(debug_assertions) { "debug build" } else { "TC_PERF_BUDGETS=off" });
            return;
        }
        panic!("{}: {} stages over budget:\n  {}", self.test, over.len(), over.join("\n  "));
    }
}

// ---------------------------------------------------------------------------------------------
// Setup

/// The startup sample project (material ball, two texture sets with smart materials whose
/// generators read the baked maps) at `res`, in a headless backend with the GPU tracer installed.
/// The sample's own bake is a cheap 256² one: tests rebake at the size they measure.
fn sample(res: u32) -> Headless {
    // `Headless::new` installs the GPU tracer; the sample session must bake after it.
    let mut h = Headless::new(Headless::default_session().unwrap(), Filesystem::Denied);
    let mut s = sample_session(res, 256).unwrap();
    s.viewport = Vec2::new(1000.0, 880.0);
    h.session = s;
    h.session.update();
    h
}

/// Every tile the viewport would upload, converted to every display format as the UI does
/// (in parallel); returns the bytes.
fn display_all(s: &mut Session) -> usize {
    let mut bytes = 0;
    for ts in 0..s.set_count() {
        let tiles = s.take_uploads(ts);
        let baked = s.baked(ts).cloned().unwrap_or_default();
        let comp = &s.caches[ts].composite;
        bytes += tiles.par_iter().flat_map_iter(|&(tx, ty)| DisplayKind::ALL.map(|k| tile_bytes(comp, &baked, k, tx, ty).len())).sum::<usize>();
    }
    bytes
}

fn tiles_of(s: &Session) -> usize {
    s.caches.iter().map(|c| c.dirty.len()).sum()
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
fn invalidate(s: &mut Session, caches: Caches) {
    for ts in 0..s.set_count() {
        invalidate_set(s, ts, caches);
    }
}

fn invalidate_set(s: &mut Session, ts: usize, caches: Caches) {
    let c = &mut s.caches[ts];
    c.invalidate_all();
    if caches != Caches::Warm {
        c.below.clear();
    }
    if caches == Caches::None {
        c.patterns.clear();
    }
}

/// Bake every texture set at `bake_res` the way the app does: the ray-casting task (the worker
/// thread's part), then `apply_bake` and the recomposite of the layers that read the maps (the
/// frame's part). `cpu` skips the GPU tracer. Budgets are per whole-project stage.
fn bake_project(r: &mut Report, h: &mut Headless, label: &str, bake_res: u32, cpu: bool, budgets: [Option<f64>; 3]) {
    let s = &mut h.session;
    for ts in 0..s.set_count() {
        s.doc.project.texture_set_mut(ts).unwrap().bake.resolution = bake_res;
    }
    let (mut rays, mut apply, mut tracer) = (Duration::ZERO, Duration::ZERO, String::new());
    let mut outs = Vec::new();
    for ts in 0..s.set_count() {
        s.begin_bake(ts);
        let mut task = s.bake_task(ts).unwrap();
        if cpu {
            task.tracer = None;
        }
        let t = Instant::now();
        let out = task.run(&Progress::new()).unwrap();
        rays += t.elapsed();
        tracer = out.tracer.clone();
        outs.push((ts, out));
    }
    for (ts, out) in outs {
        let t = Instant::now();
        s.apply_bake(ts, out).unwrap();
        apply += t.elapsed();
    }
    let maps = s.doc.project.texture_sets[0].bake.maps.len();
    r.add(format!("{label}: bake {} sets × {maps} maps at {bake_res}²", s.set_count()), rays, budgets[0], format!("[{tracer}]"));
    r.add(format!("{label}: apply_bake (undo snapshot + store)"), apply, budgets[1], "");
    let dirty = tiles_of(s);
    r.time(format!("{label}: recomposite after bake ({dirty} tiles)"), budgets[2], || s.update());
}

// ---------------------------------------------------------------------------------------------
// Baking

/// The whole-project bake at the app's default settings (1024², 32 rays, AA 1), then 2048².
#[test]
#[ignore]
fn bake_sample_project() {
    let mut r = Report::new("bake_sample_project");
    let mut h = sample(2048);
    bake_project(&mut r, &mut h, "1K", 1024, false, [Some(5_000.0), Some(200.0), Some(6_500.0)]);
    bake_project(&mut r, &mut h, "2K", 2048, false, [Some(23_000.0), Some(400.0), Some(6_500.0)]);
    // The same through the command an agent runs: bake + recomposite in one call.
    r.time("bake.run (command, all sets, 1K)", Some(12_000.0), || h.command_run("bake.run", json!({"ts": "all", "resolution": 1024})).unwrap());
    r.finish();
}

/// Each mesh map alone at 2048² on texture set 0: which baker is the slow one.
#[test]
#[ignore]
fn bake_each_map() {
    let mut r = Report::new("bake_each_map");
    let h = sample(1024);
    let s = &h.session;
    let maps: Vec<_> = s.doc.project.texture_sets[0].bake.maps.iter().copied().collect();
    for map in maps {
        let mut task = s.bake_task_for(0, Some(map)).unwrap();
        task.settings.resolution = 2048;
        let t = Instant::now();
        let out = task.run(&Progress::new()).unwrap();
        // The hemisphere bakers cast `rays` per texel; the others one or a few.
        let budget = if matches!(map, BakedMap::AmbientOcclusion | BakedMap::Thickness | BakedMap::BentNormal) { 6_000.0 } else { 300.0 };
        r.add(format!("set 0, {} at 2048²", map.label()), t.elapsed(), Some(budget), format!("[{}]", out.tracer));
    }
    r.finish();
}

/// GPU against CPU rays on the same 1K bake (no budget on the CPU row: it is the reference).
#[test]
#[ignore]
fn bake_gpu_vs_cpu() {
    let mut r = Report::new("bake_gpu_vs_cpu");
    let mut h = sample(1024);
    bake_project(&mut r, &mut h, "GPU", 1024, false, [Some(5_000.0), None, None]);
    bake_project(&mut r, &mut h, "CPU", 1024, true, [None, None, None]);
    let rays: Vec<f64> = r.rows.iter().filter(|row| row.stage.contains(": bake ")).map(|row| ms(row.took)).collect();
    if let [gpu, cpu] = rays[..] {
        println!("  GPU speed-up over CPU: {:.1}×", cpu / gpu.max(1e-3));
    }
    r.finish();
}

// ---------------------------------------------------------------------------------------------
// Compositing

/// Full recomposite of the sample project (every texture set) at 2K and 4K with nothing
/// cached (opening a project), after an edit (a bake or stack edit: pattern values kept) and
/// warm, then the display conversion the viewport uploads and the UI's budgeted path (30 ms
/// frames) after an edit.
fn composite_full(r: &mut Report, res: u32, budgets: [f64; 3]) {
    let mut h = sample(res);
    let s = &mut h.session;
    for caches in [Caches::None, Caches::AfterEdit, Caches::Warm] {
        invalidate(s, caches);
        let tiles = tiles_of(s);
        let t = Instant::now();
        s.update();
        let took = t.elapsed();
        let per_tile = format!("{tiles} tiles, {:.0} µs/tile", took.as_secs_f64() * 1e6 / tiles.max(1) as f64);
        let budget = (caches == Caches::None).then_some(budgets[0]);
        r.add(format!("{res}²: full recomposite, {} sets, {}", s.set_count(), caches.label()), took, budget, per_tile);
    }
    let pattern_mb = s.caches.iter().map(|c| c.patterns.bytes()).sum::<usize>() as f64 / 1048576.0;
    println!("  (pattern cache {pattern_mb:.0} MB)");
    let cached = s.caches.iter().map(|c| c.below.len()).sum::<usize>();
    let covered = s.caches.iter().map(|c| c.covered_tiles.len()).sum::<usize>();
    let below_mb = s.caches.iter().map(|c| c.below.bytes()).sum::<usize>() as f64 / 1048576.0;
    println!("  (below cache holds {cached} of {covered} tiles, {below_mb:.0} MB)");
    let t = Instant::now();
    let bytes = display_all(s);
    r.add(format!("{res}²: display conversion of every tile"), t.elapsed(), Some(budgets[1]), format!("{:.0} MB", bytes as f64 / 1048576.0));

    invalidate(s, Caches::AfterEdit);
    let clock = Instant::now();
    let now = || clock.elapsed().as_secs_f64() * 1000.0;
    let (mut frames, mut worst) = (0, Duration::ZERO);
    while s.pending_tiles() > 0 && frames < 100_000 {
        let t = Instant::now();
        s.update_for(30.0, now);
        worst = worst.max(t.elapsed());
        frames += 1;
    }
    r.add(format!("{res}²: budgeted recomposite, worst frame of {frames}"), worst, Some(budgets[2]), format!("total {:.0} ms", ms(clock.elapsed())));
}

#[test]
#[ignore]
fn composite_sample_2k() {
    let mut r = Report::new("composite_sample_2k");
    composite_full(&mut r, 2048, [6_500.0, 150.0, 120.0]);
    r.finish();
}

#[test]
#[ignore]
fn composite_sample_4k() {
    let mut r = Report::new("composite_sample_4k");
    composite_full(&mut r, 4096, [24_000.0, 400.0, 120.0]);
    r.finish();
}

/// The full recomposite after an edit, frame-sliced (30 ms budget, as the UI does by default)
/// against background compositing driven every 8 ms (a frame's time with the UI's own work):
/// total time, and what compositing costs the worst frame. Budgets: [background total, worst
/// background frame].
fn composite_background(r: &mut Report, res: u32, budgets: [f64; 2]) {
    let mut h = sample(res);
    let s = &mut h.session;
    invalidate(s, Caches::AfterEdit);
    let clock = Instant::now();
    let now = || clock.elapsed().as_secs_f64() * 1000.0;
    let (mut frames, mut worst) = (0, Duration::ZERO);
    while s.pending_tiles() > 0 && frames < 100_000 {
        let t = Instant::now();
        s.update_for(30.0, now);
        worst = worst.max(t.elapsed());
        frames += 1;
    }
    r.add(format!("{res}²: frame-sliced, total ({frames} frames)"), clock.elapsed(), None, format!("worst frame {:.1} ms", ms(worst)));

    let mut bg = texturecraft_engine::BackgroundCompositor::new();
    invalidate(s, Caches::AfterEdit);
    let clock = Instant::now();
    let (mut frames, mut worst) = (0, Duration::ZERO);
    while (s.pending_tiles() > 0 || bg.busy()) && frames < 100_000 {
        let t = Instant::now();
        bg.drive(s, 30.0, || 0.0);
        worst = worst.max(t.elapsed());
        frames += 1;
        std::thread::sleep(Duration::from_millis(8));
    }
    r.add(format!("{res}²: background, total ({frames} frames)"), clock.elapsed(), Some(budgets[0]), "");
    r.add(format!("{res}²: background, worst frame's compositing"), worst, Some(budgets[1]), "applying the tiles that arrived (padded by the worker)");

    // When the tiles the 3D view shows are done: in set order, then those first.
    let view = texturecraft_engine::Focus { view3d: true, uv: None };
    let shown: Vec<std::collections::BTreeSet<(u32, u32)>> = (0..s.set_count()).map(|ts| s.visible_tiles(ts, &view)).collect();
    let n: usize = shown.iter().map(|v| v.len()).sum();
    let covered: usize = s.caches.iter().map(|c| c.covered_tiles.len()).sum();
    // Covered texels, the cost the tiles carry: per set, shown and all.
    let texels = |c: &texturecraft_engine::SetCache, only: Option<&std::collections::BTreeSet<(u32, u32)>>| -> usize {
        let w = c.gbuf.width;
        (0..c.gbuf.tri.len()).filter(|&i| c.gbuf.covered(i) && only.is_none_or(|v| v.contains(&((i as u32 % w) / 64, (i as u32 / w) / 64)))).count()
    };
    let shown_texels: usize = s.caches.iter().zip(&shown).map(|(c, v)| texels(c, Some(v))).sum();
    let all_texels: usize = s.caches.iter().map(|c| texels(c, None)).sum();
    println!("  (the 3D view's tiles hold {:.0}% of the covered texels)", 100.0 * shown_texels as f64 / all_texels.max(1) as f64);
    for (label, focus) in [("set order", texturecraft_engine::Focus::default()), ("3D view first", view)] {
        let mut bg = texturecraft_engine::BackgroundCompositor::new();
        bg.set_focus(focus);
        invalidate(s, Caches::AfterEdit);
        // A delivered tile is new data: a shown tile is done once its data is not what it was
        // (tiles in flight include the ring a job pads, so they cannot tell).
        let tile_ptr = |c: &texturecraft_engine::SetCache, t: (u32, u32)| {
            c.composite.channels.values().next().and_then(|img| img.tile_handle(t.0, t.1)).map(|d| std::sync::Arc::as_ptr(d) as usize)
        };
        // Per set: each shown tile and the address of its data before.
        type Before = Vec<((u32, u32), Option<usize>)>;
        let before: Vec<Before> = s.caches.iter().zip(&shown).map(|(c, v)| v.iter().map(|&t| (t, tile_ptr(c, t))).collect()).collect();
        let clock = Instant::now();
        let mut seen: Option<Duration> = None;
        while (s.pending_tiles() > 0 || bg.busy()) && clock.elapsed() < Duration::from_secs(120) {
            bg.drive(s, 30.0, || 0.0);
            let waiting = s.caches.iter().zip(&before).any(|(c, v)| v.iter().any(|&(t, p)| tile_ptr(c, t) == p));
            if !waiting && seen.is_none() {
                seen = Some(clock.elapsed());
            }
            std::thread::sleep(Duration::from_millis(8));
        }
        let total = clock.elapsed();
        let seen = seen.unwrap_or(total);
        r.add(format!("{res}²: background, {label}: the 3D view's tiles done"), seen, None, format!("{n} of {covered} tiles; all done at {:.0} ms", ms(total)));
    }
}

#[test]
#[ignore]
fn composite_background_2k() {
    let mut r = Report::new("composite_background_2k");
    composite_background(&mut r, 2048, [1_300.0, 8.0]);
    r.finish();
}

#[test]
#[ignore]
fn composite_background_4k() {
    let mut r = Report::new("composite_background_4k");
    composite_background(&mut r, 4096, [4_700.0, 8.0]);
    r.finish();
}

/// Editing the stack through the commands the UI runs: selecting the bottom layer (the
/// below-cache now caches everything but it) and toggling its visibility (every layer above it
/// recomposites).
#[test]
#[ignore]
fn composite_after_edits() {
    let mut r = Report::new("composite_after_edits");
    let mut h = sample(2048);
    let bottom = h.session.doc.project.texture_sets[0].stack[0];
    for (label, id, p) in [
        ("select bottom layer", "layer.select", json!({"layer": bottom.0})),
        ("toggle bottom layer visibility", "layer.visible", json!({"layer": bottom.0, "visible": false})),
        ("show it again", "layer.visible", json!({"layer": bottom.0, "visible": true})),
    ] {
        r.time(format!("{label} ({id})"), None, || h.command_run(id, p).unwrap());
        let dirty = tiles_of(&h.session);
        r.time(format!("  then recomposite ({dirty} tiles)"), Some(3_000.0), || h.session.update());
    }
    r.finish();
}

/// What each layer of the sample costs, per texture set: the full recomposite of the set with
/// that layer hidden, against all visible (a group's row includes its children), and with every
/// layer hidden (the fixed per-tile cost). Report only: it points at the layer to optimize.
#[test]
#[ignore]
fn composite_per_layer() {
    let mut r = Report::new("composite_per_layer");
    let mut h = sample(1024);
    let set_visibility = |h: &mut Headless, ids: &[texturecraft_engine::core::LayerId], visible: bool| {
        for id in ids {
            h.command_run("layer.visible", json!({"layer": id.0, "visible": visible})).unwrap();
        }
    };
    for ts in 0..h.session.set_count() {
        h.session.active_set = ts;
        // A cold composite of this set only (the others stay clean): with the below cache
        // filled, a recomposite would only evaluate the selected layer.
        let full = |h: &mut Headless| {
            invalidate_set(&mut h.session, ts, Caches::None);
            let t = Instant::now();
            h.session.update();
            t.elapsed()
        };
        full(&mut h); // the thread pool and lazily built inputs
        let all = full(&mut h);
        let set = &h.session.doc.project.texture_sets[ts];
        let tiles = h.session.caches[ts].covered_tiles.len();
        let pct = |d: Duration| format!("{:.0}% of the set", 100.0 * ms(d) / ms(all).max(1e-3));
        r.add(
            format!("set {ts} ({}), every layer visible", set.name),
            all,
            None,
            format!("{tiles} tiles, {:.0} µs/tile", all.as_secs_f64() * 1e6 / tiles.max(1) as f64),
        );
        let top = set.stack.clone();
        set_visibility(&mut h, &top, false);
        let none = full(&mut h);
        set_visibility(&mut h, &top, true);
        r.add(format!("set {ts}, every layer hidden (fixed per-tile cost)"), none, None, pct(none));
        for (id, depth) in h.session.doc.project.walk(ts) {
            let layer = h.session.doc.project.layer(id).unwrap();
            if !layer.visible {
                continue;
            }
            let name = format!("{}{}", "  ".repeat(depth), layer.name);
            set_visibility(&mut h, &[id], false);
            let hidden = full(&mut h);
            set_visibility(&mut h, &[id], true);
            let saved = all.saturating_sub(hidden);
            r.add(format!("  hide {name}"), saved, None, format!("saves {}", pct(saved)));
        }
    }
    r.finish();
}

/// Median of three cold composites of set `ts` (the other sets stay clean).
fn cold_median(h: &mut Headless, ts: usize) -> Duration {
    let mut runs: Vec<Duration> = (0..3)
        .map(|_| {
            invalidate_set(&mut h.session, ts, Caches::None);
            let t = Instant::now();
            h.session.update();
            t.elapsed()
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
    let mut r = Report::new("composite_per_effect");
    let mut h = sample(1024);
    let only: Option<Vec<String>> = std::env::var("TC_PERF_LAYERS").ok().map(|v| v.split(',').map(|s| s.trim().to_string()).collect());
    for ts in 0..h.session.set_count() {
        h.session.active_set = ts;
        cold_median(&mut h, ts); // the thread pool and lazily built inputs
        let all = cold_median(&mut h, ts);
        r.add(format!("set {ts} ({}), cold", h.session.doc.project.texture_sets[ts].name), all, None, "");
        let pct = |d: Duration| format!("{:.0}% of the set", 100.0 * ms(d) / ms(all).max(1e-3));
        for (id, _) in h.session.doc.project.walk(ts) {
            let layer = h.session.doc.project.layer(id).unwrap().clone();
            let LayerKind::Fill(fill) = &layer.kind else { continue };
            if only.as_ref().is_some_and(|names| !names.contains(&layer.name)) {
                continue;
            }
            // (row label, command, params): each row is undone before the next.
            let mut edits: Vec<(String, &str, Value)> = Vec::new();
            if let Some(mask) = &layer.mask {
                edits.push(("mask off".into(), "mask.toggle", json!({"layer": id.0, "enabled": false})));
                for (i, e) in mask.effects.iter().enumerate() {
                    let what = match &e.kind {
                        texturecraft_engine::core::EffectKind::Generator(g) => format!("{:?} generator", g.kind),
                        texturecraft_engine::core::EffectKind::Fill { source: Source::Procedural { pattern, .. }, .. } => {
                            format!("{:?} pattern (scale {})", pattern.kind, pattern.scale)
                        }
                        k => format!("{k:?}").chars().take(40).collect(),
                    };
                    edits.push((format!("mask effect {i}: {what}, {:?}", e.blend), "effect.set", json!({"effect": e.id.0, "visible": false})));
                }
            }
            for (channel, source) in &fill.sources {
                if let Source::Procedural { pattern, low, .. } = source {
                    let value = if channel.components() == 3 { json!({"color": low}) } else { json!({"value": low[0]}) };
                    let mut p = json!({"layer": id.0, "channel": channel});
                    p.as_object_mut().unwrap().extend(value.as_object().unwrap().clone());
                    edits.push((format!("{channel:?}: {:?} (scale {}) → uniform", pattern.kind, pattern.scale), "layer.fill.set", p));
                }
            }
            for (label, cmd, p) in edits {
                h.command_run(cmd, p).unwrap();
                let took = cold_median(&mut h, ts);
                h.command_run("edit.undo", Value::Null).unwrap();
                let saved = all.saturating_sub(took);
                r.add(format!("{} / {label}", layer.name), saved, None, format!("saves {}", pct(saved)));
            }
        }
    }
    r.finish();
}

/// The smart materials project every pattern triplanar (sharpness 8), which evaluates the
/// pattern on all three planes per texel. How many of those evaluations weigh too little to
/// change a texel: under 1/65535 (the u16 step layers are stored at) or under 1/255 (8-bit
/// export). Report only: the size of the prize for skipping them.
#[test]
#[ignore]
fn triplanar_skippable_planes() {
    let h = sample(1024);
    let s = &h.session;
    println!("\n=== triplanar_skippable_planes (sharpness 8)");
    for ts in 0..s.set_count() {
        let g = &s.caches[ts].gbuf;
        let (mut planes, mut under_u16, mut under_u8, mut one_plane) = (0usize, 0usize, 0usize, 0usize);
        for i in 0..(g.width * g.height) as usize {
            let Some(surf) = g.surface(&s.mesh, i) else { continue };
            let w = surf.nrm.abs().powf(8.0);
            let w = w / (w.x + w.y + w.z).max(1e-6);
            let small = w.to_array().map(|w| (w < 1.0 / 65535.0, w < 1.0 / 255.0));
            planes += 3;
            under_u16 += small.iter().filter(|s| s.0).count();
            under_u8 += small.iter().filter(|s| s.1).count();
            one_plane += usize::from(small.iter().filter(|s| s.0).count() == 2);
        }
        let pct = |n: usize, of: usize| 100.0 * n as f64 / of.max(1) as f64;
        println!(
            "  set {ts}: {:.0}% of plane evaluations weigh < 1/65535, {:.0}% < 1/255; {:.0}% of texels need one plane only",
            pct(under_u16, planes),
            pct(under_u8, planes),
            pct(one_plane, planes / 3)
        );
    }
}

/// One interactive painting frame on the sample project at 2K: stamping, compositing the dirty
/// tiles, display conversion; then the commit at release. Budgets are per frame (60 fps is 16 ms).
#[test]
#[ignore]
fn composite_painting_frames() {
    let mut r = Report::new("composite_painting_frames");
    let mut h = sample(2048);
    let s = &mut h.session;
    display_all(s);
    for (size, brush) in [(0.02, "default brush"), (0.08, "large brush")] {
        s.tool.brush.size = size;
        let (mut stamp, mut comp, mut conv, mut worst, mut tiles) = (Duration::ZERO, Duration::ZERO, Duration::ZERO, Duration::ZERO, 0);
        s.begin_stroke(Vec2::new(350.0, 400.0), s.viewport, 1.0).unwrap();
        let frames = 60;
        for i in 0..frames {
            let p = Vec2::new(350.0 + i as f32 * 5.0, 400.0 + (i as f32 * 0.2).sin() * 60.0);
            let t = Instant::now();
            s.continue_stroke(p, 1.0);
            let a = t.elapsed();
            tiles += tiles_of(s);
            let t = Instant::now();
            s.update();
            let b = t.elapsed();
            let t = Instant::now();
            display_all(s);
            let c = t.elapsed();
            (stamp, comp, conv, worst) = (stamp + a, comp + b, conv + c, worst.max(a + b + c));
        }
        let f = frames as u32;
        let note = format!("{:.0} dirty tiles/frame", tiles as f64 / frames as f64);
        r.add(format!("{brush}: stamp per frame"), stamp / f, Some(8.0), "");
        r.add(format!("{brush}: composite per frame"), comp / f, Some(16.0), note);
        r.add(format!("{brush}: display per frame"), conv / f, Some(8.0), "");
        r.add(format!("{brush}: worst frame"), worst, Some(50.0), "");
        r.time(format!("{brush}: end_stroke"), Some(100.0), || s.end_stroke().unwrap());
        r.time(format!("{brush}: recomposite after commit"), Some(200.0), || s.update());
    }
    r.finish();
}
