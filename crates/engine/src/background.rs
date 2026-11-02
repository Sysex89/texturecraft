//! Background compositing (desktop): the dirty tiles of a texture set are composited on a
//! worker thread while the UI keeps drawing, instead of in slices of each frame
//! ([`Session::update_for`]). One job runs at a time; each frame [`BackgroundCompositor::drive`]
//! applies the tiles that arrived and starts the next job.
//!
//! A job works on a snapshot of what the compositor reads and of the set's composite
//! (copy-on-write, so cheap) and the set's layout and pattern cache. It also pads: a tile goes
//! out once it and its neighbours in the job are composited, padded from them (neighbours
//! outside the job pad from the snapshot), so applying a delivery costs the UI a pointer swap
//! per tile; padding on the UI thread competed with the worker for every core. The job's tiles
//! and the ring of tiles around them its padding reaches move from `dirty` to
//! [`SetCache::in_flight`]. A
//! composite on the session's own thread meanwhile (a command calling `update`, a stroke) takes
//! the tiles it composited or padded out of `in_flight`, so the job's result for it, older by then, is dropped; a result
//! for a tile still listed is applied even if an edit dirtied the tile again, since it is newer
//! than what the tile shows, and the tile stays dirty for the next job. Strokes composite on
//! the session's thread (their frames must show at once): a stroke cancels the job.
//!
//! [`SetCache::in_flight`]: crate::SetCache::in_flight

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, TryRecvError};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use texturecraft_compose::{BakedMaps, Composite, ImageStore, Inputs, PatternCache, PixelStore, composite_tiles, pad_exact, tile_neighbourhood};
use texturecraft_core::{Channel, Project};
use texturecraft_geom::glam::Vec3;
use texturecraft_geom::{Mesh, UvGBuffer};
use texturecraft_texel::{DilationMap, TileData};

use crate::Session;

/// Tiles a worker composites between two deliveries (and cancellation checks).
pub const JOB_CHUNK: usize = 64;

/// One composited tile: its data per channel (None: no covered texel).
type Tile = ((u32, u32), Vec<(Channel, Option<TileData>)>);

/// What a job reads, owned (sent to the worker).
struct Job {
    project: Project,
    pixels: PixelStore,
    baked: BakedMaps,
    images: ImageStore,
    mesh: Arc<Mesh>,
    gbuf: Arc<UvGBuffer>,
    bounds: (Vec3, Vec3),
    patterns: Arc<PatternCache>,
    dilation: Arc<DilationMap>,
    /// The set's composite when the job started: the tiles around the job pad from it.
    base: Composite,
    ts: usize,
    tiles: Vec<(u32, u32)>,
}

/// Threads for background composites: one fewer than the machine has, so the UI keeps a core.
fn pool() -> Option<&'static rayon::ThreadPool> {
    static POOL: OnceLock<Option<rayon::ThreadPool>> = OnceLock::new();
    POOL.get_or_init(|| {
        let n = std::thread::available_parallelism().map_or(2, std::num::NonZeroUsize::get);
        rayon::ThreadPoolBuilder::new().num_threads(n.saturating_sub(1).max(1)).thread_name(|i| format!("texturecraft-composite-{i}")).build().ok()
    })
    .as_ref()
}

impl Job {
    /// Composite the job's tiles chunk by chunk and send the tiles that are ready (composited
    /// and padded) after each, until done, cancelled or nobody listens.
    fn run(self, cancel: &AtomicBool, tx: &Sender<Vec<Tile>>) {
        let Job { project, pixels, baked, images, mesh, gbuf, bounds, patterns, dilation, base, ts, mut tiles } = self;
        let inp = Inputs {
            project: &project,
            ts,
            mesh: &mesh,
            gbuf: &gbuf,
            pixels: &pixels,
            baked: &baked,
            preview: None,
            images: &images,
            bounds,
            patterns: Some(&*patterns),
        };
        let mut out = base;
        // Block by block (16 × 16 tiles), as the frame-sliced path takes them.
        tiles.sort_by_key(|&(x, y)| (y / 16, x / 16, y, x));
        let job: BTreeSet<(u32, u32)> = tiles.iter().copied().collect();
        let mut done = BTreeSet::new();
        // Tiles to pad and send once their neighbours in the job are composited.
        let mut waiting = BTreeSet::new();
        for chunk in tiles.chunks(JOB_CHUNK) {
            if cancel.load(Ordering::Relaxed) {
                return;
            }
            composite_tiles(&inp, &mut out, chunk);
            done.extend(chunk.iter().copied());
            waiting.extend(tile_neighbourhood(&out, chunk));
            let ready: Vec<(u32, u32)> =
                waiting.iter().copied().filter(|t| tile_neighbourhood(&out, &[*t]).iter().all(|n| !job.contains(n) || done.contains(n))).collect();
            for t in &ready {
                waiting.remove(t);
            }
            pad_exact(&mut out, &dilation, &ready);
            let delivery: Vec<Tile> =
                ready.iter().map(|&(x, y)| ((x, y), out.channels.iter().map(|(c, img)| (*c, img.tile_handle(x, y).cloned())).collect())).collect();
            if !delivery.is_empty() && tx.send(delivery).is_err() {
                return;
            }
        }
    }
}

/// The running job, seen from the session's thread.
struct Running {
    ts: usize,
    /// The layout the job composites for: results for another one are dropped.
    gbuf: Arc<UvGBuffer>,
    channels: Vec<Channel>,
    /// The tiles the job composites (`in_flight` also holds the ring around them it pads).
    job: BTreeSet<(u32, u32)>,
    cancel: Arc<AtomicBool>,
    rx: Receiver<Vec<Tile>>,
}

/// See the module docs. Lives beside the session (the UI's GPU link, a test).
#[derive(Default)]
pub struct BackgroundCompositor {
    running: Option<Running>,
    /// A worker thread could not be started, or jobs kept stopping early: composite in frame
    /// slices from then on.
    unavailable: bool,
    /// Jobs in a row that stopped before delivering every tile without being cancelled.
    failures: u32,
}

/// Jobs in a row that may stop early before background compositing is given up.
const MAX_FAILURES: u32 = 3;

impl BackgroundCompositor {
    pub fn new() -> Self {
        Self::default()
    }

    /// A job is running.
    pub fn busy(&self) -> bool {
        self.running.is_some()
    }

    /// Composite in the background: apply the tiles that arrived, start the next job. While a
    /// stroke runs (or no worker can start), composite on this thread within `budget_ms` of
    /// `clock` instead ([`Session::update_for`]). Returns whether a composite changed.
    pub fn drive(&mut self, s: &mut Session, budget_ms: f64, clock: impl Fn() -> f64) -> bool {
        if s.stroke_active() || self.unavailable {
            self.cancel(s);
            return s.update_for(budget_ms, clock);
        }
        let changed = self.collect(s, None);
        if self.running.is_none() {
            self.dispatch(s);
        }
        changed
    }

    /// Block until every pending tile is composited (tests, a save that needs the result, or
    /// leaving background mode).
    pub fn finish(&mut self, s: &mut Session) {
        loop {
            self.drive(s, f64::INFINITY, || 0.0);
            if self.running.is_none() && s.pending_tiles() == 0 {
                return;
            }
            self.collect(s, Some(Duration::from_millis(20)));
        }
    }

    /// Stop the running job; the tiles it has not delivered are dirty again.
    pub fn cancel(&mut self, s: &mut Session) {
        let Some(r) = self.running.take() else { return };
        r.cancel.store(true, Ordering::Relaxed);
        if let Some(c) = s.caches.get_mut(r.ts) {
            let back = std::mem::take(&mut c.in_flight);
            c.dirty.extend(back);
        }
    }

    /// Start a job on the first texture set (the active one first) with tiles to composite.
    fn dispatch(&mut self, s: &mut Session) {
        let active = s.ts();
        let order = std::iter::once(active).chain((0..s.caches.len()).filter(|t| *t != active));
        let Some(ts) = order.into_iter().find(|t| s.caches.get(*t).is_some_and(|c| !c.dirty.is_empty() || !c.settle.is_empty())) else { return };
        let Ok(set) = s.doc.project.texture_set(ts) else { return };
        let channels = set.channel_list();
        let Some(c) = s.caches.get_mut(ts) else { return };
        let mut tiles: BTreeSet<(u32, u32)> = std::mem::take(&mut c.dirty);
        tiles.append(&mut c.settle);
        let job_tiles: Vec<(u32, u32)> = tiles.iter().copied().collect();
        c.in_flight.extend(tile_neighbourhood(&c.composite, &job_tiles));
        let job = Job {
            project: s.doc.project.clone(),
            pixels: s.doc.pixels.clone(),
            baked: s.doc.baked.get(&ts).cloned().unwrap_or_default(),
            images: s.images.clone(),
            mesh: s.mesh.clone(),
            gbuf: c.gbuf.clone(),
            bounds: s.mesh.bounds(),
            patterns: c.patterns.clone(),
            dilation: c.dilation.clone(),
            base: c.composite.clone(),
            ts,
            tiles: job_tiles,
        };
        let gbuf = c.gbuf.clone();
        let cancel = Arc::new(AtomicBool::new(false));
        let (tx, rx) = std::sync::mpsc::channel();
        let flag = cancel.clone();
        let spawned = std::thread::Builder::new().name("texturecraft-composite".into()).spawn(move || match pool() {
            Some(p) => p.install(|| job.run(&flag, &tx)),
            None => job.run(&flag, &tx),
        });
        self.running = Some(Running { ts, gbuf, channels, job: tiles, cancel, rx });
        if let Err(e) = spawned {
            self.unavailable = true;
            self.cancel(s);
            s.log.push(format!("Background compositing unavailable ({e}): compositing in frame slices"));
        }
    }

    /// Apply what the running job delivered (waiting up to `wait` for a delivery); returns
    /// whether a composite changed.
    fn collect(&mut self, s: &mut Session, wait: Option<Duration>) -> bool {
        let Some(r) = &self.running else { return false };
        let mut tiles = Vec::new();
        let mut done = false;
        if let Some(w) = wait {
            match r.rx.recv_timeout(w) {
                Ok(t) => tiles.extend(t),
                Err(RecvTimeoutError::Timeout) => {}
                Err(RecvTimeoutError::Disconnected) => done = true,
            }
        }
        while !done {
            match r.rx.try_recv() {
                Ok(t) => tiles.extend(t),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => done = true,
            }
        }
        let ts = r.ts;
        let Some(c) = s.caches.get_mut(ts) else {
            self.running = None;
            return false;
        };
        // A new layout (resolution, UV tiles) or channel list: nothing of this job applies.
        let same = Arc::ptr_eq(&r.gbuf, &c.gbuf) && c.composite.channels.keys().copied().eq(r.channels.iter().copied());
        if !same {
            self.cancel(s);
            return false;
        }
        let mut applied = Vec::new();
        for ((tx, ty), data) in tiles {
            // Composited on this thread since (newer), or no longer wanted: drop it.
            if !c.in_flight.remove(&(tx, ty)) {
                continue;
            }
            for (ch, d) in data {
                if let Some(img) = c.composite.channels.get_mut(&ch) {
                    img.set_tile(tx, ty, d);
                }
            }
            applied.push((tx, ty));
        }
        let changed = !applied.is_empty();
        if changed {
            // Padded by the worker already.
            c.upload.extend(applied);
            c.revision = c.revision.wrapping_add(1);
        }
        if done {
            // Finished or failed (a cancelled job is no longer collected): what never arrived
            // is dirty again.
            let back = std::mem::take(&mut c.in_flight);
            self.failures = if back.is_empty() { 0 } else { self.failures + 1 };
            c.dirty.extend(back);
            self.running = None;
            if self.failures >= MAX_FAILURES && !self.unavailable {
                self.unavailable = true;
                s.log.push("Background composites keep stopping early: compositing in frame slices".into());
            }
        } else if self.running.as_ref().is_some_and(|r| {
            let mut left = r.job.intersection(&c.in_flight).peekable();
            left.peek().is_some() && left.all(|t| c.dirty.contains(t))
        }) {
            // Every tile left was edited again (a slider being dragged): start over on the new
            // state instead of finishing stale work.
            self.cancel(s);
        }
        changed
    }
}
