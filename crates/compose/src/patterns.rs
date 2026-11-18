//! A per-texture-set cache of procedural pattern values. A pattern's value at a texel depends
//! only on the pattern, its projection, the resolution it is drawn at, the images it reads and
//! where the texel is on the mesh, never on the stack around it or the baked maps; yet every
//! bake or stack edit recomposites the layers that use it, and patterns are most of a smart
//! material's cost. The cache keeps, per pattern and per band of rows of a tile, the values
//! computed so far (exact `f32`, so cached and computed composites are identical), within a
//! byte budget, least recently used first out.
//!
//! The owner (the session's per-set cache) drops it whenever the texel layout changes; the
//! key also holds the layout's size and the mesh bounds, as a guard. The images a pattern
//! reads are in the key by identity (which allocation the name points to), so a resource
//! replaced under the same name is another key.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, Weak};

use texturecraft_core::{Procedural, Projection};

use crate::store::{ImageRes, ImageStore};

/// Default byte budget of a [`PatternCache`] (a 2K texture set with a dozen patterns needs
/// about 200 MB). A session shares it between its texture sets.
pub const PATTERN_BUDGET: usize = 768 << 20;

/// Lookups recorded in the use order beyond twice the bands held before it is rebuilt: every
/// lookup appends one, so this bounds the order's garbage while keeping rebuilds rare.
const ORDER_SLACK: usize = 4096;

/// Values of one band of one tile, NaN where not computed yet.
type Values = Arc<[f32]>;

/// The images a key names, held weakly: while an entry lives, the allocation behind each
/// address in its key cannot be reused by another image, so an address is never served for an
/// image it did not belong to (the pixels themselves are freed with the resource).
type Pins = Arc<[Weak<ImageRes>]>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    /// Layout guard, pattern, projection, resolution and image identities (`Debug` prints
    /// floats losslessly).
    pattern: Arc<str>,
    tile: (u32, u32),
    band: u32,
}

#[derive(Debug)]
struct Entry {
    values: Values,
    _pins: Pins,
    /// Last use (see `Inner::order`).
    stamp: u64,
    /// Composite pass of the last use: entries of the running pass are never evicted.
    pass: u64,
}

#[derive(Debug, Default)]
struct Inner {
    entries: HashMap<Key, Entry>,
    /// Uses, oldest first; entries a later use superseded are skipped.
    order: VecDeque<(Key, u64)>,
    clock: u64,
    pass: u64,
    bytes: usize,
}

impl Inner {
    /// Drop the least recently used entries until `need` more bytes fit in `budget`, sparing
    /// the running pass's when `spare_pass`. False when they still do not fit.
    fn make_room(&mut self, need: usize, budget: usize, spare_pass: bool) -> bool {
        while self.bytes + need > budget {
            let Some((k, stamp)) = self.order.pop_front() else { break };
            match self.entries.get(&k) {
                Some(e) if e.stamp == stamp && spare_pass && e.pass == self.pass => {
                    // Everything older is gone: the running pass alone fills the budget.
                    self.order.push_front((k, stamp));
                    return false;
                }
                Some(e) if e.stamp == stamp => {
                    let freed = entry_bytes(&k, &e.values, &e._pins);
                    self.entries.remove(&k);
                    self.bytes = self.bytes.saturating_sub(freed);
                }
                _ => {}
            }
        }
        self.bytes + need <= budget
    }
}

/// See the module docs. One per texture set; shared by the threads of a composite.
#[derive(Debug)]
pub struct PatternCache {
    inner: Mutex<Inner>,
    /// Byte budget; beyond it the least recently used bands are dropped.
    budget: AtomicUsize,
}

impl Default for PatternCache {
    fn default() -> Self {
        PatternCache { inner: Mutex::new(Inner::default()), budget: AtomicUsize::new(PATTERN_BUDGET) }
    }
}

/// What an entry holds: its values, its key's text (a few hundred bytes of pattern and
/// projection parameters, next to 4 KB of values per band) and its image pins.
fn entry_bytes(key: &Key, v: &Values, pins: &Pins) -> usize {
    v.len() * std::mem::size_of::<f32>() + key.pattern.len() + pins.len() * std::mem::size_of::<Weak<ImageRes>>()
}

/// The key text of `pattern` through `proj` drawn at `pixels` under layout guard `salt`, with
/// the identity of each image the pattern reads from `images` (pinned into `pins`).
fn key_text(salt: &str, pattern: &Procedural, proj: &Projection, pixels: Option<u32>, images: Option<&ImageStore>, pins: &mut Vec<Weak<ImageRes>>) -> String {
    let mut text = format!("{salt}|{pattern:?}|{proj:?}|{pixels:?}");
    for name in crate::pattern::image_names(pattern) {
        match images.and_then(|images| images.get(name)) {
            Some(image) => {
                text.push_str(&format!("|{name:?}@{:p}", Arc::as_ptr(image)));
                pins.push(Arc::downgrade(image));
            }
            None => text.push_str(&format!("|{name:?}@-")),
        }
    }
    text
}

impl PatternCache {
    /// An empty cache holding at most `budget` bytes.
    pub fn with_budget(budget: usize) -> Self {
        PatternCache { budget: AtomicUsize::new(budget), ..Self::default() }
    }

    /// The byte budget.
    pub fn budget(&self) -> usize {
        self.budget.load(Ordering::Relaxed)
    }

    /// Change the byte budget (a session shares one budget between its texture sets). What is
    /// held beyond a smaller budget goes at once, least recently used first: a set that is not
    /// composited again would otherwise keep its bytes for good.
    pub fn set_budget(&self, budget: usize) {
        self.budget.store(budget, Ordering::Relaxed);
        self.lock().make_room(0, budget, false);
    }

    fn lock(&self) -> MutexGuard<'_, Inner> {
        // A panic elsewhere cannot leave the map half-updated in a way that matters: recover.
        self.inner.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    /// Start a composite pass: what it uses stays until the next one starts.
    pub fn begin_pass(&self) {
        self.lock().pass += 1;
    }

    pub fn clear(&self) {
        let mut g = self.lock();
        g.entries.clear();
        g.order.clear();
        g.bytes = 0;
    }

    pub fn bytes(&self) -> usize {
        self.lock().bytes
    }

    /// Bands held.
    pub fn len(&self) -> usize {
        self.lock().entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn get(&self, key: &Key) -> Option<Values> {
        let mut g = self.lock();
        g.clock += 1;
        let (stamp, pass) = (g.clock, g.pass);
        let values = g.entries.get_mut(key).map(|e| {
            e.stamp = stamp;
            e.pass = pass;
            e.values.clone()
        })?;
        g.order.push_back((key.clone(), stamp));
        compact(&mut g);
        Some(values)
    }

    fn insert(&self, key: Key, values: Values, pins: Pins) {
        let need = entry_bytes(&key, &values, &pins);
        let budget = self.budget();
        if need > budget {
            return;
        }
        let mut g = self.lock();
        if let Some(old) = g.entries.remove(&key) {
            g.bytes = g.bytes.saturating_sub(entry_bytes(&key, &old.values, &old._pins));
        }
        if !g.make_room(need, budget, true) {
            return;
        }
        g.clock += 1;
        let (stamp, pass) = (g.clock, g.pass);
        g.bytes += need;
        g.order.push_back((key.clone(), stamp));
        g.entries.insert(key, Entry { values, _pins: pins, stamp, pass });
        compact(&mut g);
    }
}

/// Rebuild the use order when superseded uses pile up (every lookup appends one).
fn compact(g: &mut Inner) {
    if g.order.len() <= 2 * g.entries.len() + ORDER_SLACK {
        return;
    }
    let mut live: Vec<(Key, u64)> = g.entries.iter().map(|(k, e)| (k.clone(), e.stamp)).collect();
    live.sort_by_key(|(_, s)| *s);
    g.order = live.into();
}

/// The patterns one band of a composite has looked up: per pattern, the cached values and the
/// values computed in this band, published to the cache when the band is done.
#[derive(Default)]
pub(crate) struct BandPatterns<'c> {
    cache: Option<&'c PatternCache>,
    images: Option<&'c ImageStore>,
    /// Layout guard of the key. An `Option`, so the default (a `Texel`'s, moved in and out per
    /// texel) allocates nothing: an empty `Arc<str>` would, twice per texel.
    salt: Option<Arc<str>>,
    tile: (u32, u32),
    band: u32,
    /// Tile offset of the band's first texel, and its texel count.
    start: usize,
    len: usize,
    slots: Vec<Slot>,
}

struct Slot {
    /// Addresses of the pattern and projection (stable during a composite) and the resolution
    /// the pattern is drawn at (it may differ between UV tiles).
    id: (usize, usize, Option<u32>),
    key: Key,
    pins: Pins,
    cached: Option<Values>,
    fresh: Option<Vec<f32>>,
}

impl<'c> BandPatterns<'c> {
    /// The patterns of band `band` of tile `tile` (texels `start..start + len` of the tile),
    /// keyed under layout guard `salt`; `images` are the images patterns read in this composite.
    pub(crate) fn new(cache: Option<&'c PatternCache>, images: &'c ImageStore, salt: Arc<str>, tile: (u32, u32), band: u32, start: usize, len: usize) -> Self {
        BandPatterns { cache, images: Some(images), salt: cache.map(|_| salt), tile, band, start, len, slots: Vec::new() }
    }

    fn slot(&mut self, pattern: &Procedural, proj: &Projection, pixels: Option<u32>) -> Option<&mut Slot> {
        let cache = self.cache?;
        let id = (pattern as *const Procedural as usize, proj as *const Projection as usize, pixels);
        let i = match self.slots.iter().position(|s| s.id == id) {
            Some(i) => i,
            None => {
                let mut pins = Vec::new();
                let text = key_text(self.salt.as_deref().unwrap_or_default(), pattern, proj, pixels, self.images, &mut pins);
                let key = Key { pattern: text.into(), tile: self.tile, band: self.band };
                let cached = cache.get(&key).filter(|v| v.len() == self.len);
                self.slots.push(Slot { id, key, pins: pins.into(), cached, fresh: None });
                self.slots.len() - 1
            }
        };
        self.slots.get_mut(i)
    }

    /// The cached value of `pattern` through `proj`, drawn at `pixels`, at tile offset `o`.
    pub(crate) fn get(&mut self, pattern: &Procedural, proj: &Projection, pixels: Option<u32>, o: usize) -> Option<f32> {
        let i = o.checked_sub(self.start)?;
        let s = self.slot(pattern, proj, pixels)?;
        let v = match &s.fresh {
            Some(f) => f.get(i).copied(),
            None => s.cached.as_ref().and_then(|c| c.get(i).copied()),
        }?;
        (!v.is_nan()).then_some(v)
    }

    /// Record a value computed at tile offset `o` (after a [`BandPatterns::get`] miss).
    pub(crate) fn put(&mut self, pattern: &Procedural, proj: &Projection, pixels: Option<u32>, o: usize, v: f32) {
        let Some(i) = o.checked_sub(self.start) else { return };
        let len = self.len;
        let Some(s) = self.slot(pattern, proj, pixels) else { return };
        let fresh = s.fresh.get_or_insert_with(|| s.cached.as_ref().map_or_else(|| vec![f32::NAN; len], |c| c.to_vec()));
        if let Some(slot) = fresh.get_mut(i) {
            *slot = v;
        }
    }

    /// Hand the values computed in this band to the cache.
    pub(crate) fn publish(self) {
        let Some(cache) = self.cache else { return };
        for s in self.slots {
            if let Some(f) = s.fresh {
                cache.insert(s.key, f.into(), s.pins);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use texturecraft_core::{ProceduralKind, TileGenerator, TileShape};

    static NO_IMAGES: std::sync::LazyLock<ImageStore> = std::sync::LazyLock::new(ImageStore::new);

    fn band<'c>(cache: &'c PatternCache, tile: (u32, u32)) -> BandPatterns<'c> {
        band_with(cache, tile, &NO_IMAGES)
    }

    fn band_with<'c>(cache: &'c PatternCache, tile: (u32, u32), images: &'c ImageStore) -> BandPatterns<'c> {
        BandPatterns::new(Some(cache), images, "guard".into(), tile, 0, 100, 8)
    }

    fn entry(a: &Procedural, proj: &Projection) -> usize {
        8 * 4 + key_text("guard", a, proj, None, None, &mut Vec::new()).len()
    }

    #[test]
    fn values_come_back_per_pattern_and_projection() {
        let cache = PatternCache::default();
        let (a, b) = (Procedural::new(ProceduralKind::Clouds), Procedural::new(ProceduralKind::Grunge));
        let proj = Projection::default();
        let mut p = band(&cache, (0, 0));
        assert_eq!(p.get(&a, &proj, None, 103), None);
        p.put(&a, &proj, None, 103, 0.25);
        p.put(&b, &proj, None, 104, 0.75);
        assert_eq!(p.get(&a, &proj, None, 103), Some(0.25));
        p.publish();
        // Another band of the same tile, a fresh composite: the same values, not the others.
        let mut q = band(&cache, (0, 0));
        let a2 = a.clone();
        assert_eq!(q.get(&a2, &proj, None, 103), Some(0.25));
        assert_eq!(q.get(&a2, &proj, None, 104), None);
        assert_eq!(q.get(&b, &proj, None, 104), Some(0.75));
        // A changed parameter is another pattern, and so is another resolution.
        let edited = Procedural { scale: 3.0, ..a.clone() };
        assert_eq!(q.get(&edited, &proj, None, 103), None);
        assert_eq!(q.get(&a2, &proj, Some(256), 103), None);
        assert_eq!(band(&cache, (1, 0)).get(&a, &proj, None, 103), None);
        // Offsets outside the band are never served.
        assert_eq!(q.get(&a2, &proj, None, 99), None);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn an_image_replaced_under_the_same_name_is_another_pattern() {
        let image = |v: f32| Arc::new(ImageRes { width: 1, height: 1, rgba: vec![[v, v, v, 1.0]], srgb: false });
        let tile = TileGenerator { shape: TileShape::Image, inputs: vec!["stone".into()], background_image: Some("ground".into()), ..TileGenerator::default() };
        let pattern = Procedural { tile: Some(Box::new(tile)), ..Procedural::new(ProceduralKind::TileGenerator) };
        let proj = Projection::default();
        let cache = PatternCache::default();
        let first: ImageStore = [("stone".to_string(), image(0.2))].into_iter().collect();
        let mut p = band_with(&cache, (0, 0), &first);
        p.put(&pattern, &proj, None, 100, 0.2);
        p.publish();
        assert_eq!(band_with(&cache, (0, 0), &first).get(&pattern, &proj, None, 100), Some(0.2));

        // The same names, other images (the first still alive, so its address is not reused)
        let replaced: ImageStore = [("stone".to_string(), image(0.9))].into_iter().collect();
        assert_eq!(band_with(&cache, (0, 0), &replaced).get(&pattern, &proj, None, 100), None);
        let added: ImageStore = [("stone".to_string(), first["stone"].clone()), ("ground".to_string(), image(0.5))].into_iter().collect();
        assert_eq!(band_with(&cache, (0, 0), &added).get(&pattern, &proj, None, 100), None, "a background image appeared");
        assert_eq!(band(&cache, (0, 0)).get(&pattern, &proj, None, 100), None, "the image went away");

        // The entry pins the first image's allocation after the resource is dropped, so a new
        // image can never take its address and be served its values.
        let address = Arc::as_ptr(&first["stone"]) as usize;
        drop(first);
        drop(added);
        for _ in 0..64 {
            let fresh: ImageStore = [("stone".to_string(), image(0.9))].into_iter().collect();
            assert_ne!(Arc::as_ptr(&fresh["stone"]) as usize, address, "a new image took a cached image's address");
            assert_eq!(band_with(&cache, (0, 0), &fresh).get(&pattern, &proj, None, 100), None);
        }
    }

    #[test]
    fn the_budget_drops_least_recently_used_bands_but_never_the_running_pass() {
        let a = Procedural::new(ProceduralKind::Clouds);
        let proj = Projection::default();
        // Three bands of 8 values, each with its key text.
        let entry = entry(&a, &proj);
        let cache = PatternCache::with_budget(3 * entry);
        let fill = |tile: u32| {
            let mut p = band(&cache, (tile, 0));
            p.put(&a, &proj, None, 100, tile as f32);
            p.publish();
        };
        cache.begin_pass();
        for t in 0..3 {
            fill(t);
        }
        assert_eq!(cache.bytes(), 3 * entry);
        // Same pass: the budget is full of this pass's bands, the fourth is not kept.
        fill(3);
        assert_eq!((cache.len(), band(&cache, (3, 0)).get(&a, &proj, None, 100)), (3, None));
        // Next pass: tile 0 is used, so tile 1 (least recently used) makes room for tile 3.
        cache.begin_pass();
        assert_eq!(band(&cache, (0, 0)).get(&a, &proj, None, 100), Some(0.0));
        cache.begin_pass();
        fill(3);
        assert_eq!(band(&cache, (1, 0)).get(&a, &proj, None, 100), None);
        assert_eq!(band(&cache, (0, 0)).get(&a, &proj, None, 100), Some(0.0));
        assert_eq!(band(&cache, (3, 0)).get(&a, &proj, None, 100), Some(3.0));
        assert!(cache.bytes() <= cache.budget());
    }

    #[test]
    fn a_smaller_budget_applies_to_the_next_bands() {
        let a = Procedural::new(ProceduralKind::Clouds);
        let proj = Projection::default();
        let entry = entry(&a, &proj);
        let cache = PatternCache::with_budget(10 * entry);
        let fill = |tile: u32| {
            let mut p = band(&cache, (tile, 0));
            p.put(&a, &proj, None, 100, 1.0);
            p.publish();
        };
        cache.begin_pass();
        for t in 0..4 {
            fill(t);
        }
        assert_eq!(cache.len(), 4);
        cache.set_budget(2 * entry);
        cache.begin_pass();
        fill(9);
        assert!(cache.bytes() <= 2 * entry, "{} > {}", cache.bytes(), 2 * entry);
        assert_eq!(cache.budget(), 2 * entry);
    }

    #[test]
    fn shrinking_the_budget_frees_bands_at_once() {
        let a = Procedural::new(ProceduralKind::Clouds);
        let proj = Projection::default();
        let entry = entry(&a, &proj);
        let cache = PatternCache::with_budget(10 * entry);
        cache.begin_pass();
        for tile in 0..4 {
            let mut p = band(&cache, (tile, 0));
            p.put(&a, &proj, None, 100, tile as f32);
            p.publish();
        }
        assert_eq!(cache.bytes(), 4 * entry);
        // Tile 0 is used last, so tile 1 goes first.
        cache.begin_pass();
        assert_eq!(band(&cache, (0, 0)).get(&a, &proj, None, 100), Some(0.0));
        cache.set_budget(2 * entry);
        assert!(cache.bytes() <= 2 * entry, "{} > {}", cache.bytes(), 2 * entry);
        assert_eq!(band(&cache, (1, 0)).get(&a, &proj, None, 100), None);
        assert_eq!(band(&cache, (0, 0)).get(&a, &proj, None, 100), Some(0.0));
        cache.set_budget(0);
        assert_eq!((cache.bytes(), cache.len()), (0, 0));
        // A zero budget keeps nothing and does not fail.
        let mut p = band(&cache, (5, 0));
        p.put(&a, &proj, None, 100, 1.0);
        p.publish();
        assert_eq!((cache.bytes(), cache.len()), (0, 0));
    }

    #[test]
    fn without_a_cache_nothing_is_kept() {
        let a = Procedural::new(ProceduralKind::Clouds);
        let proj = Projection::default();
        let mut p = BandPatterns::default();
        p.put(&a, &proj, None, 0, 0.5);
        assert_eq!(p.get(&a, &proj, None, 0), None);
        p.publish();
    }
}
