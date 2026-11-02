//! A per-texture-set cache of procedural pattern values. A pattern's value at a texel depends
//! only on the pattern, its projection and where the texel is on the mesh, never on the stack
//! around it or the baked maps; yet every bake or stack edit recomposites the layers that use
//! it, and patterns are most of a smart material's cost. The cache keeps, per pattern and per
//! band of rows of a tile, the values computed so far (exact `f32`, so cached and computed
//! composites are identical), within a byte budget, least recently used first out.
//!
//! The owner (the session's per-set cache) drops it whenever the texel layout changes; the
//! key also holds the layout's size and the mesh bounds, as a guard.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};

use texturecraft_core::{Procedural, Projection};

/// Default byte budget of a [`PatternCache`] (a 2K texture set with a dozen patterns needs
/// about 200 MB).
pub const PATTERN_BUDGET: usize = 768 << 20;

/// Values of one band of one tile, NaN where not computed yet.
type Values = Arc<[f32]>;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct Key {
    /// Layout guard, pattern and projection (`Debug` prints floats losslessly).
    pattern: Arc<str>,
    tile: (u32, u32),
    band: u32,
}

#[derive(Debug)]
struct Entry {
    values: Values,
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

/// See the module docs. One per texture set; shared by the threads of a composite.
#[derive(Debug)]
pub struct PatternCache {
    inner: Mutex<Inner>,
    /// Byte budget; beyond it the least recently used bands are dropped.
    pub budget: usize,
}

impl Default for PatternCache {
    fn default() -> Self {
        PatternCache { inner: Mutex::new(Inner::default()), budget: PATTERN_BUDGET }
    }
}

fn values_bytes(v: &Values) -> usize {
    v.len() * std::mem::size_of::<f32>()
}

impl PatternCache {
    /// An empty cache holding at most `budget` bytes.
    pub fn with_budget(budget: usize) -> Self {
        PatternCache { budget, ..Self::default() }
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

    fn insert(&self, key: Key, values: Values) {
        let need = values_bytes(&values);
        if need > self.budget {
            return;
        }
        let mut g = self.lock();
        if let Some(old) = g.entries.remove(&key) {
            g.bytes -= values_bytes(&old.values);
        }
        while g.bytes + need > self.budget {
            let Some((k, stamp)) = g.order.pop_front() else { break };
            let pass = g.pass;
            match g.entries.get(&k) {
                Some(e) if e.stamp == stamp && e.pass == pass => {
                    // Everything older is gone: the running pass alone fills the budget.
                    g.order.push_front((k, stamp));
                    return;
                }
                Some(e) if e.stamp == stamp => {
                    let freed = values_bytes(&e.values);
                    g.entries.remove(&k);
                    g.bytes -= freed;
                }
                _ => {}
            }
        }
        if g.bytes + need > self.budget {
            return;
        }
        g.clock += 1;
        let (stamp, pass) = (g.clock, g.pass);
        g.bytes += need;
        g.order.push_back((key.clone(), stamp));
        g.entries.insert(key, Entry { values, stamp, pass });
        compact(&mut g);
    }
}

/// Rebuild the use order when superseded uses pile up (every lookup appends one).
fn compact(g: &mut Inner) {
    if g.order.len() <= 2 * g.entries.len() + 4096 {
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
    /// Layout guard of the key.
    salt: Arc<str>,
    tile: (u32, u32),
    band: u32,
    /// Tile offset of the band's first texel, and its texel count.
    start: usize,
    len: usize,
    slots: Vec<Slot>,
}

struct Slot {
    /// Addresses of the pattern and projection (stable during a composite).
    id: (usize, usize),
    key: Key,
    cached: Option<Values>,
    fresh: Option<Vec<f32>>,
}

impl<'c> BandPatterns<'c> {
    pub(crate) fn new(cache: Option<&'c PatternCache>, salt: Arc<str>, tile: (u32, u32), band: u32, start: usize, len: usize) -> Self {
        BandPatterns { cache, salt, tile, band, start, len, slots: Vec::new() }
    }

    fn slot(&mut self, pattern: &Procedural, proj: &Projection) -> Option<&mut Slot> {
        let cache = self.cache?;
        let id = (pattern as *const Procedural as usize, proj as *const Projection as usize);
        let i = match self.slots.iter().position(|s| s.id == id) {
            Some(i) => i,
            None => {
                let key = Key { pattern: format!("{}|{pattern:?}|{proj:?}", self.salt).into(), tile: self.tile, band: self.band };
                let cached = cache.get(&key).filter(|v| v.len() == self.len);
                self.slots.push(Slot { id, key, cached, fresh: None });
                self.slots.len() - 1
            }
        };
        self.slots.get_mut(i)
    }

    /// The cached value of `pattern` through `proj` at tile offset `o`.
    pub(crate) fn get(&mut self, pattern: &Procedural, proj: &Projection, o: usize) -> Option<f32> {
        let i = o.checked_sub(self.start)?;
        let s = self.slot(pattern, proj)?;
        let v = match &s.fresh {
            Some(f) => f.get(i).copied(),
            None => s.cached.as_ref().and_then(|c| c.get(i).copied()),
        }?;
        (!v.is_nan()).then_some(v)
    }

    /// Record a value computed at tile offset `o` (after a [`BandPatterns::get`] miss).
    pub(crate) fn put(&mut self, pattern: &Procedural, proj: &Projection, o: usize, v: f32) {
        let Some(i) = o.checked_sub(self.start) else { return };
        let len = self.len;
        let Some(s) = self.slot(pattern, proj) else { return };
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
                cache.insert(s.key, f.into());
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use texturecraft_core::ProceduralKind;

    fn band<'c>(cache: &'c PatternCache, tile: (u32, u32)) -> BandPatterns<'c> {
        BandPatterns::new(Some(cache), "guard".into(), tile, 0, 100, 8)
    }

    #[test]
    fn values_come_back_per_pattern_and_projection() {
        let cache = PatternCache::default();
        let (a, b) = (Procedural::new(ProceduralKind::Clouds), Procedural::new(ProceduralKind::Grunge));
        let proj = Projection::default();
        let mut p = band(&cache, (0, 0));
        assert_eq!(p.get(&a, &proj, 103), None);
        p.put(&a, &proj, 103, 0.25);
        p.put(&b, &proj, 104, 0.75);
        assert_eq!(p.get(&a, &proj, 103), Some(0.25));
        p.publish();
        // Another band of the same tile, a fresh composite: the same values, not the others.
        let mut q = band(&cache, (0, 0));
        let a2 = a.clone();
        assert_eq!(q.get(&a2, &proj, 103), Some(0.25));
        assert_eq!(q.get(&a2, &proj, 104), None);
        assert_eq!(q.get(&b, &proj, 104), Some(0.75));
        // A changed parameter is another pattern.
        let edited = Procedural { scale: 3.0, ..a.clone() };
        assert_eq!(q.get(&edited, &proj, 103), None);
        assert_eq!(band(&cache, (1, 0)).get(&a, &proj, 103), None);
        // Offsets outside the band are never served.
        assert_eq!(q.get(&a2, &proj, 99), None);
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn the_budget_drops_least_recently_used_bands_but_never_the_running_pass() {
        let cache = PatternCache::with_budget(3 * 8 * 4);
        let a = Procedural::new(ProceduralKind::Clouds);
        let proj = Projection::default();
        let fill = |tile: u32| {
            let mut p = band(&cache, (tile, 0));
            p.put(&a, &proj, 100, tile as f32);
            p.publish();
        };
        cache.begin_pass();
        for t in 0..3 {
            fill(t);
        }
        assert_eq!(cache.bytes(), 3 * 8 * 4);
        // Same pass: the budget is full of this pass's bands, the fourth is not kept.
        fill(3);
        assert_eq!((cache.len(), band(&cache, (3, 0)).get(&a, &proj, 100)), (3, None));
        // Next pass: tile 0 is used, so tile 1 (least recently used) makes room for tile 3.
        cache.begin_pass();
        assert_eq!(band(&cache, (0, 0)).get(&a, &proj, 100), Some(0.0));
        cache.begin_pass();
        fill(3);
        assert_eq!(band(&cache, (1, 0)).get(&a, &proj, 100), None);
        assert_eq!(band(&cache, (0, 0)).get(&a, &proj, 100), Some(0.0));
        assert_eq!(band(&cache, (3, 0)).get(&a, &proj, 100), Some(3.0));
        assert!(cache.bytes() <= cache.budget);
    }

    #[test]
    fn without_a_cache_nothing_is_kept() {
        let a = Procedural::new(ProceduralKind::Clouds);
        let proj = Projection::default();
        let mut p = BandPatterns::default();
        p.put(&a, &proj, 0, 0.5);
        assert_eq!(p.get(&a, &proj, 0), None);
        p.publish();
    }
}
