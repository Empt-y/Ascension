//! Nature scatter: trees, shrubs, plants, grass, rocks and debris placed on the terrain by biome,
//! streamed in around the camera.
//!
//! Placement is a pure function of the seed: each terrain column (32 m square) and kind gets a
//! jittered grid of candidates, sized from that kind's highest density, and each candidate is kept
//! with the probability its biome's density gives (trees also thin out towards the edges of woods).
//! A low-priority worker generates (column, kind) pairs nearest first, each kind out to its own draw
//! distance. On the main thread instances are dropped where the ground has been dug or filled or a
//! building stands, then handed to the renderer as one batch per (column, kind). Trunks and big
//! rocks become colliders.
//!
//! Ore nodes (see `ore_nodes`) stream as two more kinds, placed by their own rules. They can be
//! mined away; the ones that have been are kept in a set and filtered out like dug instances.

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicI64, Ordering};
use std::thread;

use asc_build::Structures;
use asc_render::{ModelId, PartInstance, Renderer, ScatterBatch};
use asc_world::{Biome, World, WorldGen};
use crossbeam_channel::{Receiver, Sender};
use glam::{IVec2, Vec2, Vec3};

use crate::assets::{Collide, NatureModels, OreModels};

/// Side of a scatter column in metres (the terrain chunk size).
pub const COLUMN: f32 = 32.0;
/// Keep generated columns this much beyond a kind's draw distance before dropping them.
const KEEP_MARGIN: f32 = 2.0 * COLUMN;

fn biome_index(b: Biome) -> usize {
    Biome::ALL.iter().position(|x| *x == b).unwrap_or(0)
}

#[derive(Clone, Copy, Debug)]
struct Model {
    id: ModelId,
    /// Collider at unit scale: trunk radius, or rock sphere radius and centre height.
    radius: f32,
    center_y: f32,
}

/// Which ore nodes a kind places.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    Surface,
    Deep,
}

#[derive(Clone, Debug)]
struct Kind {
    name: String,
    distance: f32,
    fade: f32,
    scale: [f32; 2],
    sway: f32,
    max_slope: f32,
    shadow_distance: f32,
    collide: Collide,
    /// Side of a candidate cell: one candidate per cell at the kind's highest density.
    cell: f32,
    density: [f32; Biome::ALL.len()],
    models: [Vec<Model>; Biome::ALL.len()],
    /// Ore node kinds: which nodes, and each ore's model.
    nodes: Option<(NodeKind, Vec<Model>)>,
}

fn rock_collider(min: Vec3, max: Vec3) -> (f32, f32) {
    let half = (max - min) * 0.5;
    (0.85 * half.x.min(half.z).min(half.y * 1.3), (max.y + min.y) * 0.5)
}

/// Everything the generator needs, shared with the worker.
pub struct Config {
    seed: u64,
    kinds: Vec<Kind>,
}

impl Config {
    pub fn new(seed: i32, nature: &NatureModels, ores: Option<&OreModels>) -> Self {
        // Debugging aid: ASC_SCATTER_ONLY=tree,rock scatters only those kinds.
        let only: Option<Vec<String>> =
            std::env::var("ASC_SCATTER_ONLY").ok().map(|v| v.split(',').map(str::to_owned).collect());
        let kinds = nature
            .kinds
            .iter()
            .filter(|(name, _)| only.as_ref().is_none_or(|o| o.contains(name)))
            .map(|(name, spec)| {
                let mut density = [0.0; Biome::ALL.len()];
                let mut models: [Vec<Model>; Biome::ALL.len()] = Default::default();
                for (i, biome) in Biome::ALL.into_iter().enumerate() {
                    density[i] =
                        nature.density.get(biome.name()).and_then(|d| d.get(name)).copied().unwrap_or(0.0);
                    for m in &nature.models {
                        if m.spec.kind == *name && m.spec.biomes.iter().any(|b| b == biome.name()) {
                            let (radius, center_y) = match spec.collide {
                                Collide::Trunk => (0.3, 0.0),
                                Collide::Rock => rock_collider(m.min, m.max),
                                Collide::None => (0.0, 0.0),
                            };
                            models[i].push(Model { id: m.id, radius, center_y });
                        }
                    }
                }
                let max = density.iter().copied().fold(0.0f32, f32::max);
                Kind {
                    name: name.clone(),
                    distance: spec.distance,
                    fade: spec.fade,
                    scale: spec.scale,
                    sway: spec.sway,
                    max_slope: spec.max_slope,
                    shadow_distance: spec.shadow_distance,
                    collide: spec.collide,
                    cell: if max > 0.0 { 1.0 / max.sqrt() } else { 0.0 },
                    density,
                    models,
                    nodes: None,
                }
            })
            .collect::<Vec<_>>();
        let mut kinds = kinds;
        if let Some(ores) = ores {
            let node_kind = |name: &str, node: NodeKind, distance: f32, shadow: f32| {
                let list = match node {
                    NodeKind::Surface => &ores.surface,
                    NodeKind::Deep => &ores.deep,
                };
                let models = list
                    .iter()
                    .map(|m| {
                        let (radius, center_y) = rock_collider(m.min, m.max);
                        Model { id: m.id, radius, center_y }
                    })
                    .collect();
                Kind {
                    name: name.into(),
                    distance,
                    fade: distance * 0.2,
                    scale: [1.0, 1.0],
                    sway: 0.0,
                    max_slope: f32::MAX,
                    shadow_distance: shadow,
                    collide: Collide::Rock,
                    cell: 1.0,
                    density: [0.0; Biome::ALL.len()],
                    models: Default::default(),
                    nodes: Some((node, models)),
                }
            };
            kinds.push(node_kind("ore_surface", NodeKind::Surface, 150.0, 50.0));
            // Underground: no sun reaches them, so no shadows.
            kinds.push(node_kind("ore_deep", NodeKind::Deep, 80.0, 0.0));
        }
        Self { seed: seed as u32 as u64, kinds }
    }

    /// Index of the kind called `name`.
    #[cfg(test)]
    pub fn kind(&self, name: &str) -> Option<usize> {
        self.kinds.iter().position(|k| k.name == name)
    }
}

/// One placed instance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Placed {
    pub model: ModelId,
    pub position: Vec3,
    pub yaw: f32,
    pub scale: f32,
    /// Collider radius and centre height (rocks), already scaled.
    collider_radius: f32,
    collider_center: f32,
    /// Ore nodes: which ore.
    pub ore: Option<u16>,
}

/// A node's identity: its position on a quarter-metre grid (placement is deterministic, so this
/// names the same node in every session).
pub type NodeId = glam::IVec3;

pub fn node_id(p: Vec3) -> NodeId {
    (p * 4.0).round().as_ivec3()
}

/// The ore node a crosshair is on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeHit {
    pub id: NodeId,
    pub ore: usize,
    pub kind: NodeKind,
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> f32 {
        self.0 = splitmix(self.0);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// The instances of `kind` in `column`. Pure: the same inputs always give the same instances.
pub fn generate(config: &Config, worldgen: &WorldGen, column: IVec2, kind: usize) -> Vec<Placed> {
    let k = &config.kinds[kind];
    if let Some((node, models)) = &k.nodes {
        let nodes = match node {
            NodeKind::Surface => crate::ore_nodes::surface(config.seed, worldgen, column),
            NodeKind::Deep => crate::ore_nodes::deep(config.seed, worldgen, column),
        };
        return nodes
            .into_iter()
            .filter_map(|n| {
                let m = models.get(n.ore)?;
                Some(Placed {
                    model: m.id,
                    position: n.position,
                    yaw: n.yaw,
                    scale: n.scale,
                    collider_radius: m.radius * n.scale,
                    collider_center: m.center_y * n.scale,
                    ore: Some(n.ore as u16),
                })
            })
            .collect();
    }
    let mut out = Vec::new();
    if k.cell <= 0.0 {
        return out;
    }
    let n = (COLUMN / k.cell).ceil().max(1.0) as i32;
    let cell = COLUMN / n as f32;
    let origin = column.as_vec2() * COLUMN;
    let woodland = |b: Biome| matches!(b, Biome::Forest | Biome::Taiga | Biome::Jungle | Biome::Swamp);
    for j in 0..n {
        for i in 0..n {
            let gx = (column.x * n + i) as i64 as u64;
            let gz = (column.y * n + j) as i64 as u64;
            let mut r = Rng(splitmix(config.seed ^ splitmix(gx ^ splitmix(gz ^ ((kind as u64) << 56)))));
            let x = origin.x + (i as f32 + r.next()) * cell;
            let z = origin.y + (j as f32 + r.next()) * cell;
            let roll = r.next();
            let pick = r.next();
            let yaw = r.next() * std::f32::consts::TAU;
            let scale = k.scale[0] + (k.scale[1] - k.scale[0]) * r.next();
            let g = worldgen.ground(x, z);
            if g.underwater || g.slope > k.max_slope {
                continue;
            }
            let b = biome_index(g.biome);
            let mut density = k.density[b];
            if k.name == "tree" && woodland(g.biome) {
                density *= 0.15 + 0.85 * g.forest;
            } else if k.name == "grass" && woodland(g.biome) {
                // Less grass in the shade of woods.
                density *= 1.0 - 0.6 * g.forest;
            }
            if roll >= density * cell * cell {
                continue;
            }
            let models = &k.models[b];
            if models.is_empty() {
                continue;
            }
            let m = models[((pick * models.len() as f32) as usize).min(models.len() - 1)];
            // The height field is the surface except where caves break through it (cave mouths):
            // check the real density, and leave out anything that would float or be buried.
            let base = Vec3::new(x, g.height, z);
            if worldgen.sample(base + Vec3::Y * 0.25).0 <= 0.0
                || worldgen.sample(base - Vec3::Y * 0.25).0 >= 0.0
            {
                continue;
            }
            out.push(Placed {
                model: m.id,
                position: Vec3::new(x, g.height, z),
                yaw,
                scale,
                collider_radius: m.radius * scale,
                collider_center: m.center_y * scale,
                ore: None,
            });
        }
    }
    out
}

type Key = (IVec2, usize);

fn batch_key((column, kind): Key) -> u64 {
    ((column.x as u32 as u64 & 0x0FFF_FFFF) << 36)
        | ((column.y as u32 as u64 & 0x0FFF_FFFF) << 8)
        | kind as u64
}

/// Squared horizontal distance from `p` to the nearest point of `column`.
fn column_distance(column: IVec2, p: Vec2) -> f32 {
    let lo = column.as_vec2() * COLUMN;
    let nearest = p.clamp(lo, lo + COLUMN);
    nearest.distance(p)
}

/// A collider: a capsule from `a` to `b` (a sphere when they're equal).
#[derive(Clone, Copy, Debug)]
struct Collider {
    a: Vec3,
    b: Vec3,
    radius: f32,
    /// Set for ore nodes.
    node: Option<NodeHit>,
}

impl Collider {
    fn distance(&self, p: Vec3) -> f32 {
        let ab = self.b - self.a;
        let t = if ab.length_squared() > 0.0 {
            ((p - self.a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0)
        } else {
            0.0
        };
        (p - (self.a + ab * t)).length() - self.radius
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub struct ScatterStats {
    pub batches: usize,
    pub instances: usize,
    pub in_flight: usize,
}

pub struct Scatter {
    config: Arc<Config>,
    jobs: Sender<Key>,
    results: Receiver<(Key, Option<Vec<Placed>>)>,
    /// The column the camera is in, for the worker to skip jobs it has left behind.
    center: Arc<AtomicI64>,
    raw: HashMap<Key, Vec<Placed>>,
    in_flight: HashSet<Key>,
    /// Batches to (re)filter and hand to the renderer.
    dirty: HashSet<Key>,
    colliders: HashMap<IVec2, Vec<Collider>>,
    /// Ore nodes that have been mined away.
    removed: HashSet<NodeId>,
    last_center: Option<IVec2>,
    stats: ScatterStats,
}

fn pack(c: IVec2) -> i64 {
    ((c.x as i64) << 32) | (c.y as u32 as i64)
}

fn unpack(v: i64) -> IVec2 {
    IVec2::new((v >> 32) as i32, v as i32)
}

impl Scatter {
    pub fn new(config: Config, worldgen: Arc<WorldGen>) -> Self {
        let config = Arc::new(config);
        let (job_tx, job_rx) = crossbeam_channel::unbounded::<Key>();
        let (done_tx, done_rx) = crossbeam_channel::unbounded();
        let center = Arc::new(AtomicI64::new(pack(IVec2::ZERO)));
        {
            let config = config.clone();
            let center = center.clone();
            thread::Builder::new()
                .name("scatter".into())
                .spawn(move || {
                    let _ =
                        thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Min);
                    for key in job_rx {
                        let (column, kind) = key;
                        // Skip work the camera has already moved away from.
                        let c = unpack(center.load(Ordering::Relaxed));
                        let here = (c.as_vec2() + 0.5) * COLUMN;
                        let wanted =
                            column_distance(column, here) < config.kinds[kind].distance + KEEP_MARGIN;
                        let result = wanted.then(|| generate(&config, &worldgen, column, kind));
                        if done_tx.send((key, result)).is_err() {
                            break;
                        }
                    }
                })
                .expect("spawning scatter worker");
        }
        Self {
            config,
            jobs: job_tx,
            results: done_rx,
            center,
            raw: HashMap::new(),
            in_flight: HashSet::new(),
            dirty: HashSet::new(),
            colliders: HashMap::new(),
            removed: HashSet::new(),
            last_center: None,
            stats: ScatterStats::default(),
        }
    }

    pub fn stats(&self) -> ScatterStats {
        ScatterStats { in_flight: self.in_flight.len(), ..self.stats }
    }

    /// Request what's newly in range, drop what's out of it, and upload finished batches.
    pub fn update(&mut self, camera: Vec3, world: &World, structures: &Structures, renderer: &mut Renderer) {
        let here = Vec2::new(camera.x, camera.z);
        let center = (here / COLUMN).floor().as_ivec2();
        self.center.store(pack(center), Ordering::Relaxed);
        if self.last_center != Some(center) {
            self.last_center = Some(center);
            let mut wanted: Vec<(f32, Key)> = Vec::new();
            for (kind, k) in self.config.kinds.iter().enumerate() {
                if k.cell <= 0.0 {
                    continue;
                }
                let reach = (k.distance / COLUMN).ceil() as i32 + 1;
                for dz in -reach..=reach {
                    for dx in -reach..=reach {
                        let column = center + IVec2::new(dx, dz);
                        let d = column_distance(column, here);
                        let key = (column, kind);
                        if d < k.distance && !self.raw.contains_key(&key) && !self.in_flight.contains(&key) {
                            wanted.push((d, key));
                        }
                    }
                }
            }
            wanted.sort_by(|a, b| a.0.total_cmp(&b.0));
            for (_, key) in wanted {
                self.in_flight.insert(key);
                let _ = self.jobs.send(key);
            }
            // Drop what has fallen well out of range.
            let config = &self.config;
            let gone: Vec<Key> = self
                .raw
                .keys()
                .filter(|(column, kind)| {
                    column_distance(*column, here) > config.kinds[*kind].distance + KEEP_MARGIN
                })
                .copied()
                .collect();
            for key in gone {
                self.raw.remove(&key);
                self.dirty.remove(&key);
                renderer.set_scatter(batch_key(key), None);
                if self.config.kinds[key.1].collide != Collide::None {
                    self.colliders.remove(&key.0);
                }
            }
        }
        for (key, result) in self.results.try_iter() {
            self.in_flight.remove(&key);
            if let Some(placed) = result {
                self.raw.insert(key, placed);
                self.dirty.insert(key);
            }
        }
        if !self.dirty.is_empty() {
            for key in std::mem::take(&mut self.dirty) {
                self.upload(key, world, structures, renderer);
            }
            self.stats.batches = self.raw.len();
            self.stats.instances = renderer.scatter_count();
        }
    }

    /// Filter one batch against edits and buildings and hand it to the renderer.
    fn upload(&mut self, key: Key, world: &World, structures: &Structures, renderer: &mut Renderer) {
        let Some(raw) = self.raw.get(&key) else { return };
        let k = &self.config.kinds[key.1];
        let removed = &self.removed;
        let kept: Vec<&Placed> = raw.iter().filter(|p| keep(world, structures, removed, p)).collect();
        let instances = kept
            .iter()
            .map(|p| PartInstance {
                model: p.model,
                position: p.position,
                yaw: p.yaw,
                scale: p.scale,
                sway: k.sway,
            })
            .collect();
        renderer.set_scatter(
            batch_key(key),
            Some(ScatterBatch {
                instances,
                distance: k.distance,
                fade: k.fade,
                shadow_distance: k.shadow_distance,
            }),
        );
        if k.collide != Collide::None {
            // Colliders for a column come from every colliding kind there; rebuild them all.
            let mut list = Vec::new();
            for (kind, kk) in self.config.kinds.iter().enumerate() {
                if kk.collide == Collide::None {
                    continue;
                }
                let Some(raw) = self.raw.get(&(key.0, kind)) else { continue };
                for p in raw.iter().filter(|p| keep(world, structures, removed, p)) {
                    let node = match (&kk.nodes, p.ore) {
                        (Some((node, _)), Some(ore)) => {
                            Some(NodeHit { id: node_id(p.position), ore: ore as usize, kind: *node })
                        }
                        _ => None,
                    };
                    match kk.collide {
                        Collide::Trunk => list.push(Collider {
                            a: p.position - Vec3::Y * 0.5,
                            b: p.position + Vec3::Y * (3.5 * p.scale),
                            radius: p.collider_radius,
                            node,
                        }),
                        Collide::Rock if p.collider_radius > 0.25 || node.is_some() => {
                            let c = p.position + Vec3::Y * p.collider_center;
                            list.push(Collider { a: c, b: c, radius: p.collider_radius, node });
                        }
                        _ => {}
                    }
                }
            }
            self.colliders.insert(key.0, list);
        }
    }

    /// Re-filter every batch near `p` (after digging, filling or building there).
    pub fn invalidate(&mut self, p: Vec3, radius: f32) {
        let here = Vec2::new(p.x, p.z);
        for key in self.raw.keys() {
            if column_distance(key.0, here) < radius {
                self.dirty.insert(*key);
            }
        }
    }

    /// The ore node whose collider `p` is on (within `tolerance`), if any.
    pub fn node_at(&self, p: Vec3, tolerance: f32) -> Option<NodeHit> {
        let column = (Vec2::new(p.x, p.z) / COLUMN).floor().as_ivec2();
        let mut best: Option<(f32, NodeHit)> = None;
        for dz in -1..=1 {
            for dx in -1..=1 {
                for c in self.colliders.get(&(column + IVec2::new(dx, dz))).into_iter().flatten() {
                    if let Some(node) = c.node {
                        let d = c.distance(p);
                        if d < tolerance && best.is_none_or(|(b, _)| d < b) {
                            best = Some((d, node));
                        }
                    }
                }
            }
        }
        best.map(|(_, n)| n)
    }

    /// Take a node out of the world (it has been mined).
    pub fn remove_node(&mut self, id: NodeId) {
        self.removed.insert(id);
        self.invalidate(id.as_vec3() / 4.0, 1.0);
    }

    /// The nodes mined so far, for saving.
    pub fn removed_nodes(&self) -> Vec<NodeId> {
        let mut v: Vec<NodeId> = self.removed.iter().copied().collect();
        v.sort_by_key(|n| (n.x, n.y, n.z));
        v
    }

    /// Restore the mined nodes from a save.
    pub fn set_removed_nodes(&mut self, nodes: impl IntoIterator<Item = NodeId>) {
        self.removed = nodes.into_iter().collect();
        self.dirty.extend(self.raw.keys().copied());
    }

    /// Signed distance to the nearest trunk or rock collider (large when there's none nearby).
    pub fn sdf(&self, p: Vec3) -> f32 {
        let column = (Vec2::new(p.x, p.z) / COLUMN).floor().as_ivec2();
        let mut best = f32::MAX;
        for dz in -1..=1 {
            for dx in -1..=1 {
                if let Some(list) = self.colliders.get(&(column + IVec2::new(dx, dz))) {
                    for c in list {
                        best = best.min(c.distance(p));
                    }
                }
            }
        }
        best
    }
}

/// Whether an instance still has its ground, isn't in a building's way and (ore nodes) hasn't
/// been mined.
fn keep(world: &World, structures: &Structures, removed: &HashSet<NodeId>, placed: &Placed) -> bool {
    let p = placed.position;
    if placed.ore.is_some() && removed.contains(&node_id(p)) {
        return false;
    }
    if structures.sdf(p + Vec3::Y * 0.5) < 1.0 {
        return false;
    }
    if world.edits.ops_near(p).next().is_some() {
        // Edited here: keep it only if the ground surface is still where it was planted.
        return world.density_at(p + Vec3::Y * 0.4) > 0.0 && world.density_at(p - Vec3::Y * 0.4) < 0.0;
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config() -> Config {
        let model = |id: u16, kind: &str, biomes: &[&str]| crate::assets::NatureModel {
            spec: crate::assets::NatureSpec {
                id: format!("m{id}"),
                kind: kind.into(),
                biomes: biomes.iter().map(|b| b.to_string()).collect(),
            },
            id: ModelId(id),
            min: Vec3::new(-1.0, -0.3, -1.0),
            max: Vec3::new(1.0, 8.0, 1.0),
        };
        let kind = |distance: f32, collide: Collide| crate::assets::KindSpec {
            distance,
            fade: 10.0,
            scale: [0.8, 1.2],
            sway: 1.0,
            max_slope: 0.8,
            shadow_distance: 100.0,
            collide,
            lod: None,
            mid_lod: None,
        };
        let all: Vec<&str> = Biome::ALL.iter().map(|b| b.name()).collect();
        let nature = NatureModels {
            models: vec![model(0, "tree", &all), model(1, "grass", &all)],
            kinds: [
                ("tree".to_string(), kind(600.0, Collide::Trunk)),
                ("grass".to_string(), kind(45.0, Collide::None)),
            ]
            .into_iter()
            .collect(),
            density: Biome::ALL
                .iter()
                .map(|b| {
                    (
                        b.name().to_string(),
                        [("tree".to_string(), 0.03), ("grass".to_string(), 1.0)].into_iter().collect(),
                    )
                })
                .collect(),
        };
        Config::new(1337, &nature, None)
    }

    #[test]
    fn placement_is_deterministic_and_inside_its_column() {
        let config = config();
        let worldgen = WorldGen::new(1337);
        for column in [IVec2::new(0, 0), IVec2::new(-3, 7), IVec2::new(120, -45)] {
            for kind in 0..2 {
                let a = generate(&config, &worldgen, column, kind);
                let b = generate(&config, &worldgen, column, kind);
                assert_eq!(a, b);
                let lo = column.as_vec2() * COLUMN;
                for p in &a {
                    assert!(p.position.x >= lo.x && p.position.x < lo.x + COLUMN);
                    assert!(p.position.z >= lo.y && p.position.z < lo.y + COLUMN);
                    assert!((p.position.y - worldgen.height(p.position.x, p.position.z)).abs() < 1e-3);
                }
            }
        }
    }

    #[test]
    fn densities_come_out_about_right() {
        let config = config();
        let worldgen = WorldGen::new(1337);
        // Grass at 1 per m^2 everywhere (less in woods, none underwater or on steep ground).
        let mut grass = 0;
        let mut land = 0;
        for c in 0..8 {
            let column = IVec2::new(c * 37 - 100, c * 11 + 5);
            grass += generate(&config, &worldgen, column, config.kind("grass").unwrap()).len();
            land += 1;
        }
        let per_m2 = grass as f32 / (land as f32 * COLUMN * COLUMN);
        assert!(per_m2 > 0.2 && per_m2 < 1.05, "grass {per_m2}/m^2");
    }

    #[test]
    fn instances_stand_on_the_real_surface() {
        // The placement height is the height field; the terrain is its density field with caves
        // carved in. Find the surface under each instance by bisection and compare.
        let config = config();
        let worldgen = WorldGen::new(1337);
        let mut worst = 0.0f32;
        let mut checked = 0;
        for c in 0..40 {
            let column = IVec2::new(c * 7 - 140, (c * 13) % 50 - 25);
            for p in generate(&config, &worldgen, column, config.kind("tree").unwrap()) {
                let density = |y: f32| worldgen.sample(Vec3::new(p.position.x, y, p.position.z)).0;
                let (mut lo, mut hi) = (p.position.y - 6.0, p.position.y + 6.0);
                if density(lo) >= 0.0 || density(hi) <= 0.0 {
                    continue;
                }
                for _ in 0..30 {
                    let mid = 0.5 * (lo + hi);
                    if density(mid) < 0.0 { lo = mid } else { hi = mid }
                }
                worst = worst.max((lo - p.position.y).abs());
                checked += 1;
            }
        }
        assert!(checked > 50, "only {checked} checked");
        assert!(worst < 0.25, "an instance is {worst:.2} m off the surface");
    }

    #[test]
    fn capsule_distance() {
        let c = Collider { a: Vec3::ZERO, b: Vec3::Y * 4.0, radius: 0.3, node: None };
        assert!((c.distance(Vec3::new(1.0, 2.0, 0.0)) - 0.7).abs() < 1e-5);
        assert!((c.distance(Vec3::new(0.0, 5.0, 0.0)) - 0.7).abs() < 1e-5);
        assert!(c.distance(Vec3::new(0.1, 1.0, 0.0)) < 0.0);
    }
}
