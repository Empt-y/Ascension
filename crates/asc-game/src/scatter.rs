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

pub use crate::ore_nodes::COLUMN;
/// Keep generated columns this much beyond a kind's draw distance before dropping them.
const KEEP_MARGIN: f32 = 2.0 * COLUMN;

#[derive(Clone, Copy, Debug)]
struct Model {
    id: ModelId,
    /// Collider at unit scale: trunk radius, or rock sphere radius and centre height.
    radius: f32,
    center_y: f32,
}

pub use crate::ore_nodes::{NodeId, NodeKind, node_id};

#[derive(Clone, Debug)]
struct Kind {
    name: String,
    distance: f32,
    fade: f32,
    sway: f32,
    shadow_distance: f32,
    collide: Collide,
    /// Plants and rocks: its index in `Config::nature` (where it grows).
    grows: Option<usize>,
    /// Ore node kinds: which nodes, and each ore's model.
    nodes: Option<(NodeKind, Vec<Model>)>,
}

/// Trunk collider radius at unit scale.
pub const TRUNK_RADIUS: f32 = 0.3;

/// A rock's collider sphere at unit scale, from its model's bounds: radius and centre height.
pub fn rock_collider(min: Vec3, max: Vec3) -> (f32, f32) {
    let half = (max - min) * 0.5;
    (0.85 * half.x.min(half.z).min(half.y * 1.3), (max.y + min.y) * 0.5)
}

/// Everything the generator needs, shared with the worker.
pub struct Config {
    seed: u64,
    kinds: Vec<Kind>,
    nature: asc_world::nature::Nature,
    /// By `nature` model index: its renderer model and collider.
    models: Vec<Model>,
    /// Each nature model's burnt stages.
    burnt: HashMap<ModelId, [ModelId; 3]>,
}

impl Config {
    pub fn new(seed: i32, nature: &NatureModels, ores: Option<&OreModels>) -> Self {
        // Debugging aid: ASC_SCATTER_ONLY=tree,rock scatters only those kinds.
        let only: Option<Vec<String>> =
            std::env::var("ASC_SCATTER_ONLY").ok().map(|v| v.split(',').map(str::to_owned).collect());
        let placement = asc_world::nature::Nature::from_parts(
            nature.kinds.iter().map(|(name, k)| (name.as_str(), k.scale, k.max_slope, k.collide)),
            &nature.density,
            nature
                .models
                .iter()
                .map(|m| asc_world::nature::ModelSpec {
                    id: m.spec.id.clone(),
                    kind: m.spec.kind.clone(),
                    biomes: m.spec.biomes.clone(),
                })
                .collect(),
        )
        .expect("nature.toml was checked as it loaded");
        let models = nature
            .models
            .iter()
            .map(|m| {
                let (radius, center_y) = match nature.kinds[&m.spec.kind].collide {
                    Collide::Trunk => (TRUNK_RADIUS, 0.0),
                    Collide::Rock => rock_collider(m.min, m.max),
                    Collide::None => (0.0, 0.0),
                };
                Model { id: m.id, radius, center_y }
            })
            .collect();
        let kinds = nature
            .kinds
            .iter()
            .enumerate()
            .filter(|(_, (name, _))| only.as_ref().is_none_or(|o| o.contains(name)))
            .map(|(index, (name, spec))| Kind {
                name: name.clone(),
                distance: spec.distance,
                fade: spec.fade,
                sway: spec.sway,
                shadow_distance: spec.shadow_distance,
                collide: spec.collide,
                grows: (placement.kinds[index].cell > 0.0).then_some(index),
                nodes: None,
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
                    sway: 0.0,
                    shadow_distance: shadow,
                    collide: Collide::Rock,
                    grows: None,
                    nodes: Some((node, models)),
                }
            };
            let wanted = |name: &str| only.as_ref().is_none_or(|o| o.iter().any(|n| n == name));
            if wanted("ore_surface") {
                kinds.push(node_kind("ore_surface", NodeKind::Surface, 150.0, 50.0));
            }
            if wanted("ore_deep") {
                // Underground: no sun reaches them, so no shadows.
                kinds.push(node_kind("ore_deep", NodeKind::Deep, 80.0, 0.0));
            }
        }
        Self { seed: crate::ore_nodes::seed_of(seed), kinds, nature: placement, models, burnt: nature.burnt.clone() }
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


/// The ore node a crosshair is on.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeHit {
    pub id: NodeId,
    pub ore: usize,
    pub kind: NodeKind,
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
    let Some(index) = k.grows else { return Vec::new() };
    asc_world::nature::grow(&config.nature, config.seed, worldgen, column, index)
        .into_iter()
        .map(|g| {
            let m = config.models[g.model as usize];
            Placed {
                model: m.id,
                position: g.position,
                yaw: g.yaw,
                scale: g.scale,
                collider_radius: m.radius * g.scale,
                collider_center: m.center_y * g.scale,
                ore: None,
            }
        })
        .collect()
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
    /// Burning or burnt ground: fire cell and stage (1 scorched .. 3 burnt out).
    fire: HashMap<IVec2, u8>,
    /// Where pyroclastic flows and blasts have passed: plants there are burnt out.
    scars: Vec<asc_sim::volcanism::Scar>,
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
            fire: HashMap::new(),
            scars: Vec::new(),
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
                if k.grows.is_none() && k.nodes.is_none() {
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
        // Plants on burning or burnt ground show their burnt stage.
        let scarred = |p: &Placed| self.scars.iter().any(|s| s.contains(Vec2::new(p.position.x, p.position.z)));
        let model = |p: &Placed| {
            let cell = asc_sim::fire::cell_of(p.position.x, p.position.z);
            let stage = if scarred(p) { Some(&3) } else { self.fire.get(&cell) };
            match (stage, self.config.burnt.get(&p.model)) {
                (Some(&stage), Some(stages)) if (1..=3).contains(&stage) => stages[stage as usize - 1],
                _ => p.model,
            }
        };
        let instances = kept
            .iter()
            .map(|p| PartInstance {
                model: model(p),
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

    /// Fire cells whose stage changed: plants there are redrawn at their burnt stage.
    pub fn set_fire(&mut self, changed: &[([i32; 2], u8)]) {
        for &(c, stage) in changed {
            let cell = IVec2::from_array(c);
            self.fire.insert(cell, stage);
            let centre = (cell.as_vec2() + 0.5) * asc_sim::fire::CELL;
            self.invalidate(Vec3::new(centre.x, 0.0, centre.y), asc_sim::fire::CELL);
        }
    }

    /// A pyroclastic flow or blast has passed: what grew there is burnt out.
    pub fn add_scar(&mut self, scar: asc_sim::volcanism::Scar) {
        let (centre, radius) = scar.bounds();
        self.scars.push(scar);
        self.invalidate(Vec3::new(centre.x, 0.0, centre.y), radius);
    }

    /// Take a node out of the world (it has been mined).
    pub fn remove_node(&mut self, id: NodeId) {
        self.removed.insert(id);
        self.invalidate(id.as_vec3() / 4.0, 1.0);
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
            burnt: Default::default(),
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

#[cfg(test)]
mod fingerprint {
    use super::*;

    fn rich() -> Config {
        let model = |id: u16, kind: &str, biomes: &[&str], h: f32| crate::assets::NatureModel {
            spec: crate::assets::NatureSpec { id: format!("m{id}"), kind: kind.into(), biomes: biomes.iter().map(|b| b.to_string()).collect() },
            id: ModelId(id),
            min: Vec3::new(-1.0, -0.3, -0.7),
            max: Vec3::new(1.0, h, 1.0),
        };
        let kind = |distance: f32, collide: Collide, slope: f32| crate::assets::KindSpec {
            distance, fade: 10.0, scale: [0.7, 1.3], sway: 1.0, max_slope: slope, shadow_distance: 100.0, collide, lod: None, mid_lod: None,
        };
        let all: Vec<&str> = Biome::ALL.iter().map(|b| b.name()).collect();
        let nature = NatureModels {
            models: vec![
                model(0, "tree", &["forest", "meadow", "taiga"], 8.0),
                model(1, "tree", &["forest", "jungle"], 9.0),
                model(2, "rock", &all, 1.5),
                model(3, "tree", &["savanna", "forest"], 7.0),
                model(4, "grass", &all, 0.5),
                model(5, "rock", &["tundra", "arctic"], 2.5),
            ],
            burnt: Default::default(),
            kinds: [
                ("tree".to_string(), kind(600.0, Collide::Trunk, 0.7)),
                ("rock".to_string(), kind(200.0, Collide::Rock, 1.2)),
                ("grass".to_string(), kind(45.0, Collide::None, 0.9)),
            ]
            .into_iter()
            .collect(),
            density: Biome::ALL
                .iter()
                .enumerate()
                .map(|(i, b)| {
                    (
                        b.name().to_string(),
                        [("tree".to_string(), 0.01 * i as f32), ("rock".to_string(), 0.02), ("grass".to_string(), 0.5 + 0.05 * i as f32)].into_iter().collect(),
                    )
                })
                .collect(),
        };
        Config::new(1337, &nature, None)
    }
    /// What a mix of kinds and models places hasn't changed since placement moved to
    /// `asc_world::nature` (the server places trees from there too, so they must agree).
    #[test]
    fn placement_is_unchanged() {
        let config = rich();
        let worldgen = WorldGen::new(1337);
        let mut h: u64 = 0;
        let mut n = 0;
        for c in 0..30 {
            let column = IVec2::new(c * 7 - 100, (c * 13) % 50 - 25);
            for kind in 0..config.kinds.len() {
                for p in generate(&config, &worldgen, column, kind) {
                    for v in [p.position.x, p.position.y, p.position.z, p.yaw, p.scale, p.collider_radius] {
                        h = h.rotate_left(7) ^ v.to_bits() as u64 ^ (p.model.0 as u64) << 40;
                    }
                    n += 1;
                }
            }
        }
        assert_eq!((n, h), (9988, 0xda3f_08d0_8925_8ab8));
    }
}
