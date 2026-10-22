//! The fluid simulation: water flowing through materialised chunks.
//!
//! A mass-conserving cellular automaton in integers. Each tick has two passes over the awake
//! chunks: vertical (water falls; deep water pushes up a little under pressure), then horizontal
//! (water levels out sideways). Every flow between two cells is worked out once, from last pass's
//! values, as an amount moving one way, so it leaves one cell exactly as it enters the other: water
//! is conserved, the order cells are visited in doesn't matter, and chunks run in parallel with the
//! same result on any number of threads.
//!
//! The edges of the simulated region: a neighbouring chunk that isn't materialised is read as a
//! "ghost" of its pristine state. Pristine water there is a reservoir (the sea refills whatever
//! drains from it); pristine air that water wants to enter brings that chunk to life on the next
//! tick. Settled chunks fall asleep; edits and arriving water wake them.

use std::collections::{BTreeMap, BTreeSet};

use ahash::AHashMap;
use asc_world::fluid::{self, CELLS, COMPRESS, FULL, FluidChunk, FluidKind, cell_index};
use asc_world::{CHUNK, ChunkCoord, EditOp, World};
use glam::{IVec3, Vec3};
use rayon::prelude::*;

/// Smaller sideways differences than this don't flow (or settled water would ripple forever).
const MIN_DIFFERENCE: i32 = 6;
/// Lava is viscous: it spreads a twelfth as readily and stops at a far steeper slope, so it piles
/// into thick lobes instead of thin sheets.
const LAVA_SPREAD: i32 = 12;
const LAVA_MIN_DIFFERENCE: i32 = 150;
/// Lava fuller than this is "still" and quenches to obsidian; thinner, flowing lava to basalt.
const STILL_LAVA: u16 = 900;
/// Water lost to steam each time it quenches a lava cell.
const STEAM: u16 = 250;
/// A chunk with no change for this many ticks falls asleep.
const SLEEP_AFTER: u32 = 3;
const N: i32 = CHUNK;

/// One chunk's pass: its new kinds and amounts, whether anything changed, and the ghost chunks
/// water wanted to enter.
type PassResult = (Vec<u8>, Vec<u16>, bool, Vec<ChunkCoord>);

/// What a cell looks like to its neighbours this pass.
#[derive(Clone, Copy, Default)]
struct Cell {
    kind: u8,
    amount: i32,
    solid: bool,
    /// A ghost cell: read-only (pristine water is a reservoir; air there can't be entered yet).
    ghost: bool,
}

/// How much of `total` the lower of two stacked cells holds when settled: all of it up to a full
/// cell, then slightly more than the upper one (compression), so pressure can push water up.
fn settled_lower(total: i32) -> i32 {
    let (full, comp) = (FULL as i32, COMPRESS as i32);
    if total <= full {
        total
    } else if total < 2 * full + comp {
        (full * full + total * comp) / (full + comp)
    } else {
        (total + comp) / 2
    }
}

/// Flow from `upper` down into `lower` (negative: pushed up). At most half of the giver goes, as
/// each cell has two vertical neighbours.
fn vertical(lower: Cell, upper: Cell) -> i32 {
    if lower.solid || upper.solid || (lower.kind != 0 && upper.kind != 0 && lower.kind != upper.kind) {
        return 0;
    }
    let want = settled_lower(lower.amount + upper.amount) - lower.amount;
    // Down: at most the upper's larger half; up: at most the lower's smaller half. A cell is the
    // upper of one pair and the lower of the other, so together it never gives more than it has.
    let flow = if want > 0 { want.min(upper.amount - upper.amount / 2) } else { want.max(-(lower.amount / 2)) };
    if (flow > 0 && lower.ghost && lower.kind == 0) || (flow < 0 && upper.ghost && upper.kind == 0) {
        // Into a ghost's air: wait for it to come to life.
        return 0;
    }
    flow
}

/// Flow from `a` sideways into `b` (negative: b into a).
fn sideways(a: Cell, b: Cell) -> i32 {
    if a.solid || b.solid || (a.kind != 0 && b.kind != 0 && a.kind != b.kind) {
        return 0;
    }
    let diff = a.amount - b.amount;
    let lava = a.kind == FluidKind::Lava as u8 || b.kind == FluidKind::Lava as u8;
    let (min, spread) = if lava { (LAVA_MIN_DIFFERENCE, LAVA_SPREAD) } else { (MIN_DIFFERENCE, 5) };
    if diff.abs() < min {
        return 0;
    }
    let flow = diff / spread;
    if (flow > 0 && b.ghost && b.kind == 0) || (flow < 0 && a.ghost && a.kind == 0) {
        return 0;
    }
    flow
}

pub struct FluidSim {
    pub fluids: fluid::Fluids,
    /// Pristine snapshots of the neighbours of materialised chunks (read-only edges).
    ghosts: AHashMap<ChunkCoord, FluidChunk>,
    /// Awake chunks (by coordinate array, for a deterministic order) and how many ticks each has
    /// been quiet.
    awake: BTreeMap<[i32; 3], u32>,
    /// Chunks changed since the last `take_changed`, to send to clients.
    changed: BTreeSet<[i32; 3]>,
    /// Lava cells that met water and set into rock since the last `take_solidified`.
    solidified: Vec<(IVec3, asc_world::Material)>,
}

const FACES: [IVec3; 6] =
    [IVec3::NEG_X, IVec3::X, IVec3::NEG_Y, IVec3::Y, IVec3::NEG_Z, IVec3::Z];

impl FluidSim {
    pub fn new() -> Self {
        Self {
            fluids: fluid::Fluids::default(),
            ghosts: AHashMap::new(),
            awake: BTreeMap::new(),
            changed: BTreeSet::new(),
            solidified: Vec::new(),
        }
    }

    /// Cells where lava met water and became rock, for the server to turn into terrain.
    pub fn take_solidified(&mut self) -> Vec<(IVec3, asc_world::Material)> {
        std::mem::take(&mut self.solidified)
    }

    /// Where lava touches water, the lava sets into rock (obsidian where it was still, basalt where
    /// it flowed) and the water loses some to steam. Run after the passes, one chunk at a time in
    /// coordinate order (it writes across chunk edges), so it stays deterministic.
    fn react(&mut self, any_changed: &mut BTreeSet<[i32; 3]>) {
        let coords: Vec<[i32; 3]> = self.awake.keys().copied().collect();
        for k in coords {
            let coord = IVec3::from_array(k);
            let Some(ch) = self.fluids.chunks.get(&coord) else { continue };
            if !ch.kind.contains(&(FluidKind::Lava as u8)) {
                continue;
            }
            let mut hits: Vec<(IVec3, u16, IVec3)> = Vec::new();
            for i in 0..CELLS {
                if ch.kind[i] != FluidKind::Lava as u8 || ch.amount[i] == 0 {
                    continue;
                }
                let p = IVec3::new((i % 32) as i32, ((i / 32) % 32) as i32, (i / 1024) as i32);
                for f in FACES {
                    let other = self.peek(coord, p + f);
                    if other.kind == FluidKind::Water as u8 && other.amount > 0 && !other.ghost {
                        hits.push((p, ch.amount[i], p + f));
                        break;
                    }
                }
            }
            for (p, amount, water) in hits {
                let material = if amount >= STILL_LAVA { asc_world::Material::Obsidian } else { asc_world::Material::Basalt };
                let cell = coord * N + p;
                self.solidified.push((cell, material));
                let ch = self.fluids.chunks.get_mut(&coord).expect("still there");
                let i = cell_index(p.x as usize, p.y as usize, p.z as usize);
                ch.amount[i] = 0;
                ch.kind[i] = 0;
                ch.solid[i] = true;
                // The water side gives some up as steam.
                let (wc, wl) = fluid::split(coord * N + water);
                if let Some(wch) = self.fluids.chunks.get_mut(&wc) {
                    let j = cell_index(wl.x as usize, wl.y as usize, wl.z as usize);
                    wch.amount[j] = wch.amount[j].saturating_sub(STEAM);
                    if wch.amount[j] == 0 {
                        wch.kind[j] = 0;
                    }
                    any_changed.insert(wc.to_array());
                }
                any_changed.insert(k);
            }
        }
    }

    pub fn awake_count(&self) -> usize {
        self.awake.len()
    }

    /// Chunks whose fluid changed since the last call.
    pub fn take_changed(&mut self) -> Vec<ChunkCoord> {
        std::mem::take(&mut self.changed).into_iter().map(IVec3::from_array).collect()
    }

    /// Total fluid in materialised chunks (for tests and logs).
    pub fn total(&self) -> u64 {
        self.fluids.chunks.values().map(FluidChunk::total).sum()
    }

    /// Whether any natural water lies in or right next to `coord` (a coarse look at its columns).
    fn near_water(world: &World, coord: ChunkCoord) -> bool {
        let base = (coord * N).as_vec3();
        if base.y > asc_world::SEA_LEVEL + 2.0 || base.y + (N as f32) < -60.0 {
            return false;
        }
        (0..=8).any(|j| {
            (0..=8).any(|i| {
                let (x, z) = (base.x - 2.0 + i as f32 * 4.5, base.z - 2.0 + j as f32 * 4.5);
                fluid::water_level(&world.worldgen, x, z).is_some()
            })
        })
    }

    /// Bring a chunk to life from its pristine state (or from its ghost, if there is one).
    fn materialise(&mut self, world: &World, coord: ChunkCoord) {
        if self.fluids.chunks.contains_key(&coord) {
            return;
        }
        let chunk = match self.ghosts.remove(&coord) {
            Some(g) => g,
            None => fluid::pristine_chunk(&world.worldgen, &world.build_chunk(coord)),
        };
        self.fluids.chunks.insert(coord, chunk);
        self.awake.insert(coord.to_array(), 0);
        self.changed.insert(coord.to_array());
        for f in FACES {
            self.ensure_ghost(world, coord + f);
        }
    }

    fn ensure_ghost(&mut self, world: &World, coord: ChunkCoord) {
        if !self.fluids.chunks.contains_key(&coord) && !self.ghosts.contains_key(&coord) {
            self.ghosts.insert(coord, fluid::pristine_chunk(&world.worldgen, &world.build_chunk(coord)));
        }
    }

    /// The terrain changed: refresh the solid cells of the chunks the edit touched (waking them),
    /// and bring water to life around it if there is any near. Water in cells that became solid is
    /// pushed up into the cell above where there's room.
    pub fn disturb(&mut self, world: &World, op: &EditOp) {
        let touched: Vec<ChunkCoord> = op.touched_chunks().collect();
        for &coord in &touched {
            if !self.fluids.chunks.contains_key(&coord) && Self::near_water(world, coord) {
                self.materialise(world, coord);
            }
            if self.ghosts.contains_key(&coord) {
                // A ghost's terrain changed: snapshot it again.
                self.ghosts.insert(coord, fluid::pristine_chunk(&world.worldgen, &world.build_chunk(coord)));
            }
            let Some(chunk) = self.fluids.chunks.get_mut(&coord) else { continue };
            let terrain = world.build_chunk(coord);
            let n = N as usize;
            for z in 0..n {
                for y in 0..n {
                    for x in 0..n {
                        let i = cell_index(x, y, z);
                        let solid = terrain.density[asc_world::ChunkData::index(x + 1, y + 1, z + 1)] < 0.0;
                        if solid && chunk.amount[i] > 0 && y + 1 < n {
                            let up = cell_index(x, y + 1, z);
                            if !chunk.solid[up] {
                                chunk.amount[up] = chunk.amount[up].saturating_add(chunk.amount[i]);
                                chunk.kind[up] = chunk.kind[i];
                            }
                            chunk.amount[i] = 0;
                            chunk.kind[i] = 0;
                        }
                        chunk.solid[i] = solid;
                    }
                }
            }
            self.awake.insert(coord.to_array(), 0);
            self.changed.insert(coord.to_array());
        }
    }

    /// A cell for a neighbour lookup: materialised, ghost, or (with neither) a wall.
    fn peek(&self, coord: ChunkCoord, local: IVec3) -> Cell {
        let (mut c, mut l) = (coord, local);
        for axis in 0..3 {
            if l[axis] < 0 {
                c[axis] -= 1;
                l[axis] += N;
            } else if l[axis] >= N {
                c[axis] += 1;
                l[axis] -= N;
            }
        }
        let i = cell_index(l.x as usize, l.y as usize, l.z as usize);
        if let Some(ch) = self.fluids.chunks.get(&c) {
            return Cell { kind: ch.kind[i], amount: ch.amount[i] as i32, solid: ch.solid[i], ghost: false };
        }
        if let Some(g) = self.ghosts.get(&c) {
            return Cell { kind: g.kind[i], amount: g.amount[i] as i32, solid: g.solid[i], ghost: true };
        }
        Cell { solid: true, ..Default::default() }
    }

    /// One pass over one chunk: its new cells, whether anything changed, and the ghost chunks
    /// water wanted to enter.
    fn pass(&self, coord: ChunkCoord, vertical_pass: bool) -> PassResult {
        let ch = &self.fluids.chunks[&coord];
        let mut kind = ch.kind.clone();
        let mut amount = ch.amount.clone();
        let mut changed = false;
        let mut wanted = Vec::new();
        for z in 0..N {
            for y in 0..N {
                for x in 0..N {
                    let i = cell_index(x as usize, y as usize, z as usize);
                    if ch.solid[i] {
                        continue;
                    }
                    let here = Cell { kind: ch.kind[i], amount: ch.amount[i] as i32, solid: false, ghost: false };
                    let p = IVec3::new(x, y, z);
                    let mut net = 0;
                    let mut inflow_kind = 0u8;
                    let neighbours: &[(IVec3, bool)] = if vertical_pass {
                        &[(IVec3::NEG_Y, true), (IVec3::Y, false)]
                    } else {
                        &[(IVec3::NEG_X, false), (IVec3::X, false), (IVec3::NEG_Z, false), (IVec3::Z, false)]
                    };
                    for &(d, below) in neighbours {
                        let other = self.peek(coord, p + d);
                        // Flow out of `here` into `other` (positive) or in (negative).
                        let out = if vertical_pass {
                            if below { vertical(other, here) } else { -vertical(here, other) }
                        } else {
                            sideways(here, other)
                        };
                        if out < 0 && other.kind != 0 {
                            inflow_kind = other.kind;
                        }
                        if out != 0 && other.ghost && other.kind == 0 {
                            // (Blocked above; only reached if the rules change.)
                        }
                        net -= out;
                        // Water heading into a ghost's air: that chunk must come to life.
                        let wants = if vertical_pass {
                            if below {
                                other.ghost && other.kind == 0 && !other.solid && here.amount > 0
                            } else {
                                false
                            }
                        } else {
                            other.ghost && other.kind == 0 && !other.solid && here.amount - other.amount >= MIN_DIFFERENCE
                        };
                        if wants {
                            let lp = p + d;
                            wanted.push(coord + lp.div_euclid(IVec3::splat(N)));
                        }
                    }
                    if net != 0 {
                        let new = (here.amount + net).clamp(0, u16::MAX as i32);
                        amount[i] = new as u16;
                        if new == 0 {
                            kind[i] = 0;
                        } else if here.kind == 0 {
                            kind[i] = inflow_kind.max(FluidKind::Water as u8);
                        }
                        changed = true;
                    }
                }
            }
        }
        (kind, amount, changed, wanted)
    }

    /// One tick: a vertical pass then a horizontal one over every awake chunk, in parallel.
    pub fn tick(&mut self, world: &World) {
        if self.awake.is_empty() {
            return;
        }
        let mut any_changed: BTreeSet<[i32; 3]> = BTreeSet::new();
        let mut wanted: BTreeSet<[i32; 3]> = BTreeSet::new();
        for vertical_pass in [true, false] {
            let coords: Vec<ChunkCoord> = self.awake.keys().map(|k| IVec3::from_array(*k)).collect();
            let results: Vec<(ChunkCoord, PassResult)> =
                coords.par_iter().map(|&c| (c, self.pass(c, vertical_pass))).collect();
            for (coord, (kind, amount, changed, want)) in results {
                if changed {
                    let ch = self.fluids.chunks.get_mut(&coord).expect("awake chunks are materialised");
                    ch.kind = kind;
                    ch.amount = amount;
                    any_changed.insert(coord.to_array());
                }
                wanted.extend(want.into_iter().map(|c| c.to_array()));
            }
        }
        self.react(&mut any_changed);
        for coord in wanted {
            self.materialise(world, IVec3::from_array(coord));
        }
        // Changed chunks stay awake and wake their materialised neighbours; quiet ones doze off.
        let mut wake = Vec::new();
        for (coord, quiet) in self.awake.iter_mut() {
            if any_changed.contains(coord) {
                *quiet = 0;
                for f in FACES {
                    wake.push(IVec3::from_array(*coord) + f);
                }
            } else {
                *quiet += 1;
            }
        }
        self.awake.retain(|_, quiet| *quiet < SLEEP_AFTER);
        for c in wake {
            if self.fluids.chunks.contains_key(&c) {
                self.awake.entry(c.to_array()).or_insert(0);
            }
        }
        self.changed.extend(any_changed);
    }

    /// The fluid at a world point (materialised or pristine).
    pub fn fraction_at(&self, world: &World, p: Vec3) -> (FluidKind, f32) {
        self.fluids.fraction_at(&world.worldgen, p)
    }

    /// Restore saved chunks (solidity is rebuilt from the terrain) and wake them.
    pub fn restore(&mut self, world: &World, saved: Vec<(ChunkCoord, Vec<u8>, Vec<u16>)>) {
        for (coord, kind, amount) in saved {
            if kind.len() != CELLS || amount.len() != CELLS {
                continue;
            }
            let mut ch = fluid::pristine_chunk(&world.worldgen, &world.build_chunk(coord));
            ch.kind = kind;
            ch.amount = amount;
            self.fluids.chunks.insert(coord, ch);
            self.awake.insert(coord.to_array(), 0);
        }
        let coords: Vec<ChunkCoord> = self.fluids.chunks.keys().copied().collect();
        for c in coords {
            for f in FACES {
                self.ensure_ghost(world, c + f);
            }
        }
    }
}

impl Default for FluidSim {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asc_world::EditKind;

    /// A small pool scene: a closed stone box (all chunks materialised by hand, no ghosts).
    fn boxed(world: &World) -> FluidSim {
        let mut sim = FluidSim::new();
        let mut ch = FluidChunk::empty();
        let n = N as usize;
        for z in 0..n {
            for y in 0..n {
                for x in 0..n {
                    let edge = x == 0 || z == 0 || y == 0 || x == n - 1 || z == n - 1;
                    ch.solid[cell_index(x, y, z)] = edge;
                }
            }
        }
        // A 10 x 10 x 3 block of water dropped in one corner, high up.
        for z in 2..12 {
            for x in 2..12 {
                for y in 15..18 {
                    let i = cell_index(x, y, z);
                    ch.kind[i] = 1;
                    ch.amount[i] = FULL;
                }
            }
        }
        let coord = IVec3::new(1000, 1000, 1000);
        sim.fluids.chunks.insert(coord, ch);
        sim.awake.insert(coord.to_array(), 0);
        let _ = world;
        sim
    }

    #[test]
    fn water_is_conserved_and_settles_flat() {
        let world = World::new(1);
        let mut sim = boxed(&world);
        let start = sim.total();
        let mut ticks = 0;
        while sim.awake_count() > 0 && ticks < 4000 {
            sim.tick(&world);
            ticks += 1;
            assert_eq!(sim.total(), start, "tick {ticks}");
        }
        assert!(sim.awake_count() == 0, "settled within {ticks} ticks");
        // 300 cells of water over a 30 x 30 floor: about a third of a cell deep everywhere (below
        // MIN_DIFFERENCE per cell water stops spreading, so allow a gentle slope).
        let ch = sim.fluids.chunks.values().next().unwrap();
        let floor: Vec<u16> = (1..31).flat_map(|z| (1..31).map(move |x| (x, z))).map(|(x, z)| ch.amount[cell_index(x, 1, z)]).collect();
        let (lo, hi) = (*floor.iter().min().unwrap(), *floor.iter().max().unwrap());
        assert!(lo > 200 && hi < 470, "floor varies {lo}..{hi}");
        let above: u64 = (0..CELLS).filter(|i| (i / 32) % 32 >= 2).map(|i| ch.amount[i] as u64).sum();
        assert_eq!(above, 0, "nothing left standing above the floor layer");
        println!("settled in {ticks} ticks");
    }

    #[test]
    fn the_result_does_not_depend_on_the_thread_count() {
        let world = World::new(1);
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut sim = boxed(&world);
                for _ in 0..200 {
                    sim.tick(&world);
                }
                sim.fluids.chunks.values().next().unwrap().amount.clone()
            })
        };
        assert_eq!(run(1), run(6));
    }

    #[test]
    fn a_channel_dug_from_the_sea_fills_without_draining_it() {
        let world_seed = 1337;
        let mut world = World::new(world_seed);
        let wg = world.worldgen.clone();
        // A shore: sea on one side, land just above sea level on the other.
        let (x, z) = (0..20000)
            .map(|i| ((i % 200) as f32 * 3.0 - 300.0, (i / 200) as f32 * 3.0 - 300.0))
            .find(|&(x, z)| {
                wg.height(x, z) < asc_world::SEA_LEVEL - 2.0
                    && (wg.height(x + 12.0, z) - (asc_world::SEA_LEVEL + 1.0)).abs() < 1.0
            })
            .expect("a shore");
        let mut sim = FluidSim::new();
        // Dig a trench from the water inland, below sea level.
        for k in 0..10 {
            let op = EditOp {
                center: Vec3::new(x + 2.0 + k as f32 * 1.5, asc_world::SEA_LEVEL - 1.5, z),
                radius: 1.6,
                kind: EditKind::Dig,
                material: asc_world::Material::Air,
            };
            world.edits.push(op);
            sim.disturb(&world, &op);
        }
        for _ in 0..600 {
            sim.tick(&world);
        }
        // The far end of the trench has water in it now.
        let end = Vec3::new(x + 14.0, asc_world::SEA_LEVEL - 1.5, z);
        let (k, frac) = sim.fraction_at(&world, end);
        assert_eq!(k, FluidKind::Water, "trench end at {end}");
        assert!(frac > 0.5, "trench end only {frac}");
        // And the sea is still at sea level just offshore.
        let sea = Vec3::new(x - 4.0, asc_world::SEA_LEVEL - 0.6, z);
        assert!(sim.fraction_at(&world, sea).1 > 0.9);
    }

    #[test]
    fn lava_spreads_less_than_water_and_sets_into_rock_on_meeting_it() {
        let world = World::new(1);
        // Lava: a block spreads into a thick lobe, not a film.
        let mut sim = boxed(&world);
        let coord = IVec3::new(1000, 1000, 1000);
        for k in sim.fluids.chunks.get_mut(&coord).unwrap().kind.iter_mut() {
            if *k == 1 {
                *k = FluidKind::Lava as u8;
            }
        }
        for _ in 0..3000 {
            sim.tick(&world);
        }
        let ch = &sim.fluids.chunks[&coord];
        let covered = (1..31).flat_map(|z| (1..31).map(move |x| (x, z))).filter(|&(x, z)| ch.amount[cell_index(x, 1, z)] > 0).count();
        assert!(covered < 600, "lava covered {covered} of 900 floor cells: too runny");
        assert!(covered > 100, "lava covered only {covered}");

        // Lava next to water: the lava cells touching it become rock.
        let mut sim = boxed(&world);
        let ch = sim.fluids.chunks.get_mut(&coord).unwrap();
        for z in 2..12 {
            for x in 2..12 {
                let i = cell_index(x, 1, z);
                ch.kind[i] = FluidKind::Lava as u8;
                ch.amount[i] = FULL;
                let j = cell_index(x, 1, z + 12);
                ch.kind[j] = FluidKind::Water as u8;
                ch.amount[j] = FULL;
            }
        }
        for _ in 0..200 {
            sim.tick(&world);
        }
        let rock = sim.take_solidified();
        assert!(!rock.is_empty(), "nothing set");
        assert!(rock.iter().all(|(_, m)| matches!(m, asc_world::Material::Obsidian | asc_world::Material::Basalt)));
    }

    #[test]
    fn deep_caves_hold_a_lava_sea() {
        let wg = asc_world::WorldGen::new(1337);
        // Somewhere far below, an open cell: lava.
        let found = (0..2000).find_map(|i| {
            let c = IVec3::new(i * 3 % 400 - 200, -170 - (i % 30), i * 7 % 400 - 200);
            match fluid::pristine(&wg, c) {
                (FluidKind::Lava, a) if a == FULL => Some(c),
                _ => None,
            }
        });
        assert!(found.is_some(), "no open cell found below the lava level");
        assert_ne!(fluid::pristine(&wg, IVec3::new(0, -100, 0)).0, FluidKind::Lava);
    }
}
