//! Where animals can go: a navigation grid over the ground near the player.
//!
//! The ground is split into `TILE`-cell square tiles of 1 m cells (a tile is a scatter column, so a
//! tile's trees and rocks are the column's). Each cell knows its surface height, how deep any water
//! over it is, how far it is from the nearest obstacle (trunk, rock, ore node, wall), and whether
//! animals graze or hide there. Tiles are built when something asks for them, a few each tick
//! (`build`), and rebuilt when the world changes under them (`invalidate`); far ones are dropped.
//!
//! The grid doesn't decide who can step where: that depends on the animal (`Profile`: its size, how
//! steep it climbs, how deep it wades, whether it swims), so one grid serves every kind.

use std::collections::{BTreeSet, BinaryHeap, VecDeque};
use std::sync::OnceLock;

use ahash::{AHashMap, AHashSet};
use asc_build::Structures;
use asc_world::fluid::{FluidKind, Fluids};
use asc_world::nature::{COLUMN, Collide, Nature, grow};
use asc_world::{Biome, World, raycast};
use rayon::prelude::*;
use glam::{IVec2, Vec2, Vec3};
use serde::Deserialize;

/// Cells per tile side (1 m cells, so a tile is a scatter column).
pub const TILE: i32 = COLUMN as i32;
const CELLS: usize = (TILE * TILE) as usize;
/// Clearance is measured out to this far (metres); anything further is "clear".
const MAX_CLEAR: f32 = 4.0;
/// A cell this close under a ceiling is too low to walk (metres).
const HEADROOM: f32 = 2.2;
/// Tiles kept within this of the player.
pub const KEEP: f32 = 288.0;

const COLLIDERS: &str = include_str!("../../../assets/data/colliders.toml");

/// Cell flags.
pub const BLOCKED: u8 = 1;
/// Grass or browse grows here.
pub const FORAGE: u8 = 2;
/// Woodland: somewhere to hide.
pub const COVER: u8 = 4;
/// Lava lies on it.
pub const LAVA: u8 = 8;
/// A building's floor (or roof).
pub const BUILT: u8 = 16;

/// One cell of the grid.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cell {
    /// The walking surface at the cell's middle.
    pub height: f32,
    /// Water over it, in tenths of a metre (saturating).
    pub water: u8,
    /// Distance to the nearest obstacle, in quarter metres (up to `MAX_CLEAR`).
    pub clear: u8,
    pub flags: u8,
    /// How hard going it is underfoot, in tenths over 1 (snow, sand, mud...).
    pub rough: u8,
}

impl Cell {
    pub fn depth(&self) -> f32 {
        self.water as f32 * 0.1
    }

    pub fn clearance(&self) -> f32 {
        self.clear as f32 * 0.25
    }

    pub fn has(&self, flag: u8) -> bool {
        self.flags & flag != 0
    }

    /// The cost of crossing it, as a multiple of plain ground.
    pub fn roughness(&self) -> f32 {
        1.0 + self.rough as f32 * 0.1
    }
}

/// How an animal gets about.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Profile {
    /// How wide it is: cells nearer an obstacle than this are out.
    pub radius: f32,
    /// The steepest ground it climbs (rise over run).
    pub max_slope: f32,
    /// A ledge it steps up whatever the slope (metres).
    pub step: f32,
    /// Land animals: the deepest water it wades. Swimmers: the shallowest it swims in.
    pub water: f32,
    pub swims: bool,
}

impl Profile {
    /// Whether it may stand in `c`.
    pub fn fits(&self, c: &Cell) -> bool {
        if c.has(BLOCKED) || c.has(LAVA) || c.clearance() < self.radius {
            return false;
        }
        if self.swims { c.depth() >= self.water } else { c.depth() <= self.water }
    }

    /// Whether it may go from `a` to its neighbour `b`, `run` metres apart.
    pub fn steps(&self, a: &Cell, b: &Cell, run: f32) -> bool {
        if !self.fits(b) {
            return false;
        }
        if self.swims {
            return true;
        }
        let rise = (b.height - a.height).abs();
        rise <= self.step || rise <= self.max_slope * run
    }
}

/// What the grid is built from.
pub struct Ground<'a> {
    pub world: &'a World,
    pub structures: &'a Structures,
    pub fluids: &'a Fluids,
    /// Ore nodes mined away (`ore_nodes::node_id`).
    pub mined: &'a BTreeSet<[i32; 3]>,
}

/// A trunk, rock or ore node: a circle on the ground.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Obstacle {
    pub at: Vec2,
    pub radius: f32,
}

struct Tile {
    cells: Box<[Cell]>,
    /// The tile has changed since it was built.
    stale: bool,
}

#[derive(Deserialize)]
struct CollidersFile {
    rocks: std::collections::BTreeMap<String, [f32; 2]>,
    ore_surface: std::collections::BTreeMap<String, [f32; 2]>,
}

/// Each nature model's collider radius at unit scale (0: none), and each ore's surface node's.
struct Colliders {
    nature: Vec<f32>,
    ores: Vec<f32>,
}

fn colliders() -> &'static Colliders {
    static C: OnceLock<Colliders> = OnceLock::new();
    C.get_or_init(|| {
        let file: CollidersFile = toml::from_str(COLLIDERS).expect("assets/data/colliders.toml");
        let nature = Nature::builtin();
        Colliders {
            nature: nature
                .models
                .iter()
                .map(|m| match nature.kinds[nature.kind(&m.kind).unwrap_or(0)].collide {
                    Collide::Trunk => crate::nav::TRUNK_RADIUS,
                    // Small rocks (pebbles) are stepped over, as the player does.
                    Collide::Rock => file.rocks.get(&m.id).map_or(0.0, |r| r[0]),
                    Collide::None => 0.0,
                })
                .collect(),
            ores: asc_world::ores::builtin()
                .ores
                .iter()
                .map(|o| file.ore_surface.get(&o.id).map_or(0.5, |r| r[0]))
                .collect(),
        }
    })
}

/// Trunk collider radius at unit scale (as the client's scatter).
pub const TRUNK_RADIUS: f32 = 0.3;
/// Rock colliders smaller than this (after scaling) aren't obstacles (as the client's scatter).
const SMALL_ROCK: f32 = 0.25;

/// The tile holding cell `c`, and the cell's index in it.
pub fn tile_of(c: IVec2) -> (IVec2, usize) {
    let t = IVec2::new(c.x.div_euclid(TILE), c.y.div_euclid(TILE));
    let l = c - t * TILE;
    (t, (l.y * TILE + l.x) as usize)
}

/// The cell under (x, z).
pub fn cell_at(p: Vec2) -> IVec2 {
    p.floor().as_ivec2()
}

/// A cell's middle.
pub fn centre(c: IVec2) -> Vec2 {
    c.as_vec2() + 0.5
}

#[derive(Default)]
pub struct NavGrid {
    seed: u64,
    tiles: AHashMap<IVec2, Tile>,
    /// Tiles wanted (not built yet, or stale), nearest the player first when built.
    wanted: AHashSet<IVec2>,
    /// Each column's obstacles, kept while its tiles are.
    obstacles: AHashMap<IVec2, Vec<Obstacle>>,
    /// Statistics: tiles built since the last `take_built`.
    built: usize,
}

impl NavGrid {
    pub fn new(world_seed: i32) -> Self {
        Self { seed: crate::ore_nodes::seed_of(world_seed), ..Default::default() }
    }

    /// The cell at `c`, if its tile is built.
    pub fn cell(&self, c: IVec2) -> Option<&Cell> {
        let (t, i) = tile_of(c);
        self.tiles.get(&t).map(|tile| &tile.cells[i])
    }

    /// The cell under (x, z), if built; otherwise asks for its tile.
    pub fn cell_or_want(&mut self, c: IVec2) -> Option<Cell> {
        let (t, i) = tile_of(c);
        match self.tiles.get(&t) {
            Some(tile) => Some(tile.cells[i]),
            None => {
                self.wanted.insert(t);
                None
            }
        }
    }

    /// Ask for the tiles within `radius` of `p` to be built.
    pub fn want(&mut self, p: Vec2, radius: f32) {
        let lo = cell_at(p - radius);
        let hi = cell_at(p + radius);
        let (t0, _) = tile_of(lo);
        let (t1, _) = tile_of(hi);
        for z in t0.y..=t1.y {
            for x in t0.x..=t1.x {
                let t = IVec2::new(x, z);
                if self.tiles.get(&t).is_none_or(|tile| tile.stale) {
                    self.wanted.insert(t);
                }
            }
        }
    }

    pub fn is_built(&self, p: Vec2) -> bool {
        self.tiles.contains_key(&tile_of(cell_at(p)).0)
    }

    /// The walking surface's height at (x, z), between the cells' middles; `None` if not built.
    pub fn height_at(&self, p: Vec2) -> Option<f32> {
        let q = p - 0.5;
        let c = q.floor().as_ivec2();
        let f = q - c.as_vec2();
        let h = |dx: i32, dz: i32| self.cell(c + IVec2::new(dx, dz)).map(|c| c.height);
        let (a, b, cc, d) = (h(0, 0)?, h(1, 0)?, h(0, 1)?, h(1, 1)?);
        // Across a cliff the cells' heights are metres apart: don't blend over the edge.
        let near = |x: f32| if (x - a).abs() > 1.5 { a } else { x };
        let (b, cc, d) = (near(b), near(cc), near(d));
        Some((a * (1.0 - f.x) + b * f.x) * (1.0 - f.y) + (cc * (1.0 - f.x) + d * f.x) * f.y)
    }

    /// Mark the tiles within `radius` of `p` as out of date (the ground, a building, water or an
    /// ore node changed there). They keep answering until they're rebuilt.
    pub fn invalidate(&mut self, p: Vec2, radius: f32) {
        let lo = cell_at(p - radius - MAX_CLEAR);
        let hi = cell_at(p + radius + MAX_CLEAR);
        let (t0, _) = tile_of(lo);
        let (t1, _) = tile_of(hi);
        for z in t0.y..=t1.y {
            for x in t0.x..=t1.x {
                let t = IVec2::new(x, z);
                if let Some(tile) = self.tiles.get_mut(&t) {
                    tile.stale = true;
                    self.wanted.insert(t);
                }
                self.obstacles.remove(&t);
            }
        }
    }

    /// Build (or rebuild) up to `max` wanted tiles, nearest `player` first, side by side on the
    /// thread pool; drop tiles beyond `KEEP`. Deterministic: which tiles, and what's in them,
    /// doesn't depend on timing.
    pub fn build(&mut self, ground: &Ground, player: Vec2, max: usize) {
        let mid = |t: IVec2| (t.as_vec2() + 0.5) * TILE as f32;
        self.tiles.retain(|t, _| mid(*t).distance(player) < KEEP);
        self.obstacles.retain(|t, _| mid(*t).distance(player) < KEEP + COLUMN);
        self.wanted.retain(|t| mid(*t).distance(player) < KEEP);
        if self.wanted.is_empty() {
            return;
        }
        let mut order: Vec<IVec2> = self.wanted.iter().copied().collect();
        order.sort_by(|a, b| mid(*a).distance(player).total_cmp(&mid(*b).distance(player)).then(a.to_array().cmp(&b.to_array())));
        order.truncate(max);
        // First the obstacles of every column these tiles reach into, then the tiles.
        let mut columns: Vec<IVec2> = order
            .iter()
            .flat_map(|t| (-1..=1).flat_map(move |dz| (-1..=1).map(move |dx| *t + IVec2::new(dx, dz))))
            .filter(|c| !self.obstacles.contains_key(c))
            .collect();
        columns.sort_by_key(|c| c.to_array());
        columns.dedup();
        let seed = self.seed;
        let found: Vec<(IVec2, Vec<Obstacle>)> =
            columns.par_iter().map(|c| (*c, obstacles_in(ground, seed, *c))).collect();
        self.obstacles.extend(found);
        let obstacles = &self.obstacles;
        let tiles: Vec<(IVec2, Box<[Cell]>)> = order.par_iter().map(|t| (*t, build_tile(ground, obstacles, *t))).collect();
        for (t, cells) in tiles {
            self.tiles.insert(t, Tile { cells, stale: false });
            self.wanted.remove(&t);
            self.built += 1;
        }
    }

    /// Tiles made up cell by cell (for tests).
    #[cfg(test)]
    pub(crate) fn fill(&mut self, tiles: &[IVec2], f: impl Fn(IVec2) -> Cell) {
        for t in tiles {
            let cells = (0..CELLS as i32).map(|k| f(*t * TILE + IVec2::new(k % TILE, k / TILE))).collect();
            self.tiles.insert(*t, Tile { cells, stale: false });
        }
    }

    /// Tiles built since last asked, and how many there are.
    pub fn take_built(&mut self) -> (usize, usize) {
        (std::mem::take(&mut self.built), self.tiles.len())
    }

    /// Whether a land animal (or swimmer) can go straight from `a` to `b`: every cell on the way
    /// fits it and each step between them is one it takes.
    pub fn line_clear(&self, a: Vec2, b: Vec2, profile: &Profile) -> bool {
        let d = b - a;
        let len = d.length();
        let steps = (len / 0.5).ceil().max(1.0) as i32;
        let mut prev: Option<(IVec2, Cell)> = None;
        for s in 0..=steps {
            let p = a + d * (s as f32 / steps as f32);
            let c = cell_at(p);
            if prev.is_some_and(|(pc, _)| pc == c) {
                continue;
            }
            let Some(cell) = self.cell(c) else { return false };
            match prev {
                Some((pc, pcell)) => {
                    let run = (c - pc).as_vec2().length();
                    if !profile.steps(&pcell, cell, run) {
                        return false;
                    }
                    // Cutting a corner diagonally: both cells beside the corner must fit too.
                    if run > 1.1 {
                        for side in [IVec2::new(c.x, pc.y), IVec2::new(pc.x, c.y)] {
                            if !self.cell(side).is_some_and(|sc| profile.fits(sc)) {
                                return false;
                            }
                        }
                    }
                }
                None if !profile.fits(cell) => return false,
                None => {}
            }
            prev = Some((c, *cell));
        }
        true
    }

    /// The nearest cell (by walking) from `from` where `wanted` holds, within `max` metres of
    /// walking: a Dijkstra search out from `from` over the cells `profile` can reach.
    pub fn nearest(&self, from: Vec2, profile: &Profile, max: f32, wanted: impl Fn(IVec2, &Cell) -> bool) -> Option<IVec2> {
        let start = cell_at(from);
        let first = self.cell(start)?;
        if wanted(start, first) {
            return Some(start);
        }
        let mut best: AHashMap<IVec2, f32> = AHashMap::new();
        let mut open = BinaryHeap::new();
        best.insert(start, 0.0);
        open.push(Open { cost: 0.0, cell: start });
        let mut budget = 20_000;
        while let Some(Open { cost, cell }) = open.pop() {
            if cost > best.get(&cell).copied().unwrap_or(f32::MAX) {
                continue;
            }
            budget -= 1;
            if budget == 0 {
                return None;
            }
            let here = *self.cell(cell)?;
            for (d, run) in NEIGHBOURS {
                let next = cell + d;
                let Some(nc) = self.cell(next) else { continue };
                let reach = wanted(next, nc);
                // The wanted cell needn't fit (water's edge for a drinker): only being next to it.
                if reach {
                    return Some(next);
                }
                if !profile.steps(&here, nc, run) {
                    continue;
                }
                let c = cost + run;
                if c <= max && c < best.get(&next).copied().unwrap_or(f32::MAX) {
                    best.insert(next, c);
                    open.push(Open { cost: c, cell: next });
                }
            }
        }
        None
    }

    /// The cells reachable from `from` within `max` metres of walking, nearest first (for picking
    /// somewhere to wander to).
    pub fn reachable(&self, from: Vec2, profile: &Profile, max: f32, limit: usize) -> Vec<IVec2> {
        let start = cell_at(from);
        let mut out = Vec::new();
        if self.cell(start).is_none() {
            return out;
        }
        let mut seen: AHashSet<IVec2> = AHashSet::new();
        let mut queue = VecDeque::new();
        seen.insert(start);
        queue.push_back(start);
        while let Some(c) = queue.pop_front() {
            out.push(c);
            if out.len() >= limit {
                break;
            }
            let Some(here) = self.cell(c).copied() else { continue };
            for (d, run) in NEIGHBOURS {
                let n = c + d;
                if centre(n).distance(from) > max || !seen.insert(n) {
                    continue;
                }
                if self.cell(n).is_some_and(|nc| profile.steps(&here, nc, run)) {
                    queue.push_back(n);
                }
            }
        }
        out
    }
}

/// Biome facts are looked up once per block of this many cells a side (they change over tens of
/// metres), and cave mouths checked at the blocks' corners.
const BLOCK: i32 = 4;

/// Build tile `t`'s cells, with the obstacles of it and its neighbours' columns in `obstacles`.
fn build_tile(ground: &Ground, obstacles: &AHashMap<IVec2, Vec<Obstacle>>, t: IVec2) -> Box<[Cell]> {
    let world = ground.world;
    let wg = &world.worldgen;
    let nature = Nature::builtin();
    let grass = nature.kind("grass");
    let shrub = nature.kind("shrub");
    let lo = (t * TILE).as_vec2();
    // Everything within reach of the tile's cells' clearance.
    let mut near: Vec<Obstacle> = Vec::new();
    for dz in -1..=1 {
        for dx in -1..=1 {
            near.extend(obstacles.get(&(t + IVec2::new(dx, dz))).into_iter().flatten().filter(|o| {
                let d = (o.at - (lo + TILE as f32 * 0.5)).abs() - TILE as f32 * 0.5;
                d.max_element() < MAX_CLEAR + o.radius
            }));
        }
    }
    // Cave mouths break the height field: where one is near a block, every cell of the block finds
    // its surface by raycast.
    let corners = TILE / BLOCK + 1;
    let mouth: Vec<bool> = (0..corners * corners)
        .map(|k| {
            let p = lo + Vec2::new((k % corners) as f32, (k / corners) as f32) * BLOCK as f32;
            let base = Vec3::new(p.x, wg.height(p.x, p.y), p.y);
            wg.sample(base + Vec3::Y * 0.25).0 <= 0.0 || wg.sample(base - Vec3::Y * 0.25).0 >= 0.0
        })
        .collect();
    let blocks = TILE / BLOCK;
    let facts: Vec<asc_world::Ground> = (0..blocks * blocks)
        .map(|k| {
            let p = lo + (Vec2::new((k % blocks) as f32, (k / blocks) as f32) + 0.5) * BLOCK as f32;
            wg.ground(p.x, p.y)
        })
        .collect();
    let solid = |q: Vec3| world.density_at(q).min(ground.structures.sdf(q));
    let mut cells = vec![Cell { height: 0.0, water: 0, clear: 0, flags: 0, rough: 0 }; CELLS].into_boxed_slice();
    let mut walls: Vec<bool> = vec![false; CELLS];
    for j in 0..TILE {
        for i in 0..TILE {
            let k = (j * TILE + i) as usize;
            let p = lo + Vec2::new(i as f32 + 0.5, j as f32 + 0.5);
            let (bx, bz) = (i / BLOCK, j / BLOCK);
            let g = facts[(bz * blocks + bx) as usize];
            let mouthy = [(0, 0), (1, 0), (0, 1), (1, 1)].iter().any(|(dx, dz)| mouth[((bz + dz) * corners + bx + dx) as usize]);
            let mut height = wg.height(p.x, p.y);
            let mut flags = 0;
            let base = Vec3::new(p.x, height, p.y);
            let edited = world.edits.ops_near(base).next().is_some();
            let built = ground.structures.sdf(base + Vec3::Y) < 4.0;
            if edited || built || mouthy {
                // Find the real surface: down from above whatever has been piled or built.
                if let Some(hit) = raycast(solid, base + Vec3::Y * 12.0, -Vec3::Y, 40.0) {
                    height = hit.point.y;
                    if built && ground.structures.sdf(hit.point - Vec3::Y * 0.1) < 0.2 {
                        flags |= BUILT;
                    }
                }
                // Too low to stand under (a tunnel, an overhang, a ceiling).
                let feet = Vec3::new(p.x, height, p.y);
                if solid(feet + Vec3::Y * 1.0) < 0.0 || solid(feet + Vec3::Y * HEADROOM) < 0.0 {
                    walls[k] = true;
                }
            }
            if built && ground.structures.sdf(Vec3::new(p.x, height + 0.9, p.y)) < 0.35 {
                // A wall standing on this cell.
                walls[k] = true;
            }
            // Water (or lava) over the surface: where nothing has disturbed it, the natural water
            // level; where the fluids have been moving, what they hold now.
            let mut level = wg.water_level(p.x, p.y);
            let disturbed = [0.3, 1.3].iter().any(|dy| {
                let c = asc_world::fluid::cell_of(Vec3::new(p.x, height + dy, p.y));
                ground.fluids.chunks.contains_key(&asc_world::fluid::split(c).0)
            });
            if disturbed {
                level = None;
                for dy in [0.3, 0.8, 1.5] {
                    let q = Vec3::new(p.x, height + dy, p.y);
                    match ground.fluids.fraction_at(wg, q) {
                        (FluidKind::Lava, fill) if fill > 0.05 => {
                            flags |= LAVA;
                            break;
                        }
                        (FluidKind::Water, fill) if fill > 0.05 => {
                            level = ground.fluids.surface_above(wg, q, 64);
                            break;
                        }
                        _ => {}
                    }
                }
            }
            let water = level.map_or(0, |l| ((l - height) * 10.0).clamp(0.0, 255.0) as u8);
            let b = Biome::ALL.iter().position(|x| *x == g.biome).unwrap_or(0);
            let browse = grass.map_or(0.0, |k| nature.kinds[k].density[b]) + shrub.map_or(0.0, |k| nature.kinds[k].density[b]) * 4.0;
            if browse * (1.0 - g.barren) > 0.15 && water < 3 && flags & BUILT == 0 {
                flags |= FORAGE;
            }
            if g.forest > 0.5 && matches!(g.biome, Biome::Forest | Biome::Taiga | Biome::Jungle | Biome::Swamp) {
                flags |= COVER;
            }
            // Heavy going underfoot (by the lie of the land: the surface material follows it).
            let rough = match g.biome {
                Biome::Arctic => 6,
                Biome::Desert | Biome::Shore => 4,
                Biome::Swamp => 7,
                _ if g.barren > 0.5 => 4,
                _ => 0,
            };
            cells[k] = Cell { height, water, clear: 0, flags, rough };
        }
    }
    // Clearance: to the nearest obstacle's edge, or the nearest wall cell (a chamfer pass over the
    // tile: walls in the neighbouring tiles aren't seen, a small error at the edges).
    let mut wall_d: Vec<f32> = walls.iter().map(|w| if *w { 0.0 } else { MAX_CLEAR }).collect();
    let n = TILE as usize;
    for pass in 0..2 {
        for step in 0..CELLS {
            let k = if pass == 0 { step } else { CELLS - 1 - step };
            let (i, j) = ((k % n) as i32, (k / n) as i32);
            let mut best = wall_d[k];
            for (dx, dz, w) in [(-1, 0, 1.0), (0, -1, 1.0), (-1, -1, 1.414), (1, -1, 1.414)] {
                let (dx, dz) = if pass == 0 { (dx, dz) } else { (-dx, -dz) };
                let (x, z) = (i + dx, j + dz);
                if x >= 0 && z >= 0 && x < TILE && z < TILE {
                    best = best.min(wall_d[(z * TILE + x) as usize] + w);
                }
            }
            wall_d[k] = best;
        }
    }
    for (k, cell) in cells.iter_mut().enumerate() {
        let p = lo + Vec2::new((k % n) as f32 + 0.5, (k / n) as f32 + 0.5);
        // Walls are half a cell either side of a wall cell's middle.
        let mut clear = if wall_d[k] == 0.0 { 0.0 } else { (wall_d[k] - 0.5).max(0.0) };
        for o in &near {
            clear = clear.min((o.at.distance(p) - o.radius).max(0.0));
        }
        if wall_d[k] == 0.0 {
            cell.flags |= BLOCKED;
        }
        cell.clear = (clear.min(MAX_CLEAR) * 4.0) as u8;
    }
    cells
}

/// The eight neighbours of a cell and how far each is./// The eight neighbours of a cell and how far each is.
pub const NEIGHBOURS: [(IVec2, f32); 8] = [
    (IVec2::new(1, 0), 1.0),
    (IVec2::new(-1, 0), 1.0),
    (IVec2::new(0, 1), 1.0),
    (IVec2::new(0, -1), 1.0),
    (IVec2::new(1, 1), std::f32::consts::SQRT_2),
    (IVec2::new(-1, 1), std::f32::consts::SQRT_2),
    (IVec2::new(1, -1), std::f32::consts::SQRT_2),
    (IVec2::new(-1, -1), std::f32::consts::SQRT_2),
];

#[derive(PartialEq)]
struct Open {
    cost: f32,
    cell: IVec2,
}

impl Eq for Open {}

impl Ord for Open {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Lowest cost first (BinaryHeap is a max-heap); ties by cell, so the order is fixed.
        other.cost.total_cmp(&self.cost).then_with(|| self.cell.to_array().cmp(&other.cell.to_array()))
    }
}

impl PartialOrd for Open {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// The trunks, rocks and ore nodes in `column` that still stand (as the client's scatter keeps
/// them: not dug out from under, not in a building's way, not mined).
pub fn obstacles_in(ground: &Ground, seed: u64, column: IVec2) -> Vec<Obstacle> {
    let world = ground.world;
    let wg = &world.worldgen;
    let nature = Nature::builtin();
    let radii = colliders();
    let stands = |p: Vec3| {
        if ground.structures.sdf(p + Vec3::Y * 0.5) < 1.0 {
            return false;
        }
        if world.edits.ops_near(p).next().is_some() {
            return world.density_at(p + Vec3::Y * 0.4) > 0.0 && world.density_at(p - Vec3::Y * 0.4) < 0.0;
        }
        true
    };
    let mut out = Vec::new();
    for (k, kind) in nature.kinds.iter().enumerate() {
        if kind.collide == Collide::None {
            continue;
        }
        for g in grow(nature, seed, wg, column, k) {
            let r = radii.nature[g.model as usize] * g.scale;
            if (kind.collide == Collide::Trunk || r > SMALL_ROCK) && stands(g.position) {
                out.push(Obstacle { at: Vec2::new(g.position.x, g.position.z), radius: r });
            }
        }
    }
    for n in crate::ore_nodes::surface(seed, wg, column) {
        if !ground.mined.contains(&crate::ore_nodes::node_id(n.position).to_array()) && stands(n.position) {
            let r = radii.ores.get(n.ore).copied().unwrap_or(0.5) * n.scale;
            out.push(Obstacle { at: Vec2::new(n.position.x, n.position.z), radius: r });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use asc_build::Lattice;

    pub(crate) struct Fixture {
        pub world: World,
        pub structures: Structures,
        pub fluids: Fluids,
        pub mined: BTreeSet<[i32; 3]>,
    }

    impl Fixture {
        pub fn new() -> Self {
            Self {
                world: World::new(1337),
                structures: Structures::new(Lattice::new(
                    &asc_build::BuildingSpec::parse(include_str!("../../../assets/data/building.toml")).unwrap(),
                )),
                fluids: Fluids::default(),
                mined: BTreeSet::new(),
            }
        }

        pub fn ground(&self) -> Ground<'_> {
            Ground { world: &self.world, structures: &self.structures, fluids: &self.fluids, mined: &self.mined }
        }

        /// A grid with the tiles round `p` built.
        pub fn grid(&self, p: Vec2, radius: f32) -> NavGrid {
            let mut grid = NavGrid::new(1337);
            grid.want(p, radius);
            grid.build(&self.ground(), p, usize::MAX);
            grid
        }
    }

    pub(crate) const HORSE: Profile = Profile { radius: 0.6, max_slope: 0.8, step: 0.5, water: 0.9, swims: false };
    pub(crate) const FOX: Profile = Profile { radius: 0.25, max_slope: 1.0, step: 0.4, water: 0.3, swims: false };

    /// A dry, gentle spot (the spawn point's search).
    fn dry(world: &World) -> Vec2 {
        let p = crate::rules::spawn_point(world);
        Vec2::new(p.x, p.z)
    }

    #[test]
    fn cells_sit_on_the_ground() {
        let f = Fixture::new();
        let p = dry(&f.world);
        let grid = f.grid(p, 20.0);
        let mut checked = 0;
        for dz in -15..15 {
            for dx in -15..15 {
                let c = cell_at(p) + IVec2::new(dx, dz);
                let cell = grid.cell(c).expect("built");
                let q = centre(c);
                let h = f.world.worldgen.height(q.x, q.y);
                assert!((cell.height - h).abs() < 0.3, "cell {c} at {} but the ground is {h}", cell.height);
                checked += 1;
            }
        }
        assert_eq!(checked, 900);
        let bilinear = grid.height_at(p + Vec2::new(0.3, 0.7)).unwrap();
        assert!((bilinear - f.world.worldgen.height(p.x + 0.3, p.y + 0.7)).abs() < 0.3);
    }

    #[test]
    fn trunks_keep_animals_off() {
        let f = Fixture::new();
        let nature = Nature::builtin();
        let tree = nature.kind("tree").unwrap();
        let seed = crate::ore_nodes::seed_of(1337);
        // A column with trees in it.
        let (column, trunk) = (0..400)
            .find_map(|i| {
                let c = IVec2::new(i % 20 - 10, i / 20 - 10);
                grow(nature, seed, &f.world.worldgen, c, tree).first().map(|g| (c, g.position))
            })
            .expect("a tree somewhere");
        let at = Vec2::new(trunk.x, trunk.z);
        let grid = f.grid(at, 4.0);
        let cell = grid.cell(cell_at(at)).unwrap();
        assert!(cell.clearance() < 0.6, "a trunk's own cell is clear by {} (column {column})", cell.clearance());
        assert!(!HORSE.fits(cell));
        assert!(!grid.line_clear(at - Vec2::X * 3.0, at + Vec2::X * 3.0, &HORSE) || cell.clearance() >= HORSE.radius);
    }

    #[test]
    fn a_dig_changes_the_tile_after_rebuilding() {
        let mut f = Fixture::new();
        let p = dry(&f.world);
        let mut grid = f.grid(p, 2.0);
        let before = grid.cell(cell_at(p)).unwrap().height;
        let op = asc_world::EditOp {
            center: Vec3::new(p.x, before, p.y),
            radius: 3.0,
            kind: asc_world::EditKind::Dig,
            material: asc_world::Material::Air,
        };
        f.world.edits.push(op);
        grid.invalidate(p, 3.0);
        assert_eq!(grid.cell(cell_at(p)).unwrap().height, before, "stale until rebuilt");
        grid.build(&f.ground(), p, usize::MAX);
        let after = grid.cell(cell_at(p)).unwrap().height;
        assert!(after < before - 1.5, "dug from {before} to {after}");
    }

    #[test]
    fn a_gap_a_fox_fits_through_keeps_a_horse_out() {
        // Two obstacles 1.2 m apart (edge to edge): a fox (0.5 m wide) fits, a horse (1.2) doesn't.
        let cell = |clear: f32| Cell { height: 0.0, water: 0, clear: (clear * 4.0) as u8, flags: 0, rough: 0 };
        let gap = cell(0.5);
        assert!(FOX.fits(&gap) && !HORSE.fits(&gap));
        // Steps: a ledge the horse climbs, a cliff it doesn't.
        let low = cell(4.0);
        let ledge = Cell { height: 0.45, ..low };
        let cliff = Cell { height: 2.5, ..low };
        assert!(HORSE.steps(&low, &ledge, 1.0));
        assert!(!HORSE.steps(&low, &cliff, 1.0));
    }

    #[test]
    fn water_is_as_deep_as_it_is() {
        let f = Fixture::new();
        let wg = &f.world.worldgen;
        // Find sea a few metres deep.
        let mut spot = None;
        'search: for r in 0..200 {
            for a in 0..16 {
                let ang = a as f32 / 16.0 * std::f32::consts::TAU;
                let p = Vec2::from_angle(ang) * r as f32 * 20.0;
                let g = wg.ground(p.x, p.y);
                if let Some(l) = wg.water_level(p.x, p.y)
                    && l - g.height > 3.0
                    && l - g.height < 12.0
                {
                    spot = Some((p, l - g.height));
                    break 'search;
                }
            }
        }
        let (p, depth) = spot.expect("some water");
        let grid = f.grid(p, 1.0);
        let c = cell_at(p);
        let cell = grid.cell(c).unwrap();
        let q = centre(c);
        let expected = wg.water_level(q.x, q.y).unwrap() - cell.height;
        assert!((cell.depth() - expected).abs() < 0.6, "depth {} vs {expected} (about {depth})", cell.depth());
        assert!(!HORSE.fits(cell), "a horse doesn't wade {} m", cell.depth());
        let fish = Profile { radius: 0.1, max_slope: 9.0, step: 9.0, water: 1.5, swims: true };
        assert!(fish.fits(cell));
    }

    #[test]
    fn building_tiles_is_quick_enough() {
        let f = Fixture::new();
        let p = dry(&f.world);
        let mut grid = NavGrid::new(1337);
        grid.want(p, 48.0);
        let start = std::time::Instant::now();
        grid.build(&f.ground(), p, usize::MAX);
        let (built, _) = grid.take_built();
        let each = start.elapsed().as_secs_f32() * 1000.0 / built as f32;
        println!("{built} tiles, {each:.2} ms each (wall clock, side by side)");
        // Release builds: a few ms a tile, its share of the columns' trees included (a debug build
        // is far slower: don't hold it to this; nor other tests sharing the cores).
        if !cfg!(debug_assertions) {
            assert!(each < 100.0, "{each:.2} ms a tile");
        }
    }
}
