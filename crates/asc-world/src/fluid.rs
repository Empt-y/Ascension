//! Fluids (water, later lava): where they are, as data shared by the server's simulation and the
//! client's drawing and swimming.
//!
//! Pristine until disturbed: worldgen defines where water naturally lies (the sea, for now), and
//! untouched water is never stored, so an ocean costs nothing. Only chunks that something has
//! disturbed (an edit near water, a flow arriving) are "materialised" into stored cells, which the
//! server then simulates. A chunk that isn't materialised reads as its pristine water.
//!
//! Cells sit on the terrain's sample points (cell `c` spans `c ± 0.5`), so a fluid chunk's solidity
//! comes straight from the terrain chunk's densities. Amounts are integers: `FULL` is a full cell;
//! deep water is very slightly compressed (up to `FULL + COMPRESS` per level of depth), which is
//! what lets it push upward through a U-bend.

use ahash::AHashMap;
use glam::{IVec3, Vec3};
use serde::{Deserialize, Serialize};

use crate::chunk::{CHUNK, ChunkCoord, ChunkData};
use crate::worldgen::{SEA_LEVEL, WorldGen};

/// A full cell.
pub const FULL: u16 = 1000;
/// Extra a cell may hold per full cell above it (pressure).
pub const COMPRESS: u16 = 20;
/// Cells per chunk.
pub const CELLS: usize = (CHUNK * CHUNK * CHUNK) as usize;

#[repr(u8)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FluidKind {
    #[default]
    None = 0,
    Water = 1,
    Lava = 2,
}

impl FluidKind {
    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Water,
            2 => Self::Lava,
            _ => Self::None,
        }
    }
}

/// Index of chunk-local cell (x, y, z), x fastest.
#[inline]
pub fn cell_index(x: usize, y: usize, z: usize) -> usize {
    x + CHUNK as usize * (y + CHUNK as usize * z)
}

/// The chunk holding cell `c`, and its local position.
#[inline]
pub fn split(c: IVec3) -> (ChunkCoord, IVec3) {
    let coord = c.div_euclid(IVec3::splat(CHUNK));
    (coord, c - coord * CHUNK)
}

/// The cell containing world point `p`.
pub fn cell_of(p: Vec3) -> IVec3 {
    (p + Vec3::splat(0.5)).floor().as_ivec3()
}

/// One materialised chunk of fluid cells.
#[derive(Clone, Debug, PartialEq)]
pub struct FluidChunk {
    pub kind: Vec<u8>,
    pub amount: Vec<u16>,
    /// Cells inside terrain or a building (from the terrain chunk's samples).
    pub solid: Vec<bool>,
}

impl FluidChunk {
    pub fn empty() -> Self {
        Self { kind: vec![0; CELLS], amount: vec![0; CELLS], solid: vec![false; CELLS] }
    }

    pub fn total(&self) -> u64 {
        self.amount.iter().map(|&a| a as u64).sum()
    }

    pub fn is_dry(&self) -> bool {
        self.amount.iter().all(|&a| a == 0)
    }
}

/// How much fluid natural water puts in a cell at height `y` (its centre), for a water surface at
/// `level`.
fn fill(y: i32, level: f32) -> u16 {
    let frac = (level - (y as f32 - 0.5)).clamp(0.0, 1.0);
    (frac * FULL as f32).round() as u16
}

/// Where water naturally lies: the surface level over column (x, z), if any. The sea fills every
/// column whose ground lies below sea level. (Lakes and rivers join this later.)
pub fn water_level(worldgen: &WorldGen, x: f32, z: f32) -> Option<f32> {
    (worldgen.height(x, z) < SEA_LEVEL).then_some(SEA_LEVEL)
}

/// Natural water in cell `c` (ignoring edits): open cells under a water surface, down to the
/// ground (caves under the sea floor stay dry until something opens them).
pub fn pristine(worldgen: &WorldGen, c: IVec3) -> (FluidKind, u16) {
    let (x, z) = (c.x as f32, c.z as f32);
    let Some(level) = water_level(worldgen, x, z) else { return (FluidKind::None, 0) };
    let amount = fill(c.y, level);
    if amount == 0 || c.y as f32 + 0.5 < worldgen.height(x, z) - 1.0 {
        return (FluidKind::None, 0);
    }
    if worldgen.sample(Vec3::new(x, c.y as f32, z)).0 <= 0.0 {
        return (FluidKind::None, 0);
    }
    (FluidKind::Water, amount)
}

/// Natural water in every cell of a chunk, given its built terrain (edits included: dug cells
/// below the sea stay dry until the simulation lets water in, filled cells are solid).
pub fn pristine_chunk(worldgen: &WorldGen, terrain: &ChunkData) -> FluidChunk {
    let mut out = FluidChunk::empty();
    let n = CHUNK as usize;
    let base = terrain.coord * CHUNK;
    for z in 0..n {
        for x in 0..n {
            let (wx, wz) = ((base.x + x as i32) as f32, (base.z + z as i32) as f32);
            let level = water_level(worldgen, wx, wz);
            // The natural ground: the first sample column of the padded chunk is one cell further out.
            let surface = terrain.surface[(x + 1) + crate::chunk::SAMPLES * (z + 1)];
            for y in 0..n {
                let i = cell_index(x, y, z);
                let s = ChunkData::index(x + 1, y + 1, z + 1);
                out.solid[i] = terrain.density[s] < 0.0;
                if out.solid[i] {
                    continue;
                }
                let wy = base.y + y as i32;
                if let Some(level) = level {
                    let a = fill(wy, level);
                    // Natural water only down to the natural ground, and only where the ground was
                    // naturally open (a pit dug below sea level stays dry: nothing has let water in).
                    let natural_air = worldgen.sample(Vec3::new(wx, wy as f32, wz)).0 > 0.0;
                    if a > 0 && wy as f32 + 0.5 >= surface - 1.0 && natural_air {
                        out.kind[i] = FluidKind::Water as u8;
                        out.amount[i] = a;
                    }
                }
            }
        }
    }
    out
}

/// Every materialised fluid chunk; everything else is pristine.
#[derive(Default)]
pub struct Fluids {
    pub chunks: AHashMap<ChunkCoord, FluidChunk>,
}

impl Fluids {
    /// The fluid in cell `c`: stored if its chunk is materialised, otherwise pristine.
    pub fn get(&self, worldgen: &WorldGen, c: IVec3) -> (FluidKind, u16) {
        let (coord, local) = split(c);
        match self.chunks.get(&coord) {
            Some(ch) => {
                let i = cell_index(local.x as usize, local.y as usize, local.z as usize);
                (FluidKind::from_u8(ch.kind[i]), ch.amount[i])
            }
            None => pristine(worldgen, c),
        }
    }

    /// How full of fluid the cell at world point `p` is, 0..1 (more than 1 when compressed).
    pub fn fraction_at(&self, worldgen: &WorldGen, p: Vec3) -> (FluidKind, f32) {
        let (k, a) = self.get(worldgen, cell_of(p));
        (k, a as f32 / FULL as f32)
    }

    /// The height of the fluid surface in the column at `p`, scanning up from it a few cells (for
    /// swimming: how deep is the player).
    pub fn surface_above(&self, worldgen: &WorldGen, p: Vec3, max_cells: i32) -> Option<f32> {
        let c = cell_of(p);
        let (k, a) = self.get(worldgen, c);
        if k == FluidKind::None || a == 0 {
            return None;
        }
        for dy in 0..max_cells {
            let cc = c + IVec3::Y * dy;
            let (_, a) = self.get(worldgen, cc);
            if a < FULL {
                return Some(cc.y as f32 - 0.5 + a as f32 / FULL as f32);
            }
        }
        Some(c.y as f32 + max_cells as f32)
    }
}

/// A fluid chunk packed for sending: runs of (kind, amount quantised to 0..=255) cells.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PackedFluid {
    /// (run length, kind, level 0..=255 where 255 is full or more).
    pub runs: Vec<(u16, u8, u8)>,
}

impl PackedFluid {
    pub fn pack(chunk: &FluidChunk) -> Self {
        let mut runs: Vec<(u16, u8, u8)> = Vec::new();
        for i in 0..CELLS {
            let level = ((chunk.amount[i].min(FULL) as u32 * 255 + FULL as u32 / 2) / FULL as u32) as u8;
            let kind = if level == 0 { 0 } else { chunk.kind[i] };
            match runs.last_mut() {
                Some((n, k, l)) if *k == kind && *l == level && *n < u16::MAX => *n += 1,
                _ => runs.push((1, kind, level)),
            }
        }
        Self { runs }
    }

    /// Unpack into kinds and 0..=255 levels per cell.
    pub fn unpack(&self) -> (Vec<u8>, Vec<u8>) {
        let mut kind = Vec::with_capacity(CELLS);
        let mut level = Vec::with_capacity(CELLS);
        for &(n, k, l) in &self.runs {
            for _ in 0..n {
                kind.push(k);
                level.push(l);
            }
        }
        kind.resize(CELLS, 0);
        level.resize(CELLS, 0);
        (kind, level)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_sea_fills_low_columns_up_to_sea_level() {
        let worldgen = WorldGen::new(1337);
        // Find a column under the sea.
        let (x, z) = (0..4000)
            .map(|i| ((i % 64) as f32 * 37.0 - 1200.0, (i / 64) as f32 * 41.0 - 1300.0))
            .find(|(x, z)| worldgen.height(*x, *z) < SEA_LEVEL - 4.0)
            .expect("some sea");
        let (xi, zi) = (x as i32, z as i32);
        let top = IVec3::new(xi, SEA_LEVEL.floor() as i32 + 2, zi);
        assert_eq!(pristine(&worldgen, top).1, 0, "dry above the sea");
        let deep = IVec3::new(xi, (worldgen.height(x, z) + 1.5).ceil() as i32, zi);
        assert_eq!(pristine(&worldgen, deep), (FluidKind::Water, FULL), "full under it");
        // A materialised chunk agrees with the pristine cells.
        let world = crate::World::new(1337);
        let coord = crate::chunk_of(Vec3::new(x, SEA_LEVEL - 1.0, z));
        let terrain = world.build_chunk(coord);
        let ch = pristine_chunk(&worldgen, &terrain);
        let mut checked = 0;
        for i in (0..CELLS).step_by(53) {
            if ch.solid[i] {
                continue;
            }
            let (lx, ly, lz) = (i % 32, (i / 32) % 32, i / 1024);
            let c = coord * CHUNK + IVec3::new(lx as i32, ly as i32, lz as i32);
            let (k, a) = pristine(&worldgen, c);
            // The column test uses the cell's own height; the chunk uses the sample column: allow
            // the odd mismatch right at the ground.
            if a != ch.amount[i] {
                assert!(c.y as f32 <= worldgen.height(c.x as f32, c.z as f32) + 1.5, "{c} {a} vs {}", ch.amount[i]);
            } else {
                assert_eq!(k as u8, ch.kind[i]);
                checked += 1;
            }
        }
        assert!(checked > 100);
    }

    #[test]
    fn packing_round_trips_levels() {
        let mut ch = FluidChunk::empty();
        for i in 0..CELLS {
            if i % 7 == 0 {
                ch.kind[i] = 1;
                ch.amount[i] = (i % 1001) as u16;
            }
        }
        let p = PackedFluid::pack(&ch);
        let (kind, level) = p.unpack();
        for i in 0..CELLS {
            let want = ((ch.amount[i] as u32 * 255 + 500) / 1000) as u8;
            assert_eq!(level[i], want);
            if want > 0 {
                assert_eq!(kind[i], ch.kind[i]);
            }
        }
        assert!(p.runs.len() < CELLS);
    }
}
