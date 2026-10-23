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
use crate::worldgen::WorldGen;
#[cfg(test)]
use crate::worldgen::SEA_LEVEL;

/// A full cell.
pub const FULL: u16 = 1000;
/// Extra a cell may hold per full cell above it (pressure).
pub const COMPRESS: u16 = 20;
/// Cells per chunk.
pub const CELLS: usize = (CHUNK * CHUNK * CHUNK) as usize;
/// Deep down, open cave space below this height is a sea of lava.
pub const LAVA_LEVEL: f32 = -160.0;

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

/// Where water naturally lies: the surface level over column (x, z), if any (sea, lake or river).
pub fn water_level(worldgen: &WorldGen, x: f32, z: f32) -> Option<f32> {
    worldgen.water_level(x, z)
}

/// Natural water in cell `c` (ignoring edits): open cells under a water surface, down to
/// `NATURAL_DEPTH` under the ground (caves deeper under the sea floor stay dry until something
/// opens them).
pub fn pristine(worldgen: &WorldGen, c: IVec3) -> (FluidKind, u16) {
    let (x, z) = (c.x as f32, c.z as f32);
    if (c.y as f32) < LAVA_LEVEL + 0.5 {
        let a = fill(c.y, LAVA_LEVEL);
        if a > 0 && worldgen.sample(Vec3::new(x, c.y as f32, z)).0 > 0.0 {
            return (FluidKind::Lava, a);
        }
        return (FluidKind::None, 0);
    }
    let Some(level) = water_level(worldgen, x, z) else { return (FluidKind::None, 0) };
    let amount = fill(c.y, level);
    if amount == 0 || c.y as f32 + 0.5 < worldgen.height(x, z) - NATURAL_DEPTH {
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
                if (wy as f32) < LAVA_LEVEL + 0.5 {
                    // The deep lava sea: every open cell below its level (from the built terrain:
                    // tunnels dug down there fill too).
                    let a = fill(wy, LAVA_LEVEL);
                    if a > 0 {
                        out.kind[i] = FluidKind::Lava as u8;
                        out.amount[i] = a;
                    }
                    continue;
                }
                if let Some(level) = level {
                    let a = fill(wy, level);
                    // Natural water only down to the natural ground, and only where the ground was
                    // naturally open (a pit dug below sea level stays dry: nothing has let water in).
                    let natural_air = worldgen.sample(Vec3::new(wx, wy as f32, wz)).0 > 0.0;
                    if a > 0 && wy as f32 + 0.5 >= surface - NATURAL_DEPTH && natural_air {
                        out.kind[i] = FluidKind::Water as u8;
                        out.amount[i] = a;
                    }
                }
            }
        }
    }
    out
}

/// Vertex bytes of a fluid mesh (in `ore`): the river's flow x and z around 128, `FLOW_SCALE` per
/// m/s; a flag (`FLAG_FLOW` with a flow, `FLAG_DEPTH` without); and the water's depth over the
/// ground, `(depth + DEPTH_OFFSET) * DEPTH_SCALE`. Shading fades shallow water out, so shores
/// follow the depth's smooth zero line rather than where two meshes cross.
pub const FLOW_SCALE: f32 = 60.0;
pub const FLAG_FLOW: u8 = 255;
pub const FLAG_DEPTH: u8 = 1;
pub const DEPTH_SCALE: f32 = 50.0;
pub const DEPTH_OFFSET: f32 = 0.5;
/// Natural water fills open cells this far below the natural ground (caves deeper stay dry).
pub const NATURAL_DEPTH: f32 = 8.0;
/// Columns of a fluid mesh's surface grid: the padded chunk's and `MARGIN` more all round.
pub const MARGIN: usize = 2;
pub const RING: usize = crate::chunk::SAMPLES + 2 * MARGIN;

/// What a fluid mesh is built from.
pub struct FluidInput<'a> {
    /// The built terrain (edits included): what is solid, and where the ground is.
    pub terrain: &'a ChunkData,
    /// Fluid kind and amount of every padded sample (`SAMPLES³`, as `ChunkData::index`).
    pub kind: &'a [u8],
    pub amount: &'a [u16],
    /// The water surface over each column of the padded chunk and the ring round it (`RING²`,
    /// padded column x at x + MARGIN), NaN where there's none.
    pub surface: &'a [f32],
    /// The river's flow per padded column (`SAMPLES²`), for natural water.
    pub flow: Option<&'a [glam::Vec2]>,
}

/// The water surface over a column, from its cells: the top of the highest water cell at or
/// below `from` (scanning down to `to`), followed up while the water continues.
pub fn column_surface(get: impl Fn(i32) -> (FluidKind, u16), from: i32, to: i32) -> f32 {
    let water = |y: i32| {
        let (k, a) = get(y);
        (k == FluidKind::Water && a > 0).then_some(a)
    };
    let Some(mut y) = (to..=from).rev().find(|y| water(*y).is_some()) else { return f32::NAN };
    while y < from + 64 && water(y + 1).is_some() {
        y += 1;
    }
    y as f32 - 0.5 + water(y).unwrap_or(0).min(FULL) as f32 / FULL as f32
}

/// Mesh a chunk's fluid. Each column's water surface is smoothed with its neighbours (so
/// simulated water on a slope doesn't settle into 1 m stairs), and the water runs on into the
/// ground under it, down and sideways into the banks, so its edges are buried and the terrain cuts
/// the shore. Every vertex carries the depth (and, for rivers, the flow).
pub fn mesh_fluid(input: &FluidInput) -> crate::ChunkMesh {
    use crate::chunk::SAMPLES;
    let terrain = input.terrain;
    let base = ChunkData::sample_pos(terrain.coord, 0, 0, 0);
    let ring = |x: usize, z: usize| input.surface[x + RING * z];
    // Smoothed surfaces of the padded columns (with their neighbours); dry columns up to two
    // from water take it on, so its buried edge ends well inside the bank (one column in, the end
    // of it could show where the bank is only a little higher than the water).
    let mut surface = vec![f32::NAN; SAMPLES * SAMPLES];
    for z in 0..SAMPLES {
        for x in 0..SAMPLES {
            let here = ring(x + MARGIN, z + MARGIN);
            let reach = if here.is_nan() { 2 } else { 1 };
            let (mut sum, mut n) = (0.0, 0);
            for dz in MARGIN - reach..=MARGIN + reach {
                for dx in MARGIN - reach..=MARGIN + reach {
                    let s = ring(x + dx, z + dz);
                    if !s.is_nan() && (here.is_nan() || (s - here).abs() < 1.5) {
                        sum += s;
                        n += 1;
                    }
                }
            }
            if n > 0 {
                surface[x + SAMPLES * z] = sum / n as f32;
            }
        }
    }
    // The ground under each column's water: going down through open cells from a little over the
    // surface (a bank's ground is just above it), the first solid. If the column is solid there
    // already (the ground is higher, or this chunk is buried) or open all the way down, the
    // natural ground, which is then above or below the chunk: never the chunk's own top face or a
    // buried cave's floor, or buried chunks under water would fill their cells.
    let mut ground = vec![f32::NAN; SAMPLES * SAMPLES];
    // Whether the water reaches that ground (water in the open cell over it, or a bank at about
    // the water's height): only then is the ground under it filled.
    let mut reached = vec![false; SAMPLES * SAMPLES];
    for z in 0..SAMPLES {
        for x in 0..SAMPLES {
            let s = surface[x + SAMPLES * z];
            if s.is_nan() {
                continue;
            }
            let d = |y: usize| terrain.density[ChunkData::index(x, y, z)];
            let mut g = terrain.surface[x + SAMPLES * z];
            let start = ((s + 3.0 - base.y).floor() as i32).min(SAMPLES as i32 - 1);
            if start >= 1 && d(start as usize) >= 0.0 {
                for y in (1..=start as usize).rev() {
                    let (lo, hi) = (d(y - 1), d(y));
                    if lo < 0.0 {
                        g = base.y + (y - 1) as f32 + lo / (lo - hi);
                        let over = ChunkData::index(x, y, z);
                        reached[x + SAMPLES * z] = input.kind[over] == FluidKind::Water as u8 || g > s - 0.5;
                        break;
                    }
                }
            }
            ground[x + SAMPLES * z] = g;
        }
    }

    let n = SAMPLES * SAMPLES * SAMPLES;
    let mut density = vec![0.5f32; n];
    let mut material = vec![1u8; n];
    for z in 0..SAMPLES {
        for x in 0..SAMPLES {
            let column = x + SAMPLES * z;
            let (s, g, reached) = (surface[column], ground[column], reached[column]);
            let mut below_filled = false;
            for y in 0..SAMPLES {
                let i = ChunkData::index(x, y, z);
                let wy = base.y + y as f32;
                let (kind, amount) = (FluidKind::from_u8(input.kind[i]), input.amount[i].min(FULL));
                let f = if kind == FluidKind::Lava {
                    material[i] = 2;
                    amount as f32 / FULL as f32
                } else if s.is_nan() || wy - 0.5 >= s {
                    if kind == FluidKind::Water { amount as f32 / FULL as f32 } else { 0.0 }
                } else {
                    let top = (s - (wy - 0.5)).clamp(0.0, 1.0);
                    let solid = terrain.density[i] < 0.0;
                    if kind == FluidKind::Water {
                        if wy + 0.5 <= s - 1.0 { 1.0 } else { top }
                    } else if solid && reached && wy + 0.5 >= g - 1.5 && wy - 0.5 < g {
                        // Just under the ground the water reaches: its edges are buried there.
                        top
                    } else if !solid && below_filled {
                        // Open over water or the buried edge, under the (smoothed) surface: water
                        // too, or a thin sheet would lie on the ground under the real surface.
                        top
                    } else {
                        0.0
                    }
                };
                below_filled = f > 0.999;
                density[i] = 0.5 - f;
            }
        }
    }
    let mut field = ChunkData {
        coord: terrain.coord,
        density,
        material,
        // Open sky everywhere: water isn't darkened as if buried.
        surface: vec![1e9; SAMPLES * SAMPLES],
        water: Vec::new(),
        uniform: None,
    };
    field.update_uniform();
    let mut mesh = crate::mesh_chunk(&field);
    // Per-vertex depth (and flow), blended over the columns round the vertex that have water.
    let last = (SAMPLES - 1) as f32;
    for v in &mut mesh.vertices {
        // Sample i sits at local i - 1.
        let (fx, fz) = ((v.position[0] + 1.0).clamp(0.0, last), (v.position[2] + 1.0).clamp(0.0, last));
        let (x0, z0) = (fx.floor() as usize, fz.floor() as usize);
        let (tx, tz) = (fx - x0 as f32, fz - z0 as f32);
        let (mut depth, mut flow, mut weight) = (0.0, glam::Vec2::ZERO, 0.0);
        let corners = [(0, 0, (1.0 - tx) * (1.0 - tz)), (1, 0, tx * (1.0 - tz)), (0, 1, (1.0 - tx) * tz), (1, 1, tx * tz)];
        for (dx, dz, w) in corners {
            let c = (x0 + dx).min(SAMPLES - 1) + SAMPLES * (z0 + dz).min(SAMPLES - 1);
            if w <= 0.0 {
                continue;
            }
            // Columns with no water near them count as dry: the faces where the water's buried
            // edge ends, next to them, fade out with the shore.
            depth += (if surface[c].is_nan() { -1.0 } else { surface[c] - ground[c] }) * w;
            if let Some(f) = input.flow {
                flow += f[c] * w;
            }
            weight += w;
        }
        let (depth, flow) = if weight > 0.0 { (depth / weight, flow / weight) } else { (4.0, glam::Vec2::ZERO) };
        let byte = |v: f32| (128.0 + v * FLOW_SCALE).round().clamp(1.0, 255.0) as u8;
        let flag = if input.flow.is_some() { FLAG_FLOW } else { FLAG_DEPTH };
        let depth = ((depth + DEPTH_OFFSET) * DEPTH_SCALE).round().clamp(0.0, 255.0) as u8;
        v.ore = [byte(flow.x), byte(flow.y), flag, depth];
    }
    mesh
}

/// The natural fluid of every padded sample of a chunk, from its unedited samples.
pub fn natural_cells(natural: &ChunkData) -> (Vec<u8>, Vec<u16>) {
    use crate::chunk::SAMPLES;
    let n = SAMPLES * SAMPLES * SAMPLES;
    let base = ChunkData::sample_pos(natural.coord, 0, 0, 0);
    let (mut kind, mut amount) = (vec![0u8; n], vec![0u16; n]);
    for z in 0..SAMPLES {
        for x in 0..SAMPLES {
            let column = x + SAMPLES * z;
            let (level, ground) = (natural.water[column], natural.surface[column]);
            for y in 0..SAMPLES {
                let i = ChunkData::index(x, y, z);
                if natural.density[i] < 0.0 {
                    continue;
                }
                let wy = base.y as i32 + y as i32;
                let (k, a) = if (wy as f32) < LAVA_LEVEL + 0.5 {
                    (FluidKind::Lava, fill(wy, LAVA_LEVEL))
                } else if !level.is_nan() && wy as f32 + 0.5 >= ground - NATURAL_DEPTH {
                    (FluidKind::Water, fill(wy, level))
                } else {
                    continue;
                };
                if a > 0 {
                    kind[i] = k as u8;
                    amount[i] = a;
                }
            }
        }
    }
    (kind, amount)
}

/// The mesh of a chunk's natural water and lava, for chunks whose fluid isn't materialised:
/// `natural` is its unedited samples (where the water is), `terrain` the built ones.
pub fn natural_mesh(worldgen: &WorldGen, natural: &ChunkData, terrain: &ChunkData) -> crate::ChunkMesh {
    use crate::chunk::SAMPLES;
    let base = ChunkData::sample_pos(natural.coord, 0, 0, 0);
    let lava_reach = base.y < LAVA_LEVEL + 1.0;
    let water_reach = natural.water.iter().any(|l| !l.is_nan() && *l >= base.y - 0.5);
    if !lava_reach && !water_reach {
        return crate::ChunkMesh::default();
    }
    let (kind, amount) = natural_cells(natural);
    let mut surface = vec![f32::NAN; RING * RING];
    for z in 0..RING {
        for x in 0..RING {
            let inside = MARGIN..SAMPLES + MARGIN;
            surface[x + RING * z] = if inside.contains(&x) && inside.contains(&z) {
                natural.water[(x - MARGIN) + SAMPLES * (z - MARGIN)]
            } else {
                let (wx, wz) = (base.x + x as f32 - MARGIN as f32, base.z + z as f32 - MARGIN as f32);
                worldgen.water_level(wx, wz).unwrap_or(f32::NAN)
            };
        }
    }
    // Rivers only: lakes and the sea are still.
    let flow: Vec<glam::Vec2> = (0..SAMPLES * SAMPLES)
        .map(|c| {
            if natural.water[c].is_nan() {
                return glam::Vec2::ZERO;
            }
            worldgen.water_flow(base.x + (c % SAMPLES) as f32, base.z + (c / SAMPLES) as f32)
        })
        .collect();
    mesh_fluid(&FluidInput { terrain, kind: &kind, amount: &amount, surface: &surface, flow: Some(&flow) })
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
    fn chunks_buried_under_water_have_no_water_mesh() {
        let worldgen = WorldGen::new(1337);
        let mut checked = 0;
        for i in 0..4000 {
            let (x, z) = ((i % 64) as f32 * 37.0 - 1200.0, (i / 64) as f32 * 41.0 - 1300.0);
            let (h, level) = worldgen.height_and_water(x, z);
            if level.is_none() {
                continue;
            }
            // A chunk whose top is well under the bed (deeper than natural water reaches into caves).
            let coord = crate::chunk_of(Vec3::new(x, h - NATURAL_DEPTH - 40.0, z));
            if ((coord.y + 1) * CHUNK) as f32 > h - NATURAL_DEPTH - 2.0 {
                continue;
            }
            let natural = worldgen.generate(coord);
            let mesh = natural_mesh(&worldgen, &natural, &natural);
            let water = mesh.vertices.iter().filter(|v| v.weights[0] > 128).count();
            assert_eq!(water, 0, "water inside the ground under ({x}, {z}), chunk {coord}");
            checked += 1;
            if checked >= 12 {
                break;
            }
        }
        assert!(checked >= 6, "only {checked} buried chunks found");
    }

    #[test]
    fn natural_water_meshes_at_its_surface_with_the_river_flow() {
        let worldgen = WorldGen::new(1337);
        // A river column (water above sea level, with a flow).
        let (x, z, level) = (0..40000)
            .map(|i| ((i % 200) as f32 * 7.0 - 700.0, (i / 200) as f32 * 7.0 - 700.0))
            .find_map(|(x, z)| {
                let l = worldgen.water_level(x, z)?;
                (l > SEA_LEVEL + 2.0 && worldgen.water_flow(x, z).length() > 0.1).then_some((x, z, l))
            })
            .expect("a river");
        let coord = crate::chunk_of(Vec3::new(x, level, z));
        let natural = worldgen.generate(coord);
        let mesh = natural_mesh(&worldgen, &natural, &natural);
        assert!(!mesh.is_empty());
        let origin = (coord * CHUNK).as_vec3();
        // The top surface over the river column sits at the water level.
        let top = mesh
            .vertices
            .iter()
            .filter(|v| (origin.x + v.position[0] - x).abs() < 1.0 && (origin.z + v.position[2] - z).abs() < 1.0)
            .map(|v| origin.y + v.position[1])
            .fold(f32::MIN, f32::max);
        assert!((top - level).abs() < 0.6, "surface {top} vs level {level}");
        let flowing = mesh.vertices.iter().filter(|v| v.ore[2] == FLAG_FLOW && (v.ore[0] != 128 || v.ore[1] != 128));
        assert!(flowing.count() > 0, "river vertices carry the flow");
        // A chunk high above everything has none.
        let sky_chunk = worldgen.generate(coord + IVec3::Y * 6);
        let sky = natural_mesh(&worldgen, &sky_chunk, &sky_chunk);
        assert!(sky.is_empty());
    }

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
