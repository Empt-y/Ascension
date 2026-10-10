//! Chunk coordinates and padded sample storage.

use glam::{IVec3, Vec3};

use crate::edit::EditOp;
use crate::worldgen::WorldGen;

/// Cells per chunk edge. One cell is one metre.
pub const CHUNK: i32 = 32;
/// Samples per chunk edge: the chunk's own `CHUNK + 1` corners plus one on the low side and one more on
/// the high side, so the mesher can build every cell touching the chunk without looking at neighbours.
pub const SAMPLES: usize = CHUNK as usize + 3;
const SAMPLES_I: i32 = SAMPLES as i32;

pub type ChunkCoord = IVec3;

/// World-space position of a chunk's minimum corner.
pub fn chunk_origin(coord: ChunkCoord) -> Vec3 {
    (coord * CHUNK).as_vec3()
}

/// The chunk containing a world-space point.
pub fn chunk_of(p: Vec3) -> ChunkCoord {
    (p / CHUNK as f32).floor().as_ivec3()
}

/// Whether every sample is on one side of the surface, so there is nothing to mesh.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Uniform {
    Air,
    Solid,
}

pub struct ChunkData {
    pub coord: ChunkCoord,
    /// `SAMPLES³` signed distances, x fastest. Sample `i` sits at `origin + (i - 1)`.
    pub density: Vec<f32>,
    pub material: Vec<u8>,
    /// Worldgen surface height of each sample column (`SAMPLES²`, x fastest). Edits don't change it;
    /// the mesher uses it to darken ambient light under the original surface (caves, overhangs).
    pub surface: Vec<f32>,
    pub uniform: Option<Uniform>,
}

impl ChunkData {
    #[inline]
    pub fn index(x: usize, y: usize, z: usize) -> usize {
        x + SAMPLES * (y + SAMPLES * z)
    }

    /// World position of sample `(x, y, z)`.
    pub fn sample_pos(coord: ChunkCoord, x: usize, y: usize, z: usize) -> Vec3 {
        (coord * CHUNK + IVec3::new(x as i32, y as i32, z as i32) - IVec3::ONE).as_vec3()
    }

    /// World-space bounds (inclusive) of the samples this chunk stores.
    pub fn sample_bounds(coord: ChunkCoord) -> (Vec3, Vec3) {
        let lo = (coord * CHUNK - IVec3::ONE).as_vec3();
        (lo, lo + Vec3::splat((SAMPLES_I - 1) as f32))
    }

    pub fn build(worldgen: &WorldGen, coord: ChunkCoord, ops: &[EditOp]) -> Self {
        let mut chunk = worldgen.generate(coord);
        if !ops.is_empty() {
            for z in 0..SAMPLES {
                for y in 0..SAMPLES {
                    for x in 0..SAMPLES {
                        let p = Self::sample_pos(coord, x, y, z);
                        let i = Self::index(x, y, z);
                        let (mut d, mut m) = (chunk.density[i], chunk.material[i]);
                        for op in ops {
                            (d, m) = op.apply(p, d, m);
                        }
                        chunk.density[i] = d;
                        chunk.material[i] = m;
                    }
                }
            }
        }
        chunk.update_uniform();
        chunk
    }

    pub fn update_uniform(&mut self) {
        let solid = self.density.iter().filter(|d| **d < 0.0).count();
        self.uniform = if solid == 0 {
            Some(Uniform::Air)
        } else if solid == self.density.len() {
            Some(Uniform::Solid)
        } else {
            None
        };
    }
}
