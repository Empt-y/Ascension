//! The world as data: a signed-distance density field split into chunks.
//!
//! Density is a signed distance in metres: negative inside solid ground, positive in air. Chunks are
//! generated deterministically from [`WorldGen`] and then the [`EditLog`] is replayed over them, so a
//! chunk can always be rebuilt from scratch and only its mesh needs to stay resident.

mod chunk;
mod collide;
mod edit;
pub mod fluid;
mod material;
mod mesh;
pub mod ores;
mod raycast;
pub mod volcano;
pub mod weather;
mod worldgen;

pub use chunk::{CHUNK, ChunkCoord, ChunkData, SAMPLES, chunk_of, chunk_origin};
pub use collide::{Capsule, Contact, box_distance, distance_estimate, push_sphere, resolve_capsule};
pub use edit::{EditKind, EditLog, EditOp};
pub use material::{Material, ORE_BASE, ore_byte, ore_index};
pub use mesh::{ChunkMesh, MATERIAL_SLOTS, POSITION_GRID, Vertex, mesh_chunk};
pub use raycast::{RayHit, raycast};
pub use worldgen::{Biome, Ground, SEA_LEVEL, WorldGen};

use std::sync::Arc;

/// Bumped whenever worldgen changes what a seed produces, so saves can warn about mismatched terrain.
pub const WORLDGEN_VERSION: u32 = 7;

use glam::Vec3;

/// Volume (m³) of each material, and of air, inside a sphere.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Census {
    /// By material; `Material::Ore` is the total of every ore.
    pub solid: [f32; Material::COUNT],
    /// By ore index (as long as the highest ore found).
    pub ores: Vec<f32>,
    pub air: f32,
}

impl Census {
    pub fn total_solid(&self) -> f32 {
        self.solid.iter().sum()
    }
}

/// Worldgen plus the edits made on top of it. Worker threads share the generator through the `Arc`.
pub struct World {
    pub worldgen: Arc<WorldGen>,
    pub edits: EditLog,
}

impl World {
    pub fn new(seed: i32) -> Self {
        Self { worldgen: Arc::new(WorldGen::new(seed)), edits: EditLog::default() }
    }

    /// Density at an arbitrary point, with edits applied. Used for raycasts, collision and queries.
    pub fn density_at(&self, p: Vec3) -> f32 {
        self.sample_at(p).0
    }

    /// Density and material at an arbitrary point, with edits applied.
    pub fn sample_at(&self, p: Vec3) -> (f32, Material) {
        let (d, m) = self.sample_raw(p);
        (d, Material::from_u8(m))
    }

    /// Density and material byte (which tells ores apart) at a point, with edits applied.
    pub fn sample_raw(&self, p: Vec3) -> (f32, u8) {
        let (mut d, mut m) = self.worldgen.sample(p);
        for op in self.edits.ops_near(p) {
            (d, m) = op.apply(p, d, m);
        }
        (d, m)
    }

    /// What a sphere currently contains, sampled on a grid of `step` metres.
    pub fn census(&self, center: Vec3, radius: f32, step: f32) -> Census {
        let mut census = Census::default();
        let cell = step * step * step;
        let n = (radius / step).ceil() as i32;
        for z in -n..=n {
            for y in -n..=n {
                for x in -n..=n {
                    // A lattice centred on the sphere, so the estimate has no directional bias.
                    let offset = (Vec3::new(x as f32, y as f32, z as f32)) * step;
                    if offset.length() > radius {
                        continue;
                    }
                    let (d, byte) = self.sample_raw(center + offset);
                    if d < 0.0 {
                        census.solid[Material::from_u8(byte) as usize] += cell;
                        if let Some(ore) = ore_index(byte) {
                            if census.ores.len() <= ore {
                                census.ores.resize(ore + 1, 0.0);
                            }
                            census.ores[ore] += cell;
                        }
                    } else {
                        census.air += cell;
                    }
                }
            }
        }
        census
    }

    /// Build a chunk's samples: worldgen first, then every edit that touches it, in order.
    pub fn build_chunk(&self, coord: ChunkCoord) -> ChunkData {
        let ops: Vec<EditOp> = self.edits.ops_for(coord).copied().collect();
        ChunkData::build(&self.worldgen, coord, &ops)
    }
}
