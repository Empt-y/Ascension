//! The world as data: a signed-distance density field split into chunks.
//!
//! Density is a signed distance in metres: negative inside solid ground, positive in air. Chunks are
//! generated deterministically from [`WorldGen`] and then the [`EditLog`] is replayed over them, so a
//! chunk can always be rebuilt from scratch and only its mesh needs to stay resident.

mod chunk;
mod edit;
mod material;
mod mesh;
mod raycast;
mod worldgen;

pub use chunk::{CHUNK, ChunkCoord, ChunkData, SAMPLES, chunk_of, chunk_origin};
pub use edit::{EditKind, EditLog, EditOp};
pub use material::Material;
pub use mesh::{ChunkMesh, MATERIAL_SLOTS, Vertex, mesh_chunk};
pub use raycast::{RayHit, raycast};
pub use worldgen::WorldGen;

use std::sync::Arc;

use glam::Vec3;

/// Worldgen plus the edits made on top of it. Worker threads share the generator through the `Arc`.
pub struct World {
    pub worldgen: Arc<WorldGen>,
    pub edits: EditLog,
}

impl World {
    pub fn new(seed: i32) -> Self {
        Self { worldgen: Arc::new(WorldGen::new(seed)), edits: EditLog::default() }
    }

    /// Density at an arbitrary point, with edits applied. Used for raycasts and queries.
    pub fn density_at(&self, p: Vec3) -> f32 {
        let (mut d, mut m) = self.worldgen.sample(p);
        for op in self.edits.ops_near(p) {
            (d, m) = op.apply(p, d, m);
        }
        d
    }

    /// Build a chunk's samples: worldgen first, then every edit that touches it, in order.
    pub fn build_chunk(&self, coord: ChunkCoord) -> ChunkData {
        let ops: Vec<EditOp> = self.edits.ops_for(coord).copied().collect();
        ChunkData::build(&self.worldgen, coord, &ops)
    }
}
