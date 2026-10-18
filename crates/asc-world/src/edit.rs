//! Terrain edits, stored as an ordered log of CSG operations.
//!
//! Densities are clamped to `±DENSITY_CLAMP`, which makes every operation strictly local: a sphere can
//! only change samples within `radius + DENSITY_CLAMP` of its centre. That bound is what the per-chunk
//! index uses, so a chunk rebuilt from worldgen plus its indexed ops matches its neighbours exactly.

use ahash::AHashMap;
use glam::{IVec3, Vec3};
use serde::{Deserialize, Serialize};

use crate::chunk::{CHUNK, ChunkCoord, ChunkData, chunk_of};
use crate::material::Material;

/// Densities are clamped to this distance from the surface (metres).
pub const DENSITY_CLAMP: f32 = 4.0;

/// What an edit does. New kinds must only ever be added at the end: saves store the variant index.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum EditKind {
    /// Remove a sphere of `radius`.
    Dig,
    /// Fill a sphere of `radius` with `material`.
    Fill,
    /// Remove a box: `radius` is its horizontal half-width (square), `half_height` its vertical.
    DigBox { half_height: f32 },
    /// Fill a box (same shape as `DigBox`) with `material`.
    FillBox { half_height: f32 },
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct EditOp {
    pub center: Vec3,
    pub radius: f32,
    pub kind: EditKind,
    pub material: Material,
}

impl EditOp {
    /// Distance from the centre beyond which this op changes nothing.
    pub fn reach(&self) -> f32 {
        let extent = match self.kind {
            EditKind::Dig | EditKind::Fill => self.radius,
            EditKind::DigBox { half_height } | EditKind::FillBox { half_height } => {
                Vec3::new(self.radius, half_height, self.radius).length()
            }
        };
        extent + DENSITY_CLAMP
    }

    /// Signed distance from `p` to the op's shape.
    #[inline]
    fn shape_distance(&self, p: Vec3) -> f32 {
        match self.kind {
            EditKind::Dig | EditKind::Fill => p.distance(self.center) - self.radius,
            EditKind::DigBox { half_height } | EditKind::FillBox { half_height } => {
                let q = (p - self.center).abs() - Vec3::new(self.radius, half_height, self.radius);
                q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
            }
        }
    }

    #[inline]
    pub fn apply(&self, p: Vec3, d: f32, m: u8) -> (f32, u8) {
        let s = self.shape_distance(p);
        match self.kind {
            EditKind::Dig | EditKind::DigBox { .. } => ((-s).max(d).min(DENSITY_CLAMP), m),
            EditKind::Fill | EditKind::FillBox { .. } if s < d => {
                (s.max(-DENSITY_CLAMP), self.material as u8)
            }
            EditKind::Fill | EditKind::FillBox { .. } => (d, m),
        }
    }

    /// The two box edits that level a square footprint to `height`: cut everything above it (up to
    /// `clear` metres) and fill everything below it (down to `depth` metres).
    pub fn flatten(
        center_xz: glam::Vec2,
        half_width: f32,
        height: f32,
        clear: f32,
        depth: f32,
        material: Material,
    ) -> [Self; 2] {
        [
            Self {
                center: Vec3::new(center_xz.x, height + clear / 2.0, center_xz.y),
                radius: half_width,
                kind: EditKind::DigBox { half_height: clear / 2.0 },
                material: Material::Air,
            },
            Self {
                center: Vec3::new(center_xz.x, height - depth / 2.0, center_xz.y),
                radius: half_width,
                kind: EditKind::FillBox { half_height: depth / 2.0 },
                material,
            },
        ]
    }

    /// Every chunk whose stored samples this op can change.
    pub fn touched_chunks(&self) -> impl Iterator<Item = ChunkCoord> {
        // A chunk stores samples from origin-1 to origin+CHUNK+1, so widen by those margins. This is a
        // superset; the exact sphere/box test below trims it.
        let reach = Vec3::splat(self.reach());
        let lo = chunk_of(self.center - reach - Vec3::splat(CHUNK as f32 + 1.0));
        let hi = chunk_of(self.center + reach + Vec3::ONE);
        let op = *self;
        (lo.z..=hi.z)
            .flat_map(move |z| {
                (lo.y..=hi.y).flat_map(move |y| (lo.x..=hi.x).map(move |x| IVec3::new(x, y, z)))
            })
            .filter(move |c| op.overlaps_chunk(*c))
    }

    fn overlaps_chunk(&self, coord: ChunkCoord) -> bool {
        let (lo, hi) = ChunkData::sample_bounds(coord);
        let nearest = self.center.clamp(lo, hi);
        nearest.distance(self.center) < self.reach()
    }
}

#[derive(Default)]
pub struct EditLog {
    ops: Vec<EditOp>,
    by_chunk: AHashMap<ChunkCoord, Vec<u32>>,
}

impl EditLog {
    /// Record an edit and return the chunks that must be rebuilt.
    pub fn push(&mut self, op: EditOp) -> Vec<ChunkCoord> {
        let id = self.ops.len() as u32;
        self.ops.push(op);
        let touched: Vec<ChunkCoord> = op.touched_chunks().collect();
        for c in &touched {
            self.by_chunk.entry(*c).or_default().push(id);
        }
        touched
    }

    pub fn ops_for(&self, coord: ChunkCoord) -> impl Iterator<Item = &EditOp> {
        self.by_chunk.get(&coord).into_iter().flatten().map(|id| &self.ops[*id as usize])
    }

    /// Ops that may affect point `p`, in application order.
    pub fn ops_near(&self, p: Vec3) -> impl Iterator<Item = &EditOp> {
        self.ops_for(chunk_of(p))
    }

    /// Every op in the order it was made, for saving.
    pub fn ops(&self) -> &[EditOp] {
        &self.ops
    }

    /// Rebuild a log (and its chunk index) from saved ops.
    pub fn from_ops(ops: impl IntoIterator<Item = EditOp>) -> Self {
        let mut log = Self::default();
        for op in ops {
            log.push(op);
        }
        log
    }

    pub fn is_edited(&self, coord: ChunkCoord) -> bool {
        self.by_chunk.contains_key(&coord)
    }

    pub fn len(&self) -> usize {
        self.ops.len()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}
