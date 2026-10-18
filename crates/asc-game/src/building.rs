//! Build mode: choosing a piece, the snapped ghost, placing and dismantling, and keeping the
//! renderer's part instances in sync with the structures.

use asc_build::{Invalid, PartKind, Piece, PieceKey, Snap, Structures, Target};
use asc_render::{Ghost, PartInstance, Renderer};
use asc_world::RayHit;
use glam::Vec3;

use crate::assets::PartModels;

/// How far away you can build.
pub const BUILD_REACH: f32 = 10.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Selection {
    pub tier: usize,
    pub kind: PartKind,
}

pub struct Building {
    pub structures: Structures,
    /// The piece being placed, while in build mode.
    pub selection: Option<Selection>,
    pub menu_open: bool,
    /// The current snap and whether it can be placed.
    pub preview: Option<(Snap, Result<(), Invalid>)>,
    dirty: bool,
}

impl Building {
    pub fn new(structures: Structures) -> Self {
        Self { structures, selection: None, menu_open: false, preview: None, dirty: true }
    }

    /// Signed distance to the nearest piece, for collision and raycasts.
    pub fn sdf(&self, p: Vec3) -> f32 {
        self.structures.sdf(p)
    }

    /// What a ray hit, as a snapping target: a piece if the hit point is on one, otherwise terrain.
    pub fn target(&self, hit: &RayHit) -> Target {
        match self.structures.piece_at(hit.point, 0.05) {
            Some(key) => Target::Piece { key, point: hit.point, normal: hit.normal },
            None => Target::Terrain(hit.point),
        }
    }

    pub fn update_preview(&mut self, hit: Option<RayHit>, camera: Vec3, terrain: &impl Fn(f32, f32) -> f32) {
        self.preview = match (self.selection, hit) {
            (Some(sel), Some(hit)) if hit.distance <= BUILD_REACH => {
                let snap = self.structures.snap(sel.kind, self.target(&hit), camera, terrain);
                let valid = self.structures.check(&snap.key, snap.grounded);
                Some((snap, valid))
            }
            _ => None,
        };
    }

    /// Place the previewed piece. Returns why not, if it can't be.
    pub fn place(&mut self) -> Result<PieceKey, Invalid> {
        let (Some(sel), Some((snap, _))) = (self.selection, self.preview) else {
            return Err(Invalid::Missing);
        };
        self.structures.place(Piece { key: snap.key, tier: sel.tier as u8, grounded: snap.grounded })?;
        self.dirty = true;
        Ok(snap.key)
    }

    /// Dismantle the piece the crosshair is on.
    pub fn dismantle(&mut self, hit: Option<RayHit>) -> Result<Piece, Invalid> {
        let hit = hit.filter(|h| h.distance <= BUILD_REACH).ok_or(Invalid::Missing)?;
        let key = self.structures.piece_at(hit.point, 0.05).ok_or(Invalid::Missing)?;
        let piece = self.structures.dismantle(&key)?;
        self.dirty = true;
        Ok(piece)
    }

    /// Push changes to the renderer: every placed piece when the set changed, and the ghost.
    pub fn sync(&mut self, renderer: &mut Renderer, models: &PartModels) {
        let lattice = &self.structures.lattice;
        if self.dirty {
            let instances: Vec<PartInstance> = self
                .structures
                .iter()
                .map(|p| {
                    let (position, rotation) = lattice.placement(&p.key);
                    PartInstance::rigid(models.model(p.tier as usize, p.key.kind), position, rotation)
                })
                .collect();
            renderer.set_parts(&instances);
            self.dirty = false;
        }
        let ghost = match (self.selection, self.preview) {
            (Some(sel), Some((snap, valid))) => {
                let (position, rotation) = lattice.placement(&snap.key);
                let model = models.model(sel.tier, snap.key.kind);
                Some(Ghost { instance: PartInstance::rigid(model, position, rotation), valid: valid.is_ok() })
            }
            _ => None,
        };
        renderer.set_ghost(ghost);
    }

    /// Place a piece directly (scripted tests and tools), bypassing snapping.
    pub fn place_key(&mut self, key: PieceKey, tier: usize, grounded: bool) -> Result<(), Invalid> {
        self.structures.place(Piece { key, tier: tier as u8, grounded })?;
        self.dirty = true;
        Ok(())
    }
}
