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
    /// Other rigid things drawn with the pieces (stashes left at death).
    extras: Vec<PartInstance>,
    moving: Vec<PartInstance>,
    dirty: bool,
}

impl Building {
    pub fn new(structures: Structures) -> Self {
        Self { structures, selection: None, menu_open: false, preview: None, extras: Vec::new(), moving: Vec::new(), dirty: true }
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

    /// The previewed piece and its gap, to ask the server to place. Returns why not, if it can't be.
    pub fn request(&self) -> Result<(Piece, f32), Invalid> {
        let (Some(sel), Some((snap, valid))) = (self.selection, self.preview) else {
            return Err(Invalid::Missing);
        };
        valid?;
        Ok((Piece { key: snap.key, tier: sel.tier as u8, grounded: snap.grounded }, snap.gap))
    }

    /// The piece the crosshair is on, to ask the server to dismantle.
    pub fn dismantle_target(&self, hit: Option<RayHit>) -> Result<PieceKey, Invalid> {
        let hit = hit.filter(|h| h.distance <= BUILD_REACH).ok_or(Invalid::Missing)?;
        self.structures.piece_at(hit.point, 0.05).ok_or(Invalid::Missing)
    }

    /// Replace the extra instances drawn alongside the pieces.
    pub fn set_extras(&mut self, extras: Vec<PartInstance>) {
        self.extras = extras;
        self.dirty = true;
    }

    /// Replace the moving instances (volcanic bombs in flight), redrawn every frame they change.
    pub fn set_moving(&mut self, moving: Vec<PartInstance>) {
        if moving.is_empty() && self.moving.is_empty() {
            return;
        }
        self.moving = moving;
        self.dirty = true;
    }

    /// The server placed a piece.
    pub fn placed(&mut self, piece: Piece) {
        self.structures.insert(piece);
        self.dirty = true;
    }

    /// The server removed a piece.
    pub fn removed(&mut self, key: &PieceKey) {
        self.structures.take(key);
        self.dirty = true;
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
            let mut instances = instances;
            instances.extend(self.extras.iter().copied());
            instances.extend(self.moving.iter().copied());
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
}
