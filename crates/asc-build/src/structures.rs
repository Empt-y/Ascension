//! Placed pieces: the rules for where they may go, snapping, dismantling, and their distance field
//! (so the player and raycasts collide with them through the same code as terrain).

use std::fmt;

use ahash::AHashMap;
use glam::{IVec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::lattice::{Lattice, PieceKey, Side, box_sdf};
use crate::spec::PartKind;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Piece {
    pub key: PieceKey,
    pub tier: u8,
    /// A foundation resting on terrain (so it needs no other support).
    pub grounded: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Invalid {
    Occupied,
    Overlaps,
    NoSupport,
    /// Dismantling would leave this piece unsupported.
    Supports(PieceKey),
    /// The ground drops away under the foundation.
    Gap,
    Missing,
}

impl fmt::Display for Invalid {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Occupied => write!(f, "something is already there"),
            Self::Overlaps => write!(f, "overlaps another piece"),
            Self::NoSupport => write!(f, "needs support"),
            Self::Supports(_) => write!(f, "other pieces rest on it"),
            Self::Gap => write!(f, "gap underneath (flatten the ground)"),
            Self::Missing => write!(f, "nothing to dismantle"),
        }
    }
}

/// What the crosshair is on, for snapping.
#[derive(Clone, Copy, Debug)]
pub enum Target {
    Terrain(Vec3),
    Piece { key: PieceKey, point: Vec3, normal: Vec3 },
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Snap {
    pub key: PieceKey,
    pub grounded: bool,
    /// Foundations: the largest drop from the foundation's base to the terrain under it.
    pub gap: f32,
}

/// The most terrain may fall away under a foundation (anywhere under its footprint) before the gap
/// counts as notable and the foundation can't be placed. Flatten the ground first.
pub const MAX_GAP: f32 = 0.5;

/// A foundation's top always clears the highest ground under it by at least this much (then rounds
/// up to the grid). Flush with the ground, terrain and floor would be coplanar and fight over which
/// is drawn (and which the crosshair hits).
pub const FOUNDATION_CLEARANCE: f32 = 0.1;

/// How far boxes are shrunk before overlap tests, so pieces resting on or against each other don't
/// count. Vertically generous (stacking), horizontally a hair (walls are only 0.2 m thick and
/// straddle cell edges, so a bigger margin would hide a wall half-buried in a foundation).
const TOUCH: Vec3 = Vec3::new(0.02, 0.15, 0.02);

pub struct Structures {
    pub lattice: Lattice,
    pieces: AHashMap<PieceKey, Piece>,
    /// Every cell a piece's box overlaps, for local queries.
    by_cell: AHashMap<IVec2, Vec<PieceKey>>,
}

fn boxes_overlap(a: (Vec3, Vec3), b: (Vec3, Vec3)) -> bool {
    a.0.cmplt(b.1).all() && b.0.cmplt(a.1).all()
}

fn shrink(b: (Vec3, Vec3), by: Vec3) -> (Vec3, Vec3) {
    (b.0 + by, b.1 - by)
}

impl Structures {
    pub fn new(lattice: Lattice) -> Self {
        Self { lattice, pieces: AHashMap::new(), by_cell: AHashMap::new() }
    }

    /// Restore saved pieces as they were, without re-checking the rules.
    pub fn from_pieces(lattice: Lattice, pieces: impl IntoIterator<Item = Piece>) -> Self {
        let mut s = Self::new(lattice);
        for p in pieces {
            s.insert(p);
        }
        s
    }

    pub fn len(&self) -> usize {
        self.pieces.len()
    }

    pub fn is_empty(&self) -> bool {
        self.pieces.is_empty()
    }

    pub fn get(&self, key: &PieceKey) -> Option<&Piece> {
        self.pieces.get(key)
    }

    pub fn iter(&self) -> impl Iterator<Item = &Piece> {
        self.pieces.values()
    }

    fn cells_of(&self, bounds: (Vec3, Vec3)) -> impl Iterator<Item = IVec2> {
        let lo = self.lattice.cell_of(bounds.0.x + 1e-3, bounds.0.z + 1e-3);
        let hi = self.lattice.cell_of(bounds.1.x - 1e-3, bounds.1.z - 1e-3);
        (lo.y..=hi.y).flat_map(move |z| (lo.x..=hi.x).map(move |x| IVec2::new(x, z)))
    }

    fn insert(&mut self, piece: Piece) {
        let cells: Vec<IVec2> = self.cells_of(self.lattice.bounds(&piece.key)).collect();
        for c in cells {
            self.by_cell.entry(c).or_default().push(piece.key);
        }
        self.pieces.insert(piece.key, piece);
    }

    fn take(&mut self, key: &PieceKey) -> Option<Piece> {
        let piece = self.pieces.remove(key)?;
        let cells: Vec<IVec2> = self.cells_of(self.lattice.bounds(key)).collect();
        for c in cells {
            if let Some(list) = self.by_cell.get_mut(&c) {
                list.retain(|k| k != key);
                if list.is_empty() {
                    self.by_cell.remove(&c);
                }
            }
        }
        Some(piece)
    }

    /// Pieces whose cells are within `radius` cells of `cell`.
    fn near(&self, cell: IVec2, radius: i32) -> impl Iterator<Item = &PieceKey> {
        (-radius..=radius)
            .flat_map(move |dz| (-radius..=radius).map(move |dx| cell + IVec2::new(dx, dz)))
            .filter_map(|c| self.by_cell.get(&c))
            .flatten()
    }

    fn has(&self, key: PieceKey) -> bool {
        self.pieces.contains_key(&key)
    }

    /// Whether a piece at `key` would be held up by what's already there.
    pub fn supported(&self, key: &PieceKey, grounded: bool) -> bool {
        let l = &self.lattice;
        match key.kind {
            PartKind::Foundation => {
                grounded
                    || self.has(PieceKey::foundation(key.cell, key.y - l.foundation_units))
                    || Side::ALL.iter().any(|s| self.has(PieceKey::foundation(key.cell + s.offset(), key.y)))
            }
            PartKind::Wall => {
                self.has(PieceKey { y: key.y - l.wall_units, ..*key })
                    || l.wall_cells(key).iter().any(|c| {
                        self.has(PieceKey::foundation(*c, key.y - l.foundation_units))
                            || self.has(PieceKey::ceiling(*c, key.y))
                    })
            }
            PartKind::Ceiling => Side::ALL.iter().any(|s| {
                self.has(PieceKey::wall_on_side(key.cell, key.y - l.wall_units, *s))
                    || self.has(PieceKey::ceiling(key.cell + s.offset(), key.y))
            }),
        }
    }

    /// Whether a piece could be placed here, and why not.
    pub fn check(&self, key: &PieceKey, grounded: bool) -> Result<(), Invalid> {
        if self.has(*key) {
            return Err(Invalid::Occupied);
        }
        let bounds = self.lattice.bounds(key);
        let overlaps = self.near(key.cell, 1).any(|other| {
            let theirs = self.lattice.bounds(other);
            match (key.kind, other.kind) {
                // Foundations are solid blocks: nothing may sit inside one, and one may not be
                // placed through anything.
                (PartKind::Foundation, _) => boxes_overlap(shrink(bounds, TOUCH), theirs),
                (_, PartKind::Foundation) => boxes_overlap(bounds, shrink(theirs, TOUCH)),
                // Walls and ceilings meet along their edges by design.
                _ => false,
            }
        });
        if overlaps {
            return Err(Invalid::Overlaps);
        }
        if !self.supported(key, grounded) {
            return Err(Invalid::NoSupport);
        }
        Ok(())
    }

    /// Check a snapped placement, including the ground under a foundation.
    pub fn check_snap(&self, snap: &Snap) -> Result<(), Invalid> {
        let key = &snap.key;
        if key.kind == PartKind::Foundation {
            let stacked = self.has(PieceKey::foundation(key.cell, key.y - self.lattice.foundation_units));
            if !stacked && snap.gap > MAX_GAP {
                // Report occupancy first: it's the more useful message when aiming at a piece.
                if self.has(*key) {
                    return Err(Invalid::Occupied);
                }
                return Err(Invalid::Gap);
            }
        }
        self.check(key, snap.grounded)
    }

    pub fn place(&mut self, piece: Piece) -> Result<(), Invalid> {
        self.check(&piece.key, piece.grounded)?;
        self.insert(piece);
        Ok(())
    }

    /// Remove a piece, unless something else rests on it.
    pub fn dismantle(&mut self, key: &PieceKey) -> Result<Piece, Invalid> {
        let piece = self.take(key).ok_or(Invalid::Missing)?;
        let neighbours: Vec<Piece> =
            self.near(key.cell, 1).filter_map(|k| self.pieces.get(k)).copied().collect();
        for other in neighbours {
            if !self.supported(&other.key, other.grounded) {
                self.insert(piece);
                return Err(Invalid::Supports(other.key));
            }
        }
        Ok(piece)
    }

    /// Signed distance to the nearest piece (large and positive when none is near).
    pub fn sdf(&self, p: Vec3) -> f32 {
        let cell = self.lattice.cell_of(p.x, p.z);
        self.near(cell, 1)
            .map(|k| {
                let (lo, hi) = self.lattice.bounds(k);
                box_sdf(p, lo, hi)
            })
            .fold(f32::MAX, f32::min)
    }

    /// The piece whose surface is at (or within `tolerance` of) `p`.
    pub fn piece_at(&self, p: Vec3, tolerance: f32) -> Option<PieceKey> {
        let cell = self.lattice.cell_of(p.x, p.z);
        self.near(cell, 1)
            .map(|k| {
                let (lo, hi) = self.lattice.bounds(k);
                (box_sdf(p, lo, hi), *k)
            })
            .filter(|(d, _)| *d < tolerance)
            .min_by(|a, b| a.0.total_cmp(&b.0))
            .map(|(_, k)| k)
    }

    /// The side of `cell` nearest to point `p` (inside or near the cell).
    fn nearest_side(&self, cell: IVec2, p: Vec3) -> Side {
        let l = &self.lattice;
        let u = p.x / l.cell - cell.x as f32;
        let v = p.z / l.cell - cell.y as f32;
        let sides = [(u, Side::NegX), (1.0 - u, Side::PosX), (v, Side::NegZ), (1.0 - v, Side::PosZ)];
        sides.into_iter().min_by(|a, b| a.0.total_cmp(&b.0)).map_or(Side::PosZ, |s| s.1)
    }

    /// Where a piece of `kind` would go for what the crosshair is on. The result may still be
    /// invalid; check it with [`Self::check`].
    pub fn snap(
        &self,
        kind: PartKind,
        target: Target,
        camera: Vec3,
        terrain: &impl Fn(f32, f32) -> f32,
    ) -> Snap {
        let l = &self.lattice;
        let terrain_units = |p: Vec3| l.units_at_or_above(terrain(p.x, p.z));
        match kind {
            PartKind::Foundation => match target {
                Target::Piece { key, point, normal } if key.kind == PartKind::Foundation => {
                    let side = if normal.y.abs() < 0.5 {
                        Side::from_dir(normal.x, normal.z)
                    } else {
                        self.nearest_side(key.cell, point)
                    };
                    let key = PieceKey::foundation(key.cell + side.offset(), key.y);
                    self.foundation_snap(key, terrain)
                }
                Target::Piece { point, .. } | Target::Terrain(point) => {
                    self.foundation_on_terrain(point, terrain)
                }
            },
            PartKind::Wall => {
                let key = match target {
                    Target::Piece { key, .. } if key.kind == PartKind::Wall => {
                        PieceKey { y: key.y + l.wall_units, ..key }
                    }
                    Target::Piece { key, point, normal } if normal.y > 0.5 => {
                        PieceKey::wall_on_side(key.cell, l.top(&key), self.nearest_side(key.cell, point))
                    }
                    Target::Piece { point, .. } | Target::Terrain(point) => {
                        let cell = l.cell_of(point.x, point.z);
                        PieceKey::wall_on_side(cell, terrain_units(point), self.nearest_side(cell, point))
                    }
                };
                Snap { key, grounded: false, gap: 0.0 }
            }
            PartKind::Ceiling => {
                let key = match target {
                    Target::Piece { key, .. } if key.kind == PartKind::Wall => {
                        // Over the room on the camera's side of the wall.
                        let [a, b] = l.wall_cells(&key);
                        let center = |c: IVec2| (c.as_vec2() + 0.5) * l.cell;
                        let cam = glam::Vec2::new(camera.x, camera.z);
                        let cell = if center(a).distance(cam) <= center(b).distance(cam) { a } else { b };
                        PieceKey::ceiling(cell, l.top(&key))
                    }
                    Target::Piece { key, point, normal } if key.kind == PartKind::Ceiling => {
                        let side = if normal.y.abs() < 0.5 {
                            Side::from_dir(normal.x, normal.z)
                        } else {
                            self.nearest_side(key.cell, point)
                        };
                        PieceKey::ceiling(key.cell + side.offset(), key.y)
                    }
                    Target::Piece { key, .. } => PieceKey::ceiling(key.cell, l.top(&key) + l.wall_units),
                    Target::Terrain(point) => {
                        PieceKey::ceiling(l.cell_of(point.x, point.z), terrain_units(point) + l.wall_units)
                    }
                };
                Snap { key, grounded: false, gap: 0.0 }
            }
        }
    }

    /// Terrain heights at the corners and centre of a cell.
    fn footprint_heights(&self, cell: IVec2, terrain: &impl Fn(f32, f32) -> f32) -> [f32; 5] {
        let l = &self.lattice;
        let (x0, z0) = (cell.x as f32 * l.cell, cell.y as f32 * l.cell);
        let inset = 0.05;
        [
            (x0 + inset, z0 + inset),
            (x0 + l.cell - inset, z0 + inset),
            (x0 + inset, z0 + l.cell - inset),
            (x0 + l.cell - inset, z0 + l.cell - inset),
            (x0 + l.cell / 2.0, z0 + l.cell / 2.0),
        ]
        .map(|(x, z)| terrain(x, z))
    }

    /// A snap for a foundation at `key`, measuring the ground under it.
    fn foundation_snap(&self, key: PieceKey, terrain: &impl Fn(f32, f32) -> f32) -> Snap {
        let base = self.lattice.height(key.y);
        let lowest = self.footprint_heights(key.cell, terrain).into_iter().fold(f32::MAX, f32::min);
        let gap = (base - lowest).max(0.0);
        Snap { key, grounded: gap <= MAX_GAP, gap }
    }

    /// A foundation over the cell at `point`, its top just above the highest terrain under it, or
    /// level with a neighbouring foundation if one is within a foundation's height.
    fn foundation_on_terrain(&self, point: Vec3, terrain: &impl Fn(f32, f32) -> f32) -> Snap {
        let l = &self.lattice;
        let cell = l.cell_of(point.x, point.z);
        let highest = self.footprint_heights(cell, terrain).into_iter().fold(f32::MIN, f32::max);
        let mut y = l.units_at_or_above(highest + FOUNDATION_CLEARANCE) - l.foundation_units;
        let neighbour = Side::ALL.iter().find_map(|s| {
            self.near(cell + s.offset(), 0)
                .find(|k| {
                    k.kind == PartKind::Foundation
                        && k.cell == cell + s.offset()
                        && (k.y - y).abs() <= l.foundation_units
                })
                .map(|k| k.y)
        });
        if let Some(ny) = neighbour {
            y = ny;
        }
        self.foundation_snap(PieceKey::foundation(cell, y), terrain)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lattice::Edge;
    use crate::spec::BuildingSpec;

    fn structures() -> Structures {
        let spec = BuildingSpec::parse(include_str!("../../../assets/data/building.toml")).unwrap();
        Structures::new(Lattice::new(&spec))
    }

    fn flat(h: f32) -> impl Fn(f32, f32) -> f32 {
        move |_, _| h
    }

    fn piece(key: PieceKey) -> Piece {
        Piece { key, tier: 0, grounded: key.kind == PartKind::Foundation }
    }

    /// A 1x1 room: foundation at y 0, four walls, a ceiling.
    fn room(s: &mut Structures) {
        let f = PieceKey::foundation(IVec2::ZERO, 0);
        s.place(piece(f)).unwrap();
        for side in Side::ALL {
            s.place(piece(PieceKey::wall_on_side(IVec2::ZERO, 2, side))).unwrap();
        }
        s.place(piece(PieceKey::ceiling(IVec2::ZERO, 6))).unwrap();
    }

    #[test]
    fn foundations_stand_clear_of_flat_ground() {
        let s = structures();
        // Ground exactly on a height unit: the top goes a whole unit up, not flush with it.
        let at = Target::Terrain(Vec3::new(1.0, 17.0, 1.0));
        let snap = s.snap(PartKind::Foundation, at, Vec3::ZERO, &flat(17.0));
        assert_eq!(s.lattice.height(s.lattice.top(&snap.key)), 17.5);
    }

    #[test]
    fn foundation_sits_above_the_highest_terrain_under_it() {
        let s = structures();
        let snap =
            s.snap(PartKind::Foundation, Target::Terrain(Vec3::new(1.0, 3.2, 1.0)), Vec3::ZERO, &flat(3.2));
        // Top rounded up to 3.5 m = 7 units, so the base is at 5.
        assert_eq!(snap.key, PieceKey::foundation(IVec2::ZERO, 5));
        assert!(snap.grounded);
        let slope = |x: f32, _z: f32| x * 0.5;
        let snap =
            s.snap(PartKind::Foundation, Target::Terrain(Vec3::new(5.0, 0.0, 1.0)), Vec3::ZERO, &slope);
        // Cell x 4..6: highest sample 5.95 * 0.5 = 2.975, plus clearance -> top 3.5 = 7 units -> base 5.
        assert_eq!(snap.key.y, 5);
    }

    #[test]
    fn foundations_line_up_with_their_neighbours() {
        let mut s = structures();
        s.place(piece(PieceKey::foundation(IVec2::ZERO, 2))).unwrap();
        // Terrain next door is a little higher, but within a foundation's height: stay level.
        let snap =
            s.snap(PartKind::Foundation, Target::Terrain(Vec3::new(3.0, 0.0, 1.0)), Vec3::ZERO, &flat(2.4));
        assert_eq!(snap.key, PieceKey::foundation(IVec2::new(1, 0), 2));
        // Looking at a foundation's side extends it that way.
        let target = Target::Piece {
            key: PieceKey::foundation(IVec2::ZERO, 2),
            point: Vec3::new(1.0, 1.5, 2.0),
            normal: Vec3::Z,
        };
        let snap = s.snap(PartKind::Foundation, target, Vec3::ZERO, &flat(0.0));
        assert_eq!(snap.key, PieceKey::foundation(IVec2::new(0, 1), 2));
        assert!(s.check(&snap.key, snap.grounded).is_ok());
    }

    #[test]
    fn foundations_need_ground_close_underneath() {
        let mut s = structures();
        // Flat ground: no gap.
        let snap =
            s.snap(PartKind::Foundation, Target::Terrain(Vec3::new(1.0, 0.0, 1.0)), Vec3::ZERO, &flat(0.0));
        assert!(snap.gap < 0.01 && s.check_snap(&snap).is_ok());
        // A 1 m foundation absorbs a 1 m rise across its cell (it sinks into the low side), but on
        // a slope rising 2 m the top clears the high side and leaves ~1 m open under the low side.
        let gentle = |x: f32, _z: f32| x * 0.5;
        let snap =
            s.snap(PartKind::Foundation, Target::Terrain(Vec3::new(5.0, 0.0, 1.0)), Vec3::ZERO, &gentle);
        assert!(s.check_snap(&snap).is_ok(), "gentle slope gap {}", snap.gap);
        let slope = |x: f32, _z: f32| x;
        let snap =
            s.snap(PartKind::Foundation, Target::Terrain(Vec3::new(5.0, 0.0, 1.0)), Vec3::ZERO, &slope);
        assert!(snap.gap > MAX_GAP, "gap {}", snap.gap);
        assert_eq!(s.check_snap(&snap), Err(Invalid::Gap));
        // Extending a foundation out over a cliff edge is refused too.
        s.place(piece(PieceKey::foundation(IVec2::ZERO, 0))).unwrap();
        let cliff = |x: f32, _z: f32| if x < 2.0 { 0.0 } else { -6.0 };
        let target = Target::Piece {
            key: PieceKey::foundation(IVec2::ZERO, 0),
            point: Vec3::new(2.0, 0.5, 1.0),
            normal: Vec3::X,
        };
        let snap = s.snap(PartKind::Foundation, target, Vec3::ZERO, &cliff);
        assert_eq!(snap.key, PieceKey::foundation(IVec2::new(1, 0), 0));
        assert_eq!(s.check_snap(&snap), Err(Invalid::Gap));
        // But stacking straight on top of a foundation needs no ground.
        let stacked = Snap { key: PieceKey::foundation(IVec2::ZERO, 2), grounded: false, gap: 99.0 };
        assert!(s.check_snap(&stacked).is_ok());
    }

    #[test]
    fn walls_snap_to_the_nearest_edge_and_stack() {
        let mut s = structures();
        let f = PieceKey::foundation(IVec2::ZERO, 0);
        s.place(piece(f)).unwrap();
        let target = Target::Piece { key: f, point: Vec3::new(1.0, 1.0, 1.8), normal: Vec3::Y };
        let snap = s.snap(PartKind::Wall, target, Vec3::ZERO, &flat(0.0));
        assert_eq!(snap.key, PieceKey::wall(IVec2::ZERO, 2, Edge::North));
        s.place(piece(snap.key)).unwrap();
        let on_wall = Target::Piece { key: snap.key, point: Vec3::new(1.0, 3.0, 2.0), normal: Vec3::Z };
        assert_eq!(s.snap(PartKind::Wall, on_wall, Vec3::ZERO, &flat(0.0)).key.y, 6);
        let west = Target::Piece { key: f, point: Vec3::new(0.1, 1.0, 1.0), normal: Vec3::Y };
        assert_eq!(
            s.snap(PartKind::Wall, west, Vec3::ZERO, &flat(0.0)).key,
            PieceKey::wall(IVec2::new(-1, 0), 2, Edge::East)
        );
    }

    #[test]
    fn ceilings_go_over_the_room_on_the_camera_side() {
        let mut s = structures();
        room(&mut s);
        let wall = PieceKey::wall(IVec2::ZERO, 2, Edge::North);
        let target = Target::Piece { key: wall, point: Vec3::new(1.0, 3.0, 2.0), normal: -Vec3::Z };
        let inside = s.snap(PartKind::Ceiling, target, Vec3::new(1.0, 1.6, 1.0), &flat(0.0));
        assert_eq!(inside.key, PieceKey::ceiling(IVec2::ZERO, 6));
        let outside = s.snap(PartKind::Ceiling, target, Vec3::new(1.0, 1.6, 5.0), &flat(0.0));
        assert_eq!(outside.key, PieceKey::ceiling(IVec2::new(0, 1), 6));
    }

    #[test]
    fn support_rules() {
        let mut s = structures();
        assert_eq!(s.check(&PieceKey::wall(IVec2::ZERO, 2, Edge::North), false), Err(Invalid::NoSupport));
        assert_eq!(s.check(&PieceKey::ceiling(IVec2::ZERO, 6), false), Err(Invalid::NoSupport));
        assert_eq!(s.check(&PieceKey::foundation(IVec2::ZERO, 0), false), Err(Invalid::NoSupport));
        room(&mut s);
        // A ceiling can extend from another ceiling...
        assert!(s.check(&PieceKey::ceiling(IVec2::new(1, 0), 6), false).is_ok());
        // ...and a second storey's walls stand on the ceiling.
        assert!(s.check(&PieceKey::wall(IVec2::new(0, 0), 6, Edge::North), false).is_ok());
    }

    #[test]
    fn nothing_goes_through_a_foundation() {
        let mut s = structures();
        s.place(piece(PieceKey::foundation(IVec2::ZERO, 0))).unwrap();
        // A wall whose base is inside the foundation.
        assert_eq!(s.check(&PieceKey::wall(IVec2::ZERO, 0, Edge::North), false), Err(Invalid::Overlaps));
        assert_eq!(s.check(&PieceKey::foundation(IVec2::ZERO, 0), true), Err(Invalid::Occupied));
        // A foundation half-overlapping one below it (base 1 unit up).
        assert_eq!(s.check(&PieceKey::foundation(IVec2::ZERO, 1), true), Err(Invalid::Overlaps));
        // But one stacked exactly on top is fine.
        assert!(s.check(&PieceKey::foundation(IVec2::ZERO, 2), false).is_ok());
    }

    #[test]
    fn dismantling_refuses_to_strand_pieces() {
        let mut s = structures();
        room(&mut s);
        let f = PieceKey::foundation(IVec2::ZERO, 0);
        assert!(matches!(s.dismantle(&f), Err(Invalid::Supports(_))));
        assert_eq!(s.len(), 6, "a refused dismantle changes nothing");
        // One wall can go: the ceiling still rests on the other three.
        assert!(s.dismantle(&PieceKey::wall(IVec2::ZERO, 2, Edge::North)).is_ok());
        assert!(s.dismantle(&PieceKey::ceiling(IVec2::ZERO, 6)).is_ok());
        assert_eq!(s.dismantle(&PieceKey::ceiling(IVec2::ZERO, 6)), Err(Invalid::Missing));
    }

    #[test]
    fn distance_field_and_lookup() {
        let mut s = structures();
        room(&mut s);
        // Standing on the floor: 0.3 m above the foundation top.
        assert!((s.sdf(Vec3::new(1.0, 1.3, 1.0)) - 0.3).abs() < 1e-4);
        // Inside the north wall.
        assert!(s.sdf(Vec3::new(1.0, 2.0, 2.0)) < 0.0);
        assert_eq!(s.piece_at(Vec3::new(1.0, 1.0, 1.0), 0.01), Some(PieceKey::foundation(IVec2::ZERO, 0)));
        assert_eq!(s.sdf(Vec3::new(50.0, 0.0, 50.0)), f32::MAX);
    }
}
