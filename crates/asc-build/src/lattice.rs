//! The building grid: where each piece sits, and how a piece's key maps to world space.
//!
//! Cells are `cell` metres square; heights are integer `height_unit` steps. A foundation fills a
//! cell from its base height up; a ceiling hangs below its top height (flush with the top of the
//! walls it rests on, so the next storey starts exactly there); a wall stands on a cell edge.
//! Walls are only ever keyed on a cell's north (+z) or east (+x) edge, so each physical edge has
//! exactly one key.

use glam::{IVec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::spec::{BuildingSpec, PartKind};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Edge {
    /// The cell's +z edge; the wall runs along x.
    North,
    /// The cell's +x edge; the wall runs along z.
    East,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PieceKey {
    pub kind: PartKind,
    pub cell: IVec2,
    /// Height in units: a foundation's or wall's base, a ceiling's top.
    pub y: i32,
    /// Walls only.
    pub edge: Option<Edge>,
}

impl PieceKey {
    pub fn foundation(cell: IVec2, y: i32) -> Self {
        Self { kind: PartKind::Foundation, cell, y, edge: None }
    }

    pub fn ceiling(cell: IVec2, y: i32) -> Self {
        Self { kind: PartKind::Ceiling, cell, y, edge: None }
    }

    pub fn wall(cell: IVec2, y: i32, edge: Edge) -> Self {
        Self { kind: PartKind::Wall, cell, y, edge: Some(edge) }
    }

    /// The wall on any side of a cell, normalised to a north or east key.
    pub fn wall_on_side(cell: IVec2, y: i32, side: Side) -> Self {
        match side {
            Side::PosZ => Self::wall(cell, y, Edge::North),
            Side::NegZ => Self::wall(cell - IVec2::Y, y, Edge::North),
            Side::PosX => Self::wall(cell, y, Edge::East),
            Side::NegX => Self::wall(cell - IVec2::X, y, Edge::East),
        }
    }
}

/// A side of a cell, in world terms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Side {
    PosX,
    NegX,
    PosZ,
    NegZ,
}

impl Side {
    pub const ALL: [Self; 4] = [Self::PosX, Self::NegX, Self::PosZ, Self::NegZ];

    pub fn offset(self) -> IVec2 {
        match self {
            Self::PosX => IVec2::X,
            Self::NegX => -IVec2::X,
            Self::PosZ => IVec2::Y,
            Self::NegZ => -IVec2::Y,
        }
    }

    /// The side a horizontal direction points to most.
    pub fn from_dir(x: f32, z: f32) -> Self {
        if x.abs() >= z.abs() {
            if x >= 0.0 { Self::PosX } else { Self::NegX }
        } else if z >= 0.0 {
            Self::PosZ
        } else {
            Self::NegZ
        }
    }
}

/// Grid geometry derived from the spec.
#[derive(Clone, Debug)]
pub struct Lattice {
    pub cell: f32,
    pub unit: f32,
    pub foundation_height: f32,
    pub foundation_units: i32,
    pub wall_height: f32,
    pub wall_units: i32,
    pub wall_thickness: f32,
    pub ceiling_thickness: f32,
}

impl Lattice {
    pub fn new(spec: &BuildingSpec) -> Self {
        let f = spec.part(PartKind::Foundation).size();
        let w = spec.part(PartKind::Wall).size();
        let c = spec.part(PartKind::Ceiling).size();
        let unit = spec.height_unit;
        Self {
            cell: spec.cell,
            unit,
            foundation_height: f.y,
            foundation_units: (f.y / unit).round() as i32,
            wall_height: w.y,
            wall_units: (w.y / unit).round() as i32,
            wall_thickness: w.z,
            ceiling_thickness: c.y,
        }
    }

    pub fn cell_of(&self, x: f32, z: f32) -> IVec2 {
        IVec2::new((x / self.cell).floor() as i32, (z / self.cell).floor() as i32)
    }

    pub fn height(&self, units: i32) -> f32 {
        units as f32 * self.unit
    }

    /// Smallest height unit at or above `y`.
    pub fn units_at_or_above(&self, y: f32) -> i32 {
        (y / self.unit - 1e-4).ceil() as i32
    }

    fn cell_min(&self, cell: IVec2) -> (f32, f32) {
        (cell.x as f32 * self.cell, cell.y as f32 * self.cell)
    }

    /// Model origin and quarter turns for drawing a piece.
    pub fn placement(&self, key: &PieceKey) -> (Vec3, u8) {
        let (x0, z0) = self.cell_min(key.cell);
        let half = self.cell / 2.0;
        let y = self.height(key.y);
        match (key.kind, key.edge) {
            (PartKind::Wall, Some(Edge::North)) => (Vec3::new(x0 + half, y, z0 + self.cell), 0),
            (PartKind::Wall, _) => (Vec3::new(x0 + self.cell, y, z0 + half), 1),
            _ => (Vec3::new(x0 + half, y, z0 + half), 0),
        }
    }

    /// World-space box a piece occupies.
    pub fn bounds(&self, key: &PieceKey) -> (Vec3, Vec3) {
        let (x0, z0) = self.cell_min(key.cell);
        let (x1, z1) = (x0 + self.cell, z0 + self.cell);
        let y = self.height(key.y);
        let t = self.wall_thickness / 2.0;
        match (key.kind, key.edge) {
            (PartKind::Foundation, _) => {
                (Vec3::new(x0, y, z0), Vec3::new(x1, y + self.foundation_height, z1))
            }
            (PartKind::Ceiling, _) => (Vec3::new(x0, y - self.ceiling_thickness, z0), Vec3::new(x1, y, z1)),
            (PartKind::Wall, Some(Edge::North)) => {
                (Vec3::new(x0, y, z1 - t), Vec3::new(x1, y + self.wall_height, z1 + t))
            }
            (PartKind::Wall, _) => (Vec3::new(x1 - t, y, z0), Vec3::new(x1 + t, y + self.wall_height, z1)),
        }
    }

    /// Height units of a piece's top surface (what stands on it starts here).
    pub fn top(&self, key: &PieceKey) -> i32 {
        match key.kind {
            PartKind::Foundation => key.y + self.foundation_units,
            PartKind::Wall => key.y + self.wall_units,
            PartKind::Ceiling => key.y,
        }
    }

    /// The two cells a wall's edge borders.
    pub fn wall_cells(&self, key: &PieceKey) -> [IVec2; 2] {
        match key.edge {
            Some(Edge::North) => [key.cell, key.cell + IVec2::Y],
            _ => [key.cell, key.cell + IVec2::X],
        }
    }
}

/// Signed distance from `p` to an axis-aligned box.
pub fn box_sdf(p: Vec3, lo: Vec3, hi: Vec3) -> f32 {
    let center = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;
    let q = (p - center).abs() - half;
    q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    pub(crate) fn lattice() -> Lattice {
        Lattice::new(&BuildingSpec::parse(include_str!("../../../assets/data/building.toml")).unwrap())
    }

    #[test]
    fn derived_sizes() {
        let l = lattice();
        assert_eq!((l.foundation_units, l.wall_units), (2, 4));
        assert_eq!(l.cell_of(-0.1, 3.9), IVec2::new(-1, 1));
        assert_eq!(l.units_at_or_above(1.0), 2);
        assert_eq!(l.units_at_or_above(1.01), 3);
    }

    #[test]
    fn walls_on_any_side_normalise_to_one_key() {
        let a = PieceKey::wall_on_side(IVec2::new(3, 4), 0, Side::PosZ);
        let b = PieceKey::wall_on_side(IVec2::new(3, 5), 0, Side::NegZ);
        assert_eq!(a, b);
        let c = PieceKey::wall_on_side(IVec2::new(3, 4), 0, Side::PosX);
        let d = PieceKey::wall_on_side(IVec2::new(4, 4), 0, Side::NegX);
        assert_eq!(c, d);
    }

    #[test]
    fn bounds_match_part_sizes_and_placement() {
        let l = lattice();
        let f = PieceKey::foundation(IVec2::new(1, -1), 4);
        assert_eq!(l.bounds(&f), (Vec3::new(2.0, 2.0, -2.0), Vec3::new(4.0, 3.0, 0.0)));
        let w = PieceKey::wall(IVec2::new(0, 0), 6, Edge::East);
        let (lo, hi) = l.bounds(&w);
        assert!((hi - lo - Vec3::new(0.2, 2.0, 2.0)).abs().max_element() < 1e-5);
        assert_eq!(l.placement(&w), (Vec3::new(2.0, 3.0, 1.0), 1));
        let c = PieceKey::ceiling(IVec2::ZERO, 10);
        assert_eq!(l.bounds(&c).1.y, 5.0);
        assert_eq!(l.top(&c), 10);
    }

    #[test]
    fn box_sdf_inside_and_out() {
        let (lo, hi) = (Vec3::ZERO, Vec3::splat(2.0));
        assert!((box_sdf(Vec3::new(1.0, 3.0, 1.0), lo, hi) - 1.0).abs() < 1e-6);
        assert!((box_sdf(Vec3::splat(1.0), lo, hi) + 1.0).abs() < 1e-6);
        assert!((box_sdf(Vec3::new(3.0, 3.0, 1.0), lo, hi) - 2f32.sqrt()).abs() < 1e-6);
    }
}
