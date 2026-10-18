//! Building: the piece spec (shared with the Blender generator), and the grid lattice, snapping and
//! support rules. Pure logic; no GPU or engine types.

mod lattice;
mod spec;
mod structures;

pub use lattice::{Edge, Lattice, PieceKey, Side, box_sdf};
pub use spec::{BuildingSpec, PartKind, PartSpec, TierSpec};
pub use structures::{Invalid, Piece, Snap, Structures, Target};
