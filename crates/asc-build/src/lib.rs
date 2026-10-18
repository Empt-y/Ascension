//! Building: the piece spec (shared with the Blender generator), and the grid lattice, snapping and
//! support rules. Pure logic; no GPU or engine types.

mod spec;

pub use spec::{BuildingSpec, PartKind, PartSpec, TierSpec};
