//! The authoritative game: its state and rules, the fixed-rate simulation, saves and (later) the
//! protocol a client talks to it with. No window, GPU or renderer types, so it can run on its own
//! thread inside the game or, one day, headless as a server.

pub mod content;
pub mod creatures;
pub mod fire;
pub mod fluids;
pub mod inventory;
pub mod islands;
pub mod link;
pub mod nav;
pub mod ore_nodes;
pub mod path;
pub mod protocol;
pub mod rules;
pub mod save;
pub mod schedule;
pub mod server;
pub mod sim;
pub mod stress;
pub mod vitals;
pub mod volcanism;

use serde::{Deserialize, Serialize};

/// Eye height above the feet, metres.
pub const EYE_HEIGHT: f32 = 1.62;

/// How a player moves: walking with physics, or flying freely.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Walk,
    Fly,
}
