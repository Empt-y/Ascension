//! The messages between a client and the server.
//!
//! The client sends what its player wants to do; the server checks it, applies it and tells every
//! client what changed. Clients keep copies of the world (seed plus edits), the building pieces,
//! the mined nodes and their own inventory, updated only from these messages. In one process the
//! messages travel as they are; everything is serde so they can later go over a network.

use asc_build::{Piece, PieceKey};
use asc_world::EditOp;
use glam::{Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::Mode;
use crate::inventory::{InventoryOp, InventorySnapshot};
use crate::ore_nodes::NodeId;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ClientMsg {
    /// Where the player is (sent every tick; the client simulates its own movement).
    Move { feet: Vec3, yaw: f32, pitch: f32, mode: Mode },
    /// A finished mining bite at `point` on a surface facing `normal`.
    Bite { point: Vec3, normal: Vec3 },
    /// Place a lump of the selected item's material. `fresh` when the button was just pressed (a
    /// refusal is only reported then, not on every repeat of a held button).
    Fill { point: Vec3, normal: Vec3, fresh: bool },
    /// Finished breaking an ore node.
    MineNode { id: NodeId },
    /// Place a building piece. `gap`: for foundations, how far the ground falls away under it.
    Place { piece: Piece, gap: f32 },
    Dismantle { key: PieceKey },
    /// Level the building cell centred at `center` to `height`.
    Flatten { center: Vec2, height: f32 },
    /// A free-form terrain edit (the fly-mode brush, a development tool).
    Edit(EditOp),
    /// An inventory click; `seq` counts them, so the client can tell which of its own ops a
    /// snapshot already includes.
    Inventory { seq: u32, op: InventoryOp },
    /// Set the time of day (development).
    SetTime(f32),
    /// Start the nearest active volcano's eruption (doing this), and the nearest geysers, now
    /// (development).
    Erupt(crate::volcanism::Kind),
    /// Save now.
    Save,
    /// Answered with `Pong` once everything sent before it has been handled (so scripted clients
    /// can wait for the results of what they asked).
    Ping(u32),
}

/// The state a client starts from.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Welcome {
    pub seed: i32,
    pub edits: Vec<EditOp>,
    pub pieces: Vec<Piece>,
    pub mined: Vec<NodeId>,
    pub ticks: u64,
    pub time_of_day: f32,
    /// This client's player.
    pub feet: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub mode: Mode,
    pub inventory: InventorySnapshot,
    pub vitals: crate::vitals::Vitals,
    /// Where things left at death lie.
    pub stashes: Vec<(u32, Vec3)>,
    /// Every materialised fluid chunk (everything else is pristine).
    pub fluids: Vec<(glam::IVec3, asc_world::fluid::PackedFluid)>,
    /// Ground that's burning or burnt: fire cell and stage (see `fire`).
    pub fire: Vec<([i32; 2], crate::fire::Stage)>,
    /// Where pyroclastic flows and blasts have passed.
    pub scars: Vec<crate::volcanism::Scar>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ServerMsg {
    Welcome(Box<Welcome>),
    Edited(EditOp),
    PiecePlaced(Piece),
    PieceRemoved(PieceKey),
    NodeMined(NodeId),
    Inventory(InventorySnapshot),
    Time { ticks: u64, time_of_day: f32 },
    /// Something to tell the player ("Inventory full", "Can't place: ...").
    Notice(String),
    Saved { ok: bool },
    Pong(u32),
    /// The player's last move was refused: put them back here, in this mode.
    Correct { feet: Vec3, mode: Mode },
    /// Once a second: ticks per second, and the slowest tick (ms).
    Stats { tps: f32, mspt: f32 },
    /// The player's health, air and temperature changed.
    Vitals(crate::vitals::Vitals),
    /// Something hurt the player.
    Hurt { amount: i32, cause: crate::vitals::Cause },
    /// The player died (a `Correct` to the spawn point follows).
    Died { cause: crate::vitals::Cause },
    StashPlaced { id: u32, at: Vec3 },
    StashTaken { id: u32 },
    /// A fluid chunk changed (or came to life).
    Fluid { coord: glam::IVec3, data: asc_world::fluid::PackedFluid },
    /// Volcanoes near the player that aren't quiet, bombs in the air and blowing geysers (each tick
    /// while there's any; one empty report when it's all over).
    Volcanism(crate::volcanism::Report),
    /// Fire cells whose stage changed (caught, burning hard, burnt out).
    Fire(Vec<([i32; 2], crate::fire::Stage)>),
    /// A pyroclastic flow or blast has scarred the land.
    Scar(crate::volcanism::Scar),
    /// Lava met water and set to rock here this tick (steam: a few spots, a cell apart at least).
    Steam(Vec<glam::Vec3>),
}
