//! The server: the authoritative game state, the rules applied to what clients ask for, the
//! fixed-rate tick and saving.
//!
//! One player for now (single player), but everything a client learns comes through `ServerMsg`s,
//! so the same server can later have remote players.

use std::collections::BTreeSet;
use std::path::PathBuf;

use anyhow::Result;
use asc_build::{Lattice, Snap, Structures};
use asc_world::World;
use std::time::{Duration, Instant};

use glam::{IVec3, Vec3};

use crate::content::{Content, ItemId};
use crate::inventory::{HOTBAR, Inventory, SLOTS, Stack};
use crate::ore_nodes::{self, NodeId, NodeKind};
use crate::protocol::{ClientMsg, ServerMsg, Welcome};
use crate::rules::{self, FillOutcome};
use crate::save::{self, PlayerSave, SaveGame, SlotSave, WorldMeta};
use crate::sim::Sim;
use crate::{EYE_HEIGHT, Mode};

/// Ticks per second.
pub const TPS: u32 = 20;
/// Autosave interval, in ticks (two minutes).
const AUTOSAVE_TICKS: u64 = 120 * TPS as u64;
/// How far from the eye a player can mine, fill or break a node (a little past the client's own
/// 5 m reach, for movement between the client's check and this one).
const ACTION_REACH: f32 = 7.5;
/// How far from the eye a player can build (client: 10 m).
const BUILD_REACH: f32 = 14.0;
/// Actions may come this much faster than their nominal time (timer jitter between client and
/// server).
const RATE_TOLERANCE: f32 = 0.8;
/// Placing repeats this often while held (matches the client).
const FILL_INTERVAL: f32 = 0.2;
/// Movement budgets, metres per second, and the most a budget can save up (bursts after a stall).
/// Walking tops out at 8 m/s sprinting; up covers jumps and step-ups; down covers falling at the
/// 60 m/s terminal speed.
const MAX_ACROSS: (f32, f32) = (12.0, 3.0);
const MAX_UP: (f32, f32) = (12.0, 3.0);
const MAX_DOWN: (f32, f32) = (70.0, 10.0);
/// A body centre this deep in terrain or a building is inside it, not brushing it.
const INSIDE: f32 = -0.25;

/// Distance a player may still cover, refilled over time.
#[derive(Clone, Copy, Debug)]
struct Budget {
    across: f32,
    up: f32,
    down: f32,
    at: Instant,
}

impl Budget {
    fn new() -> Self {
        Self { across: MAX_ACROSS.1, up: MAX_UP.1, down: MAX_DOWN.1, at: Instant::now() }
    }

    /// Refill for the time since the last move, then pay for `d` if it can be afforded.
    fn spend(&mut self, d: Vec3) -> bool {
        let now = Instant::now();
        let dt = (now - self.at).as_secs_f32();
        self.at = now;
        let refill = |v: f32, (rate, cap): (f32, f32)| (v + rate * dt).min(cap);
        let (across, up, down) =
            (refill(self.across, MAX_ACROSS), refill(self.up, MAX_UP), refill(self.down, MAX_DOWN));
        let (h, rise, fall) = (Vec3::new(d.x, 0.0, d.z).length(), d.y.max(0.0), (-d.y).max(0.0));
        (self.across, self.up, self.down) = (across, up, down);
        if h > across || rise > up || fall > down {
            return false;
        }
        (self.across, self.up, self.down) = (across - h, up - rise, down - fall);
        true
    }
}
pub const DEFAULT_SEED: i32 = 1337;

pub struct Config {
    pub save_dir: PathBuf,
    /// Seed for a new world (a saved world keeps its own).
    pub seed: Option<i32>,
    /// Move any existing world aside and start a new one.
    pub new_world: bool,
    /// Check players' reach, action rates and movement. Off for scripted tests and tools that build
    /// far away or teleport.
    pub validate: bool,
    /// The local player (the host) may fly and use the development tools (terrain brush, setting
    /// the time). Remote players, when there are any, won't by default.
    pub cheats: bool,
    /// Run a stand-in factory every tick, to measure the tick (`--tick-stress`).
    pub stress: Option<crate::stress::StressConfig>,
}

/// The one player, for now.
pub struct PlayerState {
    pub feet: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub mode: Mode,
    pub inventory: Inventory,
    /// The stack on the inventory screen's cursor.
    pub held: Option<Stack>,
    /// The last inventory op applied.
    pub seq: u32,
    /// May fly and use the development tools.
    pub cheats: bool,
    /// When the last bite or node was accepted, and the last fill.
    last_dig: Option<Instant>,
    last_fill: Option<Instant>,
    budget: Budget,
}

impl PlayerState {
    fn eye(&self) -> Vec3 {
        self.feet + Vec3::Y * EYE_HEIGHT
    }

    /// Whether an action of `seconds` nominal length may happen now, given the last one at `last`;
    /// starts its clock if so.
    fn ready(last: &mut Option<Instant>, seconds: f32) -> bool {
        let now = Instant::now();
        if last.is_some_and(|t| now - t < Duration::from_secs_f32(seconds * RATE_TOLERANCE)) {
            return false;
        }
        *last = Some(now);
        true
    }
}

pub struct Server {
    pub world: World,
    pub structures: Structures,
    pub mined: BTreeSet<[i32; 3]>,
    pub sim: Sim,
    pub content: Content,
    pub player: PlayerState,
    pub spawn: Vec3,
    seed: i32,
    save_dir: PathBuf,
    validate: bool,
    stress: Option<crate::stress::Stress>,
    last_save: u64,
    outbox: Vec<ServerMsg>,
}

impl Server {
    /// Load the world in `config.save_dir`, or start a new one there.
    pub fn open(config: Config, lattice: Lattice) -> Result<Self> {
        let content = Content::load()?;
        log::info!("content: {} items", content.items.len());
        let save_dir = config.save_dir;
        if config.new_world && save_dir.exists() {
            let stamp =
                std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map_or(0, |d| d.as_secs());
            let name = save_dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            let aside = save_dir.with_file_name(format!("{name}-replaced-{stamp}"));
            std::fs::rename(&save_dir, &aside)?;
            log::info!("new world: kept the old one as {}", aside.display());
        }
        let loaded = save::load(&save_dir)?;
        let seed = loaded.as_ref().map_or(config.seed.unwrap_or(DEFAULT_SEED), |g| g.meta.seed);
        let mut world = World::new(seed);
        let mut sim = Sim::default();
        let mut structures = Structures::new(lattice.clone());
        let mut mined = BTreeSet::new();
        match &loaded {
            Some(game) => {
                world.edits = asc_world::EditLog::from_ops(game.ops.iter().copied());
                structures = Structures::from_pieces(lattice, game.structures.iter().copied());
                mined = game.nodes.iter().copied().collect();
                sim.ticks = game.meta.ticks;
                sim.time_of_day = game.meta.time_of_day;
                log::info!(
                    "loaded {} ({} edits, {} pieces, {} nodes mined)",
                    save_dir.display(),
                    game.ops.len(),
                    structures.len(),
                    mined.len()
                );
            }
            None => log::info!("new world {} (seed {seed})", save_dir.display()),
        }
        let spawn = rules::find_ground(&world, 0.0, 0.0);
        let mut player = PlayerState {
            feet: spawn,
            yaw: 0.0,
            pitch: -0.15,
            mode: Mode::Walk,
            inventory: Inventory::default(),
            held: None,
            seq: 0,
            cheats: config.cheats,
            last_dig: None,
            last_fill: None,
            budget: Budget::new(),
        };
        if let Some(saved) = loaded.as_ref().and_then(|g| g.players.get(save::LOCAL_PLAYER)) {
            restore_player(saved, &content, &mut player);
        }
        let ticks = sim.ticks;
        Ok(Self {
            world,
            structures,
            mined,
            last_save: ticks,
            sim,
            content,
            player,
            spawn,
            seed,
            save_dir,
            validate: config.validate,
            stress: config.stress.map(|c| {
                log::info!("tick stress: {c:?}");
                crate::stress::Stress::new(c, ticks)
            }),
            outbox: Vec::new(),
        })
    }

    pub fn seed(&self) -> i32 {
        self.seed
    }

    /// What a client joining now needs.
    pub fn welcome(&self) -> Welcome {
        let p = &self.player;
        Welcome {
            seed: self.seed,
            edits: self.world.edits.ops().to_vec(),
            pieces: self.structures.iter().copied().collect(),
            mined: self.mined.iter().map(|n| IVec3::from_array(*n)).collect(),
            ticks: self.sim.ticks,
            time_of_day: self.sim.time_of_day,
            feet: p.feet,
            yaw: p.yaw,
            pitch: p.pitch,
            mode: p.mode,
            inventory: p.inventory.snapshot(p.held, p.seq),
        }
    }

    /// Messages for the client since the last call.
    pub fn drain(&mut self) -> std::vec::Drain<'_, ServerMsg> {
        self.outbox.drain(..)
    }

    fn notice(&mut self, text: String) {
        self.outbox.push(ServerMsg::Notice(text));
    }

    fn send_inventory(&mut self) {
        let p = &self.player;
        self.outbox.push(ServerMsg::Inventory(p.inventory.snapshot(p.held, p.seq)));
    }

    fn within(&self, p: Vec3, reach: f32) -> bool {
        !self.validate || self.player.eye().distance(p) <= reach
    }

    /// One fixed step of the simulation.
    pub fn tick(&mut self) {
        self.sim.tick();
        if let Some(stress) = &mut self.stress {
            stress.tick();
        }
        self.outbox.push(ServerMsg::Time { ticks: self.sim.ticks, time_of_day: self.sim.time_of_day });
        if self.sim.ticks - self.last_save >= AUTOSAVE_TICKS {
            self.save("autosaved");
        }
    }

    pub fn handle(&mut self, msg: ClientMsg) {
        match msg {
            ClientMsg::Move { feet, yaw, pitch, mode } => self.moved(feet, yaw, pitch, mode),
            ClientMsg::Bite { point, normal } => {
                if !self.within(point, ACTION_REACH) {
                    log::warn!("bite at {point} refused: out of reach");
                    return;
                }
                let (_, byte) = self.world.sample_raw(point - normal * 0.3);
                if byte == asc_world::Material::Air as u8 {
                    return;
                }
                let hardness = self.content.sample(byte).hardness;
                if self.validate && !PlayerState::ready(&mut self.player.last_dig, hardness) {
                    log::warn!("bite at {point} refused: too soon after the last");
                    return;
                }
                let bite = rules::bite(&mut self.world, &self.content, &mut self.player.inventory, point, normal);
                log::debug!("bite at {} removed {:.2} m3", bite.op.center, bite.removed.total_solid());
                self.outbox.push(ServerMsg::Edited(bite.op));
                self.send_inventory();
                if bite.overflow > 0 {
                    self.notice(format!("Inventory full: {} items lost", bite.overflow));
                }
            }
            ClientMsg::Fill { point, normal, fresh } => {
                if !self.within(point, ACTION_REACH)
                    || (self.validate && !PlayerState::ready(&mut self.player.last_fill, FILL_INTERVAL))
                {
                    return;
                }
                let feet = self.player.feet;
                match rules::fill(&mut self.world, &self.content, &mut self.player.inventory, point, normal, feet) {
                    FillOutcome::Placed(op) => {
                        self.outbox.push(ServerMsg::Edited(op));
                        self.send_inventory();
                    }
                    FillOutcome::NotEnough if fresh => {
                        let name = self
                            .player
                            .inventory
                            .selected_stack()
                            .map(|s| self.content.item(s.item).name.clone())
                            .unwrap_or_default();
                        self.notice(format!("Not enough {name} for that"));
                    }
                    FillOutcome::NotPlaceable if fresh => self.notice("That can't be placed".into()),
                    _ => {}
                }
            }
            ClientMsg::MineNode { id } => self.mine_node(id),
            ClientMsg::Place { piece, gap } => {
                let (position, _) = self.structures.lattice.placement(&piece.key);
                if !self.within(position, BUILD_REACH) {
                    return;
                }
                let snap = Snap { key: piece.key, grounded: piece.grounded, gap };
                match self.structures.check_snap(&snap).and_then(|()| self.structures.place(piece)) {
                    Ok(()) => self.outbox.push(ServerMsg::PiecePlaced(piece)),
                    Err(why) => self.notice(format!("Can't place: {why}")),
                }
            }
            ClientMsg::Dismantle { key } => {
                let (position, _) = self.structures.lattice.placement(&key);
                if !self.within(position, BUILD_REACH) {
                    return;
                }
                match self.structures.dismantle(&key) {
                    Ok(_) => self.outbox.push(ServerMsg::PieceRemoved(key)),
                    Err(why) => self.notice(format!("Can't dismantle: {why}")),
                }
            }
            ClientMsg::Flatten { center, height } => {
                let target = Vec3::new(center.x, height, center.y);
                if !self.within(target, BUILD_REACH) {
                    return;
                }
                for op in rules::flatten_ops(center, self.structures.lattice.cell / 2.0, height) {
                    self.world.edits.push(op);
                    self.outbox.push(ServerMsg::Edited(op));
                }
            }
            ClientMsg::Edit(op) => {
                // The fly-mode brush, a development tool.
                if self.player.cheats || !self.validate {
                    self.world.edits.push(op);
                    self.outbox.push(ServerMsg::Edited(op));
                }
            }
            ClientMsg::Inventory { seq, op } => {
                let p = &mut self.player;
                let lost = p.inventory.apply(&self.content, op, &mut p.held);
                p.seq = seq;
                self.send_inventory();
                if lost > 0 {
                    self.notice(format!("Inventory full: {lost} items lost"));
                }
            }
            ClientMsg::SetTime(t) => {
                if !self.player.cheats && self.validate {
                    return;
                }
                self.sim.time_of_day = t.rem_euclid(1.0);
                self.outbox.push(ServerMsg::Time { ticks: self.sim.ticks, time_of_day: self.sim.time_of_day });
            }
            ClientMsg::Save => self.save("saved"),
            ClientMsg::Ping(n) => self.outbox.push(ServerMsg::Pong(n)),
        }
    }

    /// A player's reported position, checked: no flying without permission, no faster than they
    /// can move (except straight back to spawn, after falling out of the world), and not inside
    /// terrain or a building. Refused moves put the player back where they were.
    fn moved(&mut self, feet: Vec3, yaw: f32, pitch: f32, mode: Mode) {
        let p = &mut self.player;
        (p.yaw, p.pitch) = (yaw, pitch);
        if !self.validate {
            (p.feet, p.mode) = (feet, mode);
            return;
        }
        let mode = if mode == Mode::Fly && !p.cheats { Mode::Walk } else { mode };
        let respawn = feet.distance(self.spawn) < 1.0;
        let free = p.cheats && mode == Mode::Fly;
        let center = feet + Vec3::Y * 0.9;
        let inside = !free
            && (self.world.density_at(center) < INSIDE || self.structures.sdf(center) < INSIDE);
        let fast = !free && !respawn && !p.budget.spend(feet - p.feet);
        if inside || fast || mode != Mode::Fly && p.mode == Mode::Fly && !p.cheats {
            log::warn!(
                "move to {feet} refused ({}); back to {}",
                if inside { "inside something" } else { "too fast" },
                p.feet
            );
            p.mode = mode;
            self.outbox.push(ServerMsg::Correct { feet: p.feet, mode });
            return;
        }
        if mode != p.mode && mode == Mode::Walk && !p.cheats {
            self.outbox.push(ServerMsg::Correct { feet, mode });
        }
        (p.feet, p.mode) = (feet, mode);
    }

    fn mine_node(&mut self, id: NodeId) {
        if self.mined.contains(&id.to_array()) {
            return;
        }
        let Some((kind, node)) = ore_nodes::find(ore_nodes::seed_of(self.seed), &self.world.worldgen, id) else {
            log::warn!("no ore node at {id}");
            return;
        };
        if !self.within(node.position, ACTION_REACH + 1.0) {
            return;
        }
        let hardness = self.content.ores[node.ore].hardness
            * match kind {
                NodeKind::Surface => rules::SURFACE_NODE_HARDNESS,
                NodeKind::Deep => rules::DEEP_NODE_HARDNESS,
            };
        if self.validate && !PlayerState::ready(&mut self.player.last_dig, hardness) {
            log::warn!("node {id} refused: too soon after the last dig");
            return;
        }
        self.mined.insert(id.to_array());
        self.outbox.push(ServerMsg::NodeMined(id));
        let count = match kind {
            NodeKind::Surface => rules::SURFACE_NODE_YIELD,
            NodeKind::Deep => rules::DEEP_NODE_YIELD,
        };
        if let Some(item) = self.content.ores[node.ore].drop {
            let lost = self.player.inventory.add(&self.content, item, count);
            self.send_inventory();
            if lost > 0 {
                self.notice(format!("Inventory full: {lost} items lost"));
            }
        }
        log::debug!("mined a {kind:?} {} node", asc_world::ores::builtin().ores[node.ore].name);
    }

    fn snapshot(&self) -> SaveGame {
        let p = &self.player;
        let mut inventory = p.inventory.clone();
        // A stack on the cursor belongs to the player too.
        if let Some(h) = p.held {
            inventory.add(&self.content, h.item, h.count);
        }
        let item_id = |id: ItemId| self.content.item(id).id.clone();
        SaveGame {
            meta: WorldMeta {
                format_version: save::FORMAT_VERSION,
                worldgen_version: asc_world::WORLDGEN_VERSION,
                seed: self.seed,
                ticks: self.sim.ticks,
                time_of_day: self.sim.time_of_day,
            },
            players: [(save::LOCAL_PLAYER.to_owned(), PlayerSave {
                mode: p.mode,
                feet: p.feet.into(),
                yaw: p.yaw,
                pitch: p.pitch,
                selected: inventory.selected,
                slots: inventory
                    .slots
                    .iter()
                    .enumerate()
                    .filter_map(|(index, s)| s.map(|s| SlotSave { index, item: item_id(s.item), count: s.count }))
                    .collect(),
                remainders: inventory.remainders.iter().map(|(id, v)| (item_id(*id), *v)).collect(),
            })]
            .into(),
            ops: self.world.edits.ops().to_vec(),
            structures: self.structures.iter().copied().collect(),
            nodes: self.mined.iter().copied().collect(),
        }
    }

    /// Save now; `what` names it in the log ("saved", "autosaved").
    pub fn save(&mut self, what: &str) {
        let t = std::time::Instant::now();
        let ok = match save::save(&self.save_dir, &self.snapshot()) {
            Ok(()) => {
                log::info!(
                    "{what} {} ({} edits) in {:.1} ms",
                    self.save_dir.display(),
                    self.world.edits.len(),
                    t.elapsed().as_secs_f32() * 1e3
                );
                true
            }
            Err(err) => {
                log::error!("saving failed: {err:#}");
                self.notice("Saving failed (see log)".into());
                false
            }
        };
        self.last_save = self.sim.ticks;
        self.outbox.push(ServerMsg::Saved { ok });
    }
}

/// Put a saved player and inventory back. Unknown items (removed from the content files) are dropped
/// with a warning rather than failing the load.
fn restore_player(saved: &PlayerSave, content: &Content, player: &mut PlayerState) {
    player.feet = Vec3::from(saved.feet);
    player.mode = saved.mode;
    (player.yaw, player.pitch) = (saved.yaw, saved.pitch);
    let inventory = &mut player.inventory;
    inventory.selected = saved.selected.min(HOTBAR - 1);
    for slot in &saved.slots {
        match content.item_id(&slot.item) {
            Some(item) if slot.index < SLOTS && slot.count > 0 => {
                let count = slot.count.min(content.item(item).stack);
                inventory.slots[slot.index] = Some(Stack { item, count });
            }
            _ => log::warn!("dropping saved slot {} ({} x{}): unknown item or slot", slot.index, slot.item, slot.count),
        }
    }
    for (name, value) in &saved.remainders {
        if let Some(item) = content.item_id(name) {
            inventory.remainders.insert(item, *value);
        }
    }
}

#[cfg(test)]
mod tests;
