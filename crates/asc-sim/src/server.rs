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
use crate::vitals::{Cause, Outcome, Vitals};
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
/// Walking this close to a stash picks it up.
const STASH_REACH: f32 = 2.0;

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
    pub vitals: Vitals,
    /// The highest point since last standing on something (for fall damage); None when grounded or
    /// flying.
    fall_from: Option<f32>,
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
    /// What players left where they died.
    pub stashes: Vec<save::StashSave>,
    /// Water (and later lava) where something has disturbed it.
    pub fluid_sim: crate::fluids::FluidSim,
    volcanism: crate::volcanism::Volcanism,
    fire: crate::fire::Fire,
    /// Where pyroclastic flows and blasts have passed.
    scars: Vec<crate::volcanism::Scar>,
    /// The wild creatures near the player (not saved: they're regenerated where they live).
    creatures: crate::creatures::Creatures,
    next_stash: u32,
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
        let stashes = loaded.as_ref().map_or_else(Vec::new, |g| g.stashes.clone());
        let mut fluid_sim = crate::fluids::FluidSim::new();
        let mut fire = crate::fire::Fire::new(seed);
        let mut volcanism = crate::volcanism::Volcanism::new(seed);
        if let Some(game) = &loaded {
            volcanism.restore(&world, &game.scars);
        }
        let next_stash = stashes.iter().map(|s| s.id + 1).max().unwrap_or(0);
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
        // After the edits: the fluid chunks' solid cells come from the terrain as edited.
        if let Some(game) = &loaded {
            fluid_sim.restore(&world, game.fluids.clone());
            fire.restore(game.burnt.iter().copied());
        }
        let spawn = rules::spawn_point(&world);
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
            vitals: Vitals::default(),
            fall_from: None,
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
            stashes,
            fluid_sim,
            volcanism,
            fire,
            scars: loaded.as_ref().map_or_else(Vec::new, |g| g.scars.clone()),
            creatures: crate::creatures::Creatures::new(seed),
            next_stash,
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
            vitals: p.vitals,
            stashes: self.stashes.iter().map(|s| (s.id, Vec3::from(s.at))).collect(),
            fire: self.fire.stages(self.sim.ticks),
            scars: self.scars.clone(),
            creatures: self.creatures.all().copied().collect(),
            fluids: {
                // In coordinate order: the same state always makes the same message.
                let mut f: Vec<_> = self
                    .fluid_sim
                    .fluids
                    .chunks
                    .iter()
                    .map(|(c, ch)| (*c, asc_world::fluid::PackedFluid::pack(ch)))
                    .collect();
                f.sort_by_key(|(c, _)| c.to_array());
                f
            },
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
        let (events, report) =
            self.volcanism.tick(&self.world, &mut self.fluid_sim, self.player.feet, self.sim.ticks);
        if let Some(report) = report {
            self.outbox.push(ServerMsg::Volcanism(report));
        }
        for notice in events.notices {
            self.outbox.push(ServerMsg::Notice(notice));
        }
        for (amount, cause) in events.hurt {
            let outcome = self.player.vitals.hurt(amount, cause);
            let died = matches!(outcome, Outcome::Died(_));
            self.apply(outcome);
            if died {
                // The rest was aimed where they fell, not at spawn.
                break;
            }
        }
        // A blast's collapsed flank: the terrain cut away for good.
        for op in events.edits {
            self.world.edits.push(op);
            self.fluid_sim.disturb(&self.world, &op);
            self.outbox.push(ServerMsg::Edited(op));
        }
        for scar in events.scars {
            self.scars.push(scar);
            self.outbox.push(ServerMsg::Scar(scar));
        }
        // Wild creatures come and go round the player and go about their business.
        let changes = self.creatures.tick(&self.world, self.player.feet, self.sim.ticks);
        if !changes.spawned.is_empty() || !changes.gone.is_empty() {
            self.outbox.push(ServerMsg::Creatures { spawned: changes.spawned, gone: changes.gone });
        }
        if !changes.acts.is_empty() {
            self.outbox.push(ServerMsg::CreatureActs(changes.acts));
        }
        // Fire: bombs and spilled lava set what grows where they land alight; it spreads.
        let ticks = self.sim.ticks;
        for at in events.landed {
            self.fire.ignite(&self.world.worldgen, at.x, at.z, ticks);
        }
        if ticks.is_multiple_of(20) {
            for c in self.fluid_sim.lava_on_ground() {
                self.fire.ignite(&self.world.worldgen, c.x as f32, c.z as f32, ticks);
            }
        }
        let burn = self.fire.tick(&self.world.worldgen, ticks, self.player.feet);
        if burn > 0 {
            let outcome = self.player.vitals.hurt(burn, Cause::Heat);
            self.apply(outcome);
        }
        let changed = self.fire.take_changed();
        if !changed.is_empty() {
            self.outbox.push(ServerMsg::Fire(changed));
        }
        self.fluid_sim.tick(&self.world);
        // Steam where lava met water: a few spots (one per 8 m cell), for the clients to show.
        let mut steam: Vec<Vec3> = Vec::new();
        for cell in self.fluid_sim.take_steam() {
            let p = cell.as_vec3() + Vec3::splat(0.5);
            if steam.len() < 16 && !steam.iter().any(|s| s.distance(p) < 8.0) {
                steam.push(p);
            }
        }
        if !steam.is_empty() {
            self.outbox.push(ServerMsg::Steam(steam));
        }
        for (cell, height, material) in self.fluid_sim.take_solidified() {
            // A run `height` metres up from the bottom of `cell`, as one box.
            let op = asc_world::EditOp {
                center: cell.as_vec3() + Vec3::Y * (height * 0.5 - 0.5),
                radius: 0.55,
                kind: asc_world::EditKind::FillBox { half_height: height * 0.5 + 0.05 },
                material,
            };
            self.world.edits.push(op);
            self.outbox.push(ServerMsg::Edited(op));
        }
        for coord in self.fluid_sim.take_changed() {
            if let Some(ch) = self.fluid_sim.fluids.chunks.get(&coord) {
                self.outbox.push(ServerMsg::Fluid { coord, data: asc_world::fluid::PackedFluid::pack(ch) });
            }
        }
        let before = self.player.vitals;
        let head = self.player.eye();
        let (head_kind, head_fill) = self.fluid_sim.fraction_at(&self.world, head);
        let underwater = head_kind == asc_world::fluid::FluidKind::Water && head_fill > 0.5;
        let outcome = self.player.vitals.tick(underwater);
        self.apply(outcome);
        // Lava burns fast: anything of the body in it.
        let body = self.player.feet + Vec3::Y * 0.5;
        let (k, fill) = self.fluid_sim.fraction_at(&self.world, body);
        if k == asc_world::fluid::FluidKind::Lava && fill > 0.1 && self.sim.ticks.is_multiple_of(5) {
            let outcome = self.player.vitals.hurt(60, Cause::Lava);
            self.apply(outcome);
        }
        if self.player.vitals != before {
            self.outbox.push(ServerMsg::Vitals(self.player.vitals));
        }
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
                self.fluid_sim.disturb(&self.world, &bite.op);
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
                        self.fluid_sim.disturb(&self.world, &op);
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
                    self.fluid_sim.disturb(&self.world, &op);
                    self.outbox.push(ServerMsg::Edited(op));
                }
            }
            ClientMsg::Edit(op) => {
                // The fly-mode brush, a development tool.
                if self.player.cheats || !self.validate {
                    self.world.edits.push(op);
                    self.fluid_sim.disturb(&self.world, &op);
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
            ClientMsg::Erupt(kind) => {
                if !self.player.cheats && self.validate {
                    return;
                }
                match self.volcanism.force(&self.world, self.player.feet, self.sim.ticks, kind) {
                    Some(at) => {
                        log::info!("forced an eruption at {at}");
                        self.outbox.push(ServerMsg::Notice(format!("Eruption started at ({:.0}, {:.0}, {:.0})", at.x, at.y, at.z)));
                    }
                    None => self.outbox.push(ServerMsg::Notice("No active volcano near".into())),
                }
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
            self.track_fall();
            self.pick_up_stashes();
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
        self.track_fall();
        self.pick_up_stashes();
    }

    /// Fall damage: remember the highest point while airborne, and charge for the drop on landing.
    /// Grounded is checked here, against the world and buildings, not taken from the client.
    fn track_fall(&mut self) {
        let p = &mut self.player;
        if p.mode == Mode::Fly || !p.vitals.alive() {
            p.fall_from = None;
            return;
        }
        let below = p.feet - Vec3::Y * 0.15;
        // Landing in water doesn't hurt: it counts as ground, and so does swimming.
        let in_water = self.fluid_sim.fraction_at(&self.world, p.feet + Vec3::Y * 0.5).1 > 0.3;
        let grounded = in_water || self.world.density_at(below) < 0.0 || self.structures.sdf(below) < 0.05;
        if in_water {
            p.fall_from = None;
            return;
        }
        match (grounded, p.fall_from) {
            (false, from) => p.fall_from = Some(from.map_or(p.feet.y, |f| f.max(p.feet.y))),
            (true, Some(from)) => {
                p.fall_from = None;
                let height = from - p.feet.y;
                let outcome = p.vitals.hurt(Vitals::fall_damage(height), Cause::Fall);
                self.apply(outcome);
            }
            (true, None) => {}
        }
    }

    /// Damage and death: tell the client; on death leave the inventory where the player fell and
    /// start them again at spawn.
    fn apply(&mut self, outcome: Outcome) {
        match outcome {
            Outcome::Fine => {}
            Outcome::Hurt(amount, cause) => {
                self.outbox.push(ServerMsg::Hurt { amount, cause });
                self.outbox.push(ServerMsg::Vitals(self.player.vitals));
            }
            Outcome::Died(cause) => self.die(cause),
        }
    }

    fn die(&mut self, cause: Cause) {
        let at = self.player.feet;
        log::info!("player {} at {at}", cause.describe());
        let p = &mut self.player;
        if let Some(h) = p.held.take() {
            p.inventory.add(&self.content, h.item, h.count);
        }
        let item_id = |id: ItemId| self.content.item(id).id.clone();
        let slots: Vec<SlotSave> = p
            .inventory
            .slots
            .iter()
            .enumerate()
            .filter_map(|(index, s)| s.map(|s| SlotSave { index, item: item_id(s.item), count: s.count }))
            .collect();
        let remainders = p.inventory.remainders.iter().map(|(id, v)| (item_id(*id), *v)).collect();
        let selected = p.inventory.selected;
        p.inventory = Inventory { selected, ..Inventory::default() };
        if !slots.is_empty() {
            let id = self.next_stash;
            self.next_stash += 1;
            self.stashes.push(save::StashSave { id, at: at.into(), slots, remainders });
            self.outbox.push(ServerMsg::StashPlaced { id, at });
        }
        p.vitals = Vitals::default();
        p.fall_from = None;
        p.feet = self.spawn;
        p.budget = Budget::new();
        p.mode = Mode::Walk;
        self.outbox.push(ServerMsg::Died { cause });
        self.outbox.push(ServerMsg::Correct { feet: self.spawn, mode: Mode::Walk });
        self.outbox.push(ServerMsg::Vitals(self.player.vitals));
        self.send_inventory();
    }

    /// Walking over a stash gives back whatever of it fits.
    fn pick_up_stashes(&mut self) {
        let feet = self.player.feet;
        let Some(i) = self.stashes.iter().position(|s| Vec3::from(s.at).distance(feet) < STASH_REACH) else {
            return;
        };
        let mut stash = self.stashes.remove(i);
        let inv = &mut self.player.inventory;
        let mut kept = Vec::new();
        for slot in stash.slots.drain(..) {
            match self.content.item_id(&slot.item) {
                Some(item) => {
                    let left = inv.add(&self.content, item, slot.count);
                    if left > 0 {
                        kept.push(SlotSave { count: left, ..slot });
                    }
                }
                None => log::warn!("stash item {} no longer exists", slot.item),
            }
        }
        for (name, v) in std::mem::take(&mut stash.remainders) {
            if let Some(item) = self.content.item_id(&name) {
                *inv.remainders.entry(item).or_insert(0.0) += v;
            }
        }
        self.send_inventory();
        if kept.is_empty() {
            self.outbox.push(ServerMsg::StashTaken { id: stash.id });
            self.notice("Picked up your things".into());
        } else {
            stash.slots = kept;
            self.notice("Inventory full: some of your things are still there".into());
            self.stashes.push(stash);
        }
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
                vitals: Some(p.vitals),
            })]
            .into(),
            ops: self.world.edits.ops().to_vec(),
            structures: self.structures.iter().copied().collect(),
            nodes: self.mined.iter().copied().collect(),
            stashes: self.stashes.clone(),
            fluids: {
                let mut f: Vec<_> = self
                    .fluid_sim
                    .fluids
                    .chunks
                    .iter()
                    .map(|(c, ch)| (*c, ch.kind.clone(), ch.amount.clone()))
                    .collect();
                f.sort_by_key(|(c, _, _)| c.to_array());
                f
            },
            burnt: self.fire.stages(self.sim.ticks).into_iter().map(|(c, _)| c).collect(),
            scars: self.scars.clone(),
        }
    }

    /// Set the ground at a world point alight (development: the fire cheat).
    pub fn ignite(&mut self, at: Vec3) {
        self.fire.ignite(&self.world.worldgen, at.x, at.z, self.sim.ticks);
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
    player.vitals = saved.vitals.unwrap_or_default();
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
