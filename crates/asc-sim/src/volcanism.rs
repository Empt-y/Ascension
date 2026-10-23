//! Volcanoes over time: active cones erupt now and then, and calderas' geysers blow.
//!
//! Some eruptions are worse. About a third send pyroclastic flows racing down the mountain (a
//! ground-hugging avalanche of hot ash, death to anyone it catches), and about one in twelve opens
//! with a lateral blast like Mount St Helens': the flank on its breach side collapses (the
//! terrain is cut away for good, leaving a horseshoe crater) and a shockwave races out at the
//! speed of sound, lethal for kilometres in front of the mountain. Flows and blasts leave scars
//! where nothing survives.
//!
//! Each active cone keeps its own clock, worked out from the seed and the tick count (so nothing
//! about it needs saving): a long quiet spell, then a rumbling warning, an eruption and a waning.
//! Erupting, it pours lava out of vents at the low points of its crater rim, which runs down the
//! flanks as simulated fluid (and sets into basalt as it cools), and throws volcanic bombs that
//! hurt whoever they land near. Geysers blow for a few seconds every few minutes, throwing water up
//! over their vents and scalding anyone standing over them. Only volcanoes near the player are
//! followed.

use std::sync::Arc;

use asc_world::fluid::{FULL, FluidKind};
use asc_world::volcano::{FeatureKind, Volcano, VolcanoKind};
use asc_world::World;
use glam::{IVec3, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::fluids::FluidSim;

/// Ticks from one eruption's start to the next (40–70 minutes at 20 ticks a second).
const ERUPTION_CYCLE: [u64; 2] = [48_000, 84_000];
pub const RUMBLE_TICKS: u64 = 1200;
pub const ERUPT_TICKS: u64 = 4800;
pub const WANE_TICKS: u64 = 1200;
/// Geysers blow every 3–6 minutes, for 12 seconds.
const GEYSER_CYCLE: [u64; 2] = [3600, 7200];
const GEYSER_BLOW: u64 = 240;
/// Pyroclastic flows and lateral blasts (see the module doc).
const FLOW_CHANCE: f32 = 0.35;
const BLAST_CHANCE: f32 = 0.08;
/// A flow's speed at first (m/s; it slows as it spreads), how far it runs (in cone radii), how
/// wide it gets at its end (m), and how deep its deadly front is (m).
const FLOW_SPEED: f32 = 45.0;
const FLOW_REACH: [f32; 2] = [4.0, 6.0];
const FLOW_WIDTH: f32 = 260.0;
const FLOW_FRONT: f32 = 350.0;
/// A flow's cloud is deadly up to this high over the ground.
const FLOW_HEIGHT: f32 = 60.0;
/// The blast: shockwave speed (m/s), lethal reach in front of the mountain (within
/// `BLAST_SECTOR` radians of its direction) and behind it, and painful a little beyond.
const SHOCK_SPEED: f32 = 340.0;
const BLAST_REACH: f32 = 3000.0;
const BLAST_BEHIND: f32 = 800.0;
const BLAST_SECTOR: f32 = 1.2;
const LETHAL: i32 = 1_000_000;

/// Ground where a flow or blast passed (plants there are dead and gone).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum Scar {
    Flow { origin: Vec2, dir: Vec2, length: f32, width: f32 },
    Blast { center: Vec2, dir: Vec2 },
}

impl Scar {
    /// A circle round it (centre, radius).
    pub fn bounds(&self) -> (Vec2, f32) {
        match *self {
            Scar::Flow { origin, dir, length, width } => (origin + dir * length * 0.5, length * 0.5 + width),
            Scar::Blast { center, .. } => (center, BLAST_REACH),
        }
    }

    pub fn contains(&self, p: Vec2) -> bool {
        match *self {
            Scar::Flow { origin, dir, length, width } => {
                let rel = p - origin;
                let along = rel.dot(dir);
                (0.0..=length).contains(&along) && rel.perp_dot(dir).abs() < flow_half_width(along, length, width)
            }
            Scar::Blast { center, dir } => {
                let rel = p - center;
                rel.length() < blast_reach(rel, dir) * 0.85
            }
        }
    }
}

/// Half a flow's width `along` metres down it (narrow at the summit, spreading).
fn flow_half_width(along: f32, length: f32, width: f32) -> f32 {
    width * 0.5 * (0.25 + 0.75 * (along / length).clamp(0.0, 1.0))
}

/// How far the blast is lethal in the direction of `rel` (from its centre).
fn blast_reach(rel: Vec2, dir: Vec2) -> f32 {
    let off = rel.normalize_or_zero().dot(dir).clamp(-1.0, 1.0).acos();
    let front = 1.0 - ((off - BLAST_SECTOR * 0.6) / (BLAST_SECTOR * 0.4)).clamp(0.0, 1.0);
    BLAST_BEHIND + (BLAST_REACH - BLAST_BEHIND) * front
}

/// A pyroclastic flow on its way down.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Flow {
    pub origin: Vec3,
    pub dir: Vec2,
    /// How far its front has got, and how far it will.
    pub front: f32,
    pub length: f32,
    pub width: f32,
}

/// A lateral blast's shockwave.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Blast {
    pub center: Vec3,
    pub dir: Vec2,
    /// How far the shockwave has got.
    pub radius: f32,
}

/// The worst an eruption will do: a lateral blast (its direction) and flows (when, and which way).
#[derive(Clone, Debug, Default, PartialEq)]
struct Plan {
    blast: Option<Vec2>,
    flows: Vec<(u64, Vec2)>,
}

/// What a forced eruption should do (cheats).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Kind {
    /// Whatever the eruption's own plan says.
    #[default]
    Natural,
    /// Pyroclastic flows.
    Flows,
    /// A lateral blast (and flows).
    Blast,
}

/// Volcanoes followed within this distance of the player, and told about within `NEAR`. Their
/// lava only pours, and geysers only blow, where the player could see it (simulating them further
/// off would only cost time).
const FOLLOW: f32 = 6000.0;
const NEAR: f32 = 3000.0;
const POUR_REACH: f32 = 2500.0;
const GEYSER_REACH: f32 = 500.0;
/// Lava welling up into the crater per tick at full strength, at each of `WELLS` places across
/// its lake (a cell is `FULL`): the lake rises, spills through the rim's breach and runs down the
/// channel below it.
const WELL_FLOW: u16 = 2500;
const WELLS: usize = 6;
/// A bomb thrown every this many ticks, and what one does to a player it lands beside.
const BOMB_EVERY: u64 = 6;
const BOMB_REACH: f32 = 6.0;
const BOMB_DAMAGE: i32 = 300;
const GRAVITY: f32 = 9.8;
const DT: f32 = 0.05;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Phase {
    Quiet,
    Rumbling,
    Erupting,
    Waning,
}

/// A volcano that isn't quiet, for the client (ash, shaking, notices).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Activity {
    /// The crater's middle at its rim's height.
    pub summit: Vec3,
    pub radius: f32,
    pub phase: Phase,
    /// 0..1 through the phase's swell and fade.
    pub strength: f32,
}

/// What the client needs each tick something is going on.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Report {
    pub active: Vec<Activity>,
    pub flows: Vec<Flow>,
    pub blasts: Vec<Blast>,
    /// Bombs in the air: where, and how big (0 smallest .. `BOMB_SIZES` - 1).
    pub bombs: Vec<(Vec3, u8)>,
    /// Blowing geysers: vent and strength.
    pub geysers: Vec<(Vec3, f32)>,
}

struct Tracked {
    volcano: Arc<Volcano>,
    /// A stable identity for its clock (its centre, to the metre).
    key: [i32; 2],
    summit: Vec3,
    /// Where lava wells up into the crater's lake when it erupts.
    vents: Vec<IVec3>,
    /// Geysers (calderas): vent cell, clock period and offset.
    geysers: Vec<(IVec3, u64, u64)>,
    last_phase: Phase,
}

struct Bomb {
    at: Vec3,
    vel: Vec3,
    size: u8,
}

/// Bomb sizes (the client has a model for each): small ones are far commoner, and big ones hit
/// harder and wider.
pub const BOMB_SIZES: u8 = 5;

/// Something that happened this tick, for the server to apply.
#[derive(Debug, Default)]
pub struct Events {
    /// Damage to the player and why.
    pub hurt: Vec<(i32, crate::vitals::Cause)>,
    pub notices: Vec<String>,
    /// Where bombs landed (they set what grows there alight).
    pub landed: Vec<Vec3>,
    /// Terrain blown away (a blast's collapsed flank).
    pub edits: Vec<asc_world::EditOp>,
    /// New scars (flows and blasts), to keep and tell clients about.
    pub scars: Vec<Scar>,
}

pub struct Volcanism {
    tracked: Vec<Tracked>,
    /// Where (and when) the tracked list was last looked up.
    refreshed: Option<(Vec2, u64)>,
    /// Eruptions brought forward (cheats): volcano key, the tick it started rumbling, and what it
    /// should do.
    forced: Vec<([i32; 2], u64, Kind)>,
    flows: Vec<(u64, Flow)>,
    blasts: Vec<(u64, Blast)>,
    /// Geysers brought forward: vent and tick.
    forced_geysers: Vec<(IVec3, u64)>,
    bombs: Vec<Bomb>,
    rng: u64,
    seed: i32,
    /// Whether the last report had anything in it (so an empty one is sent once, to clear).
    reported: bool,
    /// Volcanoes that have blown their tops off (by key): how far down their crater now is.
    blown: ahash::AHashMap<[i32; 2], f32>,
}

/// A lateral blast's cut: spheres dug out of the summit and the flank facing `dir`, a horseshoe
/// crater about halfway down the cone that opens out down the side (as Mt St Helens' did).
fn blast_digs(v: &Volcano, summit: Vec3, dir: Vec2) -> [asc_world::EditOp; 4] {
    let c = Vec2::new(summit.x, summit.z);
    std::array::from_fn(|k| {
        let along = (0.05 + 0.16 * k as f32) * v.radius;
        let at = c + dir * along;
        let radius = v.radius * (0.4 - 0.06 * k as f32);
        // Its bottom: half the cone's height down at the middle, deeper down the flank.
        let bottom = summit.y - v.height * (0.45 + 0.1 * k as f32);
        asc_world::EditOp {
            center: Vec3::new(at.x, bottom + radius, at.y),
            radius,
            kind: asc_world::EditKind::Dig,
            material: asc_world::Material::Air,
        }
    })
}

/// How far a blown volcano's summit (where its smoke and lava come from) drops: to just over the
/// new crater's floor (the plume sets off 10 m under the summit, as it does inside an old rim).
fn blown_drop(v: &Volcano) -> f32 {
    v.height * 0.45 - 12.0
}

fn hash(seed: i32, key: [i32; 2], salt: u64) -> u64 {
    let mut h = (seed as u32 as u64) ^ salt;
    for v in key {
        h ^= v as u32 as u64;
        h = h.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(31) ^ 0xBF58_476D_1CE4_E5B9;
    }
    h
}

fn unit(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// Where in a cycle of `period` ticks (offset by `offset`) a volcano is: its phase and strength.
fn phase_at(t: u64) -> (Phase, f32) {
    let swell = |t: u64, len: u64| {
        let f = t as f32 / len as f32;
        (f * 4.0).min(1.0).min((1.0 - f) * 4.0 + 0.25).clamp(0.0, 1.0)
    };
    if t < RUMBLE_TICKS {
        (Phase::Rumbling, t as f32 / RUMBLE_TICKS as f32)
    } else if t < RUMBLE_TICKS + ERUPT_TICKS {
        (Phase::Erupting, swell(t - RUMBLE_TICKS, ERUPT_TICKS))
    } else if t < RUMBLE_TICKS + ERUPT_TICKS + WANE_TICKS {
        (Phase::Waning, 1.0 - (t - RUMBLE_TICKS - ERUPT_TICKS) as f32 / WANE_TICKS as f32)
    } else {
        (Phase::Quiet, 0.0)
    }
}

impl Volcanism {
    pub fn new(seed: i32) -> Self {
        Self {
            tracked: Vec::new(),
            refreshed: None,
            forced: Vec::new(),
            forced_geysers: Vec::new(),
            flows: Vec::new(),
            blasts: Vec::new(),
            bombs: Vec::new(),
            rng: seed as u32 as u64 ^ 0x6572_7570,
            seed,
            reported: false,
            blown: Default::default(),
        }
    }

    /// Volcanoes blown open before (from the saved scars): their smoke comes out of the new
    /// crater.
    pub fn restore(&mut self, world: &World, scars: &[Scar]) {
        for scar in scars {
            if let Scar::Blast { center, .. } = *scar
                && let Some(v) = world.worldgen.volcanoes_near(center, 100.0).into_iter().find(|v| v.center.distance(center) < 1.0)
            {
                self.blown.insert(t_key(&v), blown_drop(&v));
            }
        }
    }

    fn random(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(0x1405_7B7E_F767_814F);
        unit(self.rng)
    }

    /// The phase of tracked volcano `t` at `ticks`.
    fn phase(&self, t: &Tracked, ticks: u64) -> (Phase, f32) {
        match self.clock(t, ticks) {
            Some((into, _, _)) => phase_at(into),
            None => (Phase::Quiet, 0.0),
        }
    }

    /// Where tracked volcano `t` is in its cycle at `ticks`: ticks into it, which cycle it is (for
    /// its plan), and what a forced eruption should do. `None` for anything but an active cone.
    fn clock(&self, t: &Tracked, ticks: u64) -> Option<(u64, u64, Kind)> {
        if t.volcano.kind != VolcanoKind::Active {
            return None;
        }
        if let Some((_, start, kind)) = self.forced.iter().find(|(k, _, _)| *k == t.key)
            && ticks >= *start
            && ticks - start < RUMBLE_TICKS + ERUPT_TICKS + WANE_TICKS
        {
            return Some((ticks - start, *start, *kind));
        }
        let h = hash(self.seed, t.key, 0x6379_636c);
        let period = ERUPTION_CYCLE[0] + ((ERUPTION_CYCLE[1] - ERUPTION_CYCLE[0]) as f32 * unit(h)) as u64;
        let offset = (period as f32 * unit(h.rotate_left(17))) as u64;
        Some(((ticks + offset) % period, (ticks + offset) / period, Kind::Natural))
    }

    /// What eruption `cycle` of tracked volcano `t` will do, beyond lava and bombs.
    fn plan(&self, t: &Tracked, cycle: u64, kind: Kind) -> Plan {
        let h = hash(self.seed, t.key, 0x706c_616e ^ cycle.wrapping_mul(0x2545_F491_4F6C_DD1D));
        let (roll_blast, roll_flow) = (unit(h), unit(h.rotate_left(21)));
        let blast = match kind {
            Kind::Blast => true,
            Kind::Flows => false,
            Kind::Natural => roll_blast < BLAST_CHANCE,
        };
        let flows = blast || kind == Kind::Flows || (kind == Kind::Natural && roll_flow < FLOW_CHANCE);
        // Through the weak side: the breach.
        let breach = if t.volcano.tilt == Vec2::ZERO { Vec2::X } else { t.volcano.tilt };
        let mut plan = Plan { blast: blast.then_some(breach), flows: Vec::new() };
        if flows {
            let count = 1 + (unit(h.rotate_left(33)) * 2.99) as u64;
            for k in 0..count {
                let a = unit(h.rotate_left(40 + k as u32 * 5)) * std::f32::consts::TAU;
                // A blast's flows pour out through the opened flank.
                let dir = if blast { Vec2::from_angle(breach.to_angle() + (unit(h ^ k) - 0.5) * 1.2) } else { Vec2::from_angle(a) };
                plan.flows.push((RUMBLE_TICKS + 200 + k * 700, dir));
            }
        }
        plan
    }

    /// Start what eruption plans call for at `into` ticks into tracked volcano `i`'s cycle.
    #[allow(clippy::too_many_arguments)]
    fn begin(&mut self, i: usize, into: u64, cycle: u64, kind: Kind, ticks: u64, world: &World, events: &mut Events) {
        let t = &self.tracked[i];
        let plan = self.plan(t, cycle, kind);
        let (v, summit) = (t.volcano.clone(), t.summit);
        let c = Vec2::new(summit.x, summit.z);
        if let Some(dir) = plan.blast
            && into == RUMBLE_TICKS
        {
            // The flank collapses: a horseshoe cut out of the summit and the side facing `dir`.
            events.edits.extend(blast_digs(&v, summit, dir));
            let drop = blown_drop(&v);
            self.blown.insert(t_key(&v), drop);
            self.tracked[i].summit.y -= drop;
            self.blasts.push((ticks, Blast { center: summit, dir, radius: 0.0 }));
            events.scars.push(Scar::Blast { center: c, dir });
            events.notices.push("The volcano explodes!".into());
        }
        for (k, &(at, dir)) in plan.flows.iter().enumerate() {
            if into != at {
                continue;
            }
            let length = v.radius * (FLOW_REACH[0] + (FLOW_REACH[1] - FLOW_REACH[0]) * unit(hash(self.seed, t_key(&v), cycle ^ k as u64)));
            let width = FLOW_WIDTH * (0.7 + 0.6 * unit(hash(self.seed, t_key(&v), cycle ^ 0x77 ^ k as u64)));
            self.flows.push((ticks, Flow { origin: summit, dir, front: 0.0, length, width }));
            events.scars.push(Scar::Flow { origin: c, dir, length, width });
            events.notices.push("A pyroclastic flow is racing down the mountain!".into());
            let _ = world;
        }
    }

    /// Whether geyser `g` (vent, period, offset) is blowing at `ticks`, and how hard.
    fn geyser(&self, g: (IVec3, u64, u64), ticks: u64) -> Option<f32> {
        let forced = self.forced_geysers.iter().any(|(v, s)| *v == g.0 && ticks >= *s && ticks - s < GEYSER_BLOW);
        let t = if forced {
            ticks - self.forced_geysers.iter().find(|(v, _)| *v == g.0).map_or(0, |(_, s)| *s)
        } else {
            (ticks + g.2) % g.1
        };
        (t < GEYSER_BLOW).then(|| {
            let f = t as f32 / GEYSER_BLOW as f32;
            (f * 6.0).min(1.0).min((1.0 - f) * 3.0)
        })
    }

    fn track(&self, world: &World, v: Arc<Volcano>) -> Tracked {
        let wg = &world.worldgen;
        let c = v.center;
        let key = [c.x.round() as i32, c.y.round() as i32];
        let rim_height = |a: f32, r: f32| wg.height(c.x + a.cos() * r, c.y + a.sin() * r);
        let mut summit = Vec3::new(c.x, rim_height(0.0, v.crater_radius), c.y);
        summit.y -= self.blown.get(&key).copied().unwrap_or(0.0);
        let mut vents = Vec::new();
        if let Some(lava) = v.lava.filter(|_| v.kind == VolcanoKind::Active) {
            // Wells over the lava lake, just over its surface (it rises), and across the breach
            // in the rim, where it spills over the lip and down the channel.
            for k in 0..WELLS {
                let a = k as f32 / WELLS as f32 * std::f32::consts::TAU;
                let r = v.crater_radius * 0.25;
                let (x, z) = (c.x + a.cos() * r, c.y + a.sin() * r);
                vents.push(IVec3::new(x.round() as i32, lava.ceil() as i32, z.round() as i32));
            }
            let across = Vec2::new(-v.tilt.y, v.tilt.x);
            for k in -2..=2 {
                let at = c + v.tilt * v.crater_radius * 0.9 + across * (k as f32 * 2.5);
                vents.push(IVec3::new(at.x.round() as i32, (wg.height(at.x, at.y) + 1.0).ceil() as i32, at.y.round() as i32));
            }
        }
        let geysers = v
            .features
            .iter()
            .filter(|f| f.kind == FeatureKind::Geyser)
            .map(|f| {
                let h = hash(self.seed, [f.at.x.round() as i32, f.at.y.round() as i32], 0x6765_7973);
                let period = GEYSER_CYCLE[0] + ((GEYSER_CYCLE[1] - GEYSER_CYCLE[0]) as f32 * unit(h)) as u64;
                let offset = (period as f32 * unit(h.rotate_left(23))) as u64;
                (IVec3::new(f.at.x.round() as i32, f.ground.round() as i32, f.at.y.round() as i32), period, offset)
            })
            .collect();
        Tracked { volcano: v, key, summit, vents, geysers, last_phase: Phase::Quiet }
    }

    /// Bring the nearest active volcano's eruption (doing `kind`), and the nearest geysers,
    /// forward to now.
    pub fn force(&mut self, world: &World, player: Vec3, ticks: u64, kind: Kind) -> Option<Vec3> {
        let near = world.worldgen.volcanoes_near(Vec2::new(player.x, player.z), 20_000.0);
        let p = Vec2::new(player.x, player.z);
        let active = near.iter().filter(|v| v.kind == VolcanoKind::Active).min_by(|a, b| a.center.distance(p).total_cmp(&b.center.distance(p)));
        let mut at = None;
        if let Some(v) = active {
            let key = [v.center.x.round() as i32, v.center.y.round() as i32];
            self.forced.retain(|(k, _, _)| *k != key);
            self.forced.push((key, ticks, kind));
            at = Some(Vec3::new(v.center.x, world.worldgen.height(v.center.x + v.crater_radius, v.center.y), v.center.y));
        }
        for v in near.iter().filter(|v| v.kind == VolcanoKind::Super) {
            for f in v.features.iter().filter(|f| f.kind == FeatureKind::Geyser) {
                self.forced_geysers.push((IVec3::new(f.at.x.round() as i32, f.ground.round() as i32, f.at.y.round() as i32), ticks));
            }
        }
        self.refreshed = None;
        at
    }

    /// One tick: follow the volcanoes near `player`, run their eruptions and geysers. Returns
    /// what happened, and the report for the client (`None` when there's nothing new to say).
    pub fn tick(&mut self, world: &World, fluids: &mut FluidSim, player: Vec3, ticks: u64) -> (Events, Option<Report>) {
        let p = Vec2::new(player.x, player.z);
        if self.refreshed.is_none_or(|(at, when)| at.distance(p) > 500.0 || ticks - when > 1200) {
            let list = world.worldgen.volcanoes_near(p, FOLLOW);
            let mut tracked = Vec::new();
            for v in list.into_iter().filter(|v| v.kind != VolcanoKind::Dormant) {
                let key = [v.center.x.round() as i32, v.center.y.round() as i32];
                match self.tracked.iter().position(|t| t.key == key) {
                    Some(i) => tracked.push(self.tracked.swap_remove(i)),
                    None => tracked.push(self.track(world, v)),
                }
            }
            self.tracked = tracked;
            self.refreshed = Some((p, ticks));
        }
        let mut events = Events::default();
        let mut report = Report::default();
        let mut throws = Vec::new();
        let clocks: Vec<Option<(u64, u64, Kind)>> = self.tracked.iter().map(|t| self.clock(t, ticks)).collect();
        for (i, clock) in clocks.into_iter().enumerate() {
            if let Some((into, cycle, kind)) = clock
                && (RUMBLE_TICKS..RUMBLE_TICKS + ERUPT_TICKS).contains(&into)
                && Vec2::new(self.tracked[i].summit.x, self.tracked[i].summit.z).distance(p) < FOLLOW
            {
                self.begin(i, into, cycle, kind, ticks, world, &mut events);
            }
        }
        self.run_flows_and_blasts(world, player, ticks, &mut events, &mut report);
        let phases: Vec<(Phase, f32)> = self.tracked.iter().map(|t| self.phase(t, ticks)).collect();
        for (i, (phase, strength)) in phases.into_iter().enumerate() {
            let t = &mut self.tracked[i];
            let near = Vec2::new(t.summit.x, t.summit.z).distance(p) < NEAR;
            if phase != t.last_phase && near {
                match phase {
                    Phase::Rumbling => events.notices.push("The ground trembles: a volcano is waking".into()),
                    Phase::Erupting => events.notices.push("The volcano erupts!".into()),
                    Phase::Waning => events.notices.push("The eruption is dying down".into()),
                    Phase::Quiet => {}
                }
            }
            t.last_phase = phase;
            if phase != Phase::Quiet {
                report.active.push(Activity { summit: t.summit, radius: t.volcano.radius, phase, strength });
            }
            let pour = Vec2::new(t.summit.x, t.summit.z).distance(p) < POUR_REACH;
            if phase == Phase::Erupting && pour {
                for &vent in &t.vents {
                    fluids.pour(world, vent, FluidKind::Lava, (WELL_FLOW as f32 * strength.max(0.2)) as u16);
                }
                if ticks.is_multiple_of(BOMB_EVERY) {
                    throws.push((t.summit, t.volcano.crater_radius, strength));
                }
            }
            let geysers = t.geysers.clone();
            for g in geysers {
                let Some(strength) = self.geyser(g, ticks) else { continue };
                if Vec2::new(g.0.x as f32, g.0.z as f32).distance(p) > GEYSER_REACH {
                    continue;
                }
                // A column of water thrown up over the vent (it falls back as the simulation runs).
                let top = (4.0 + 8.0 * strength) as i32;
                for dy in 3..top {
                    fluids.pour(world, g.0 + IVec3::Y * dy, FluidKind::Water, FULL / 3);
                }
                report.geysers.push((g.0.as_vec3(), strength));
                let rel = player - g.0.as_vec3();
                if Vec2::new(rel.x, rel.z).length() < 2.5 && (-1.0..14.0).contains(&rel.y) && ticks.is_multiple_of(5) {
                    events.hurt.push((25, crate::vitals::Cause::Heat));
                }
            }
        }
        for (summit, crater, strength) in throws {
            let a = self.random() * std::f32::consts::TAU;
            let out = 12.0 + 30.0 * self.random() * strength;
            let up = 28.0 + 22.0 * self.random();
            let start = summit + Vec3::new(a.cos(), 0.0, a.sin()) * crater * 0.3 + Vec3::Y * 8.0;
            // Mostly small: the chance of each size up halves.
            let size = (self.random().powf(2.2) * BOMB_SIZES as f32).min(BOMB_SIZES as f32 - 1.0) as u8;
            self.bombs.push(Bomb { at: start, vel: Vec3::new(a.cos() * out, up, a.sin() * out), size });
        }
        // Bombs fly, and burst where they land.
        let mut landed = Vec::new();
        for b in &mut self.bombs {
            b.vel.y -= GRAVITY * DT;
            b.at += b.vel * DT;
            if b.vel.y < 0.0 && b.at.y <= world.worldgen.height(b.at.x, b.at.z) {
                landed.push((b.at, b.size));
            }
        }
        self.bombs.retain(|b| !(b.vel.y < 0.0 && b.at.y <= world.worldgen.height(b.at.x, b.at.z)) && b.at.y > -500.0);
        for (at, size) in landed {
            events.landed.push(at);
            let d = at.distance(player);
            let scale = 0.4 + 0.3 * size as f32;
            let reach = BOMB_REACH * scale;
            if d < reach {
                events.hurt.push(((BOMB_DAMAGE as f32 * scale * (1.0 - d / reach * 0.7)) as i32, crate::vitals::Cause::Eruption));
            }
        }
        report.bombs = self.bombs.iter().map(|b| (b.at, b.size)).collect();
        // Forced eruptions and geysers that are over are forgotten.
        let span = RUMBLE_TICKS + ERUPT_TICKS + WANE_TICKS;
        self.forced.retain(|(_, s, _)| ticks < s + span);
        self.forced_geysers.retain(|(_, s)| ticks < s + GEYSER_BLOW);
        let busy = !(report.active.is_empty()
            && report.bombs.is_empty()
            && report.geysers.is_empty()
            && report.flows.is_empty()
            && report.blasts.is_empty());
        let send = busy || self.reported;
        self.reported = busy;
        (events, send.then_some(report))
    }
}

fn t_key(v: &Volcano) -> [i32; 2] {
    [v.center.x.round() as i32, v.center.y.round() as i32]
}

impl Volcanism {
    /// Flows race down and slow; the shockwave races out. Either kills a player it catches.
    fn run_flows_and_blasts(&mut self, world: &World, player: Vec3, ticks: u64, events: &mut Events, report: &mut Report) {
        let p = Vec2::new(player.x, player.z);
        let ground = world.worldgen.height(player.x, player.z);
        for (start, flow) in &mut self.flows {
            let s = (ticks - *start) as f32 / 20.0;
            // Fast at first, slowing as it spreads out (it never quite reaches `length`).
            flow.front = flow.length * (1.0 - (-FLOW_SPEED * s / flow.length).exp());
            let rel = p - Vec2::new(flow.origin.x, flow.origin.z);
            let along = rel.dot(flow.dir);
            let in_cloud = along <= flow.front
                && along > flow.front - FLOW_FRONT
                && rel.perp_dot(flow.dir).abs() < flow_half_width(along, flow.length, flow.width)
                && player.y < ground + FLOW_HEIGHT;
            if in_cloud {
                events.hurt.push((LETHAL, crate::vitals::Cause::Eruption));
            }
            report.flows.push(*flow);
        }
        // A flow's done once it's all but stopped and its cloud has had time to settle.
        self.flows.retain(|(start, f)| (ticks - start) as f32 / 20.0 < f.length / FLOW_SPEED * 3.0 + 30.0);
        for (start, blast) in &mut self.blasts {
            let before = blast.radius;
            blast.radius = SHOCK_SPEED * (ticks - *start) as f32 / 20.0;
            let rel = p - Vec2::new(blast.center.x, blast.center.z);
            let d = rel.length();
            if d > before && d <= blast.radius {
                // The shockwave passes the player.
                let reach = blast_reach(rel, blast.dir);
                if d < reach {
                    events.hurt.push((LETHAL, crate::vitals::Cause::Eruption));
                } else if d < reach * 1.5 {
                    events.hurt.push((400, crate::vitals::Cause::Eruption));
                }
            }
            report.blasts.push(*blast);
        }
        self.blasts.retain(|(_, b)| b.radius < BLAST_REACH * 1.6);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_eruption_goes_through_its_phases() {
        assert_eq!(phase_at(0).0, Phase::Rumbling);
        assert_eq!(phase_at(RUMBLE_TICKS + 10).0, Phase::Erupting);
        assert_eq!(phase_at(RUMBLE_TICKS + ERUPT_TICKS + 10).0, Phase::Waning);
        assert_eq!(phase_at(RUMBLE_TICKS + ERUPT_TICKS + WANE_TICKS + 1).0, Phase::Quiet);
        let (_, mid) = phase_at(RUMBLE_TICKS + ERUPT_TICKS / 2);
        assert!(mid > 0.9, "full strength mid-eruption: {mid}");
    }

    /// The world, an active cone, and a forced eruption of `kind` run (from its start to `ticks`)
    /// with the player at `player`: what killed or hurt them, the edits and scars, and the flows.
    fn run(kind: Kind, player: impl Fn(&Volcano) -> Vec3, ticks: u64) -> (Vec<i32>, Events, Vec<Flow>) {
        let world = World::new(1256132528);
        let v = world.worldgen.volcanoes_near(Vec2::ZERO, 16_000.0).into_iter().find(|v| v.kind == VolcanoKind::Active).unwrap();
        let at = player(&v);
        let mut volcanism = Volcanism::new(world.worldgen.seed());
        let mut fluids = FluidSim::new();
        volcanism.force(&world, at, 0, kind);
        let (mut hurt, mut all, mut flows) = (Vec::new(), Events::default(), Vec::new());
        for t in 0..ticks {
            let (events, report) = volcanism.tick(&world, &mut fluids, at, t);
            hurt.extend(events.hurt.iter().map(|(a, _)| *a));
            all.edits.extend(events.edits);
            all.scars.extend(events.scars);
            if let Some(r) = report
                && flows.is_empty()
            {
                flows = r.flows.clone();
            }
        }
        (hurt, all, flows)
    }

    #[test]
    fn a_lateral_blast_cuts_the_flank_away_and_its_shockwave_kills_in_front() {
        let front = |d: f32| move |v: &Volcano| {
            let p = v.center + v.tilt * d;
            Vec3::new(p.x, 200.0, p.y)
        };
        // 1.5 km out in front: dead when the shockwave (at 340 m/s) gets there.
        let (hurt, events, _) = run(Kind::Blast, front(1500.0), RUMBLE_TICKS + 200);
        assert!(hurt.contains(&LETHAL), "survived the blast 1.5 km in front: {hurt:?}");
        assert!(events.edits.len() >= 3, "the flank didn't collapse");
        assert!(events.scars.iter().any(|s| matches!(s, Scar::Blast { .. })));
        // 5 km out: it passes.
        let (hurt, _, _) = run(Kind::Blast, front(5000.0), RUMBLE_TICKS + 400);
        assert!(!hurt.contains(&LETHAL), "killed 5 km away");
    }

    #[test]
    fn a_pyroclastic_flow_kills_whats_in_its_path_and_spares_the_side() {
        // Where the first flow goes (watched from high over the summit, out of its reach).
        let far = |v: &Volcano| Vec3::new(v.center.x, 5000.0, v.center.y);
        let (_, events, _) = run(Kind::Flows, far, RUMBLE_TICKS + 300);
        let Some(Scar::Flow { origin, dir, length, .. }) = events.scars.first().copied() else { panic!("no flow") };
        let world = World::new(1256132528);
        let on = origin + dir * length * 0.4;
        let ground = |p: Vec2| Vec3::new(p.x, world.worldgen.height(p.x, p.y) + 1.0, p.y);
        let (hurt, _, _) = run(Kind::Flows, move |_| ground(on), RUMBLE_TICKS + 1600);
        assert!(hurt.contains(&LETHAL), "survived standing in the flow's path");
        let side = origin + dir.perp() * length * 0.4;
        let (hurt, _, _) = run(Kind::Flows, move |_| ground(side), RUMBLE_TICKS + 1600);
        assert!(!hurt.contains(&LETHAL) || events.scars.len() > 1, "killed off to the side of a single flow");
    }

    #[test]
    fn a_forced_eruption_pours_lava_and_throws_bombs() {
        let world = World::new(1256132528);
        let wg = &world.worldgen;
        let v = wg.volcanoes_near(Vec2::ZERO, 16_000.0).into_iter().find(|v| v.kind == VolcanoKind::Active).expect("an active cone");
        // Stand on the flank, a little way down.
        let (x, z) = (v.center.x + v.radius * 0.4, v.center.y);
        let player = Vec3::new(x, wg.height(x, z) + 1.0, z);
        let mut volcanism = Volcanism::new(wg.seed());
        let mut fluids = FluidSim::new();
        volcanism.force(&world, player, 0, Kind::Natural);
        let mut notices = Vec::new();
        let mut bombs_seen = 0;
        for ticks in 0..(RUMBLE_TICKS + 400) {
            let (events, report) = volcanism.tick(&world, &mut fluids, player, ticks);
            notices.extend(events.notices);
            bombs_seen = bombs_seen.max(report.map_or(0, |r| r.bombs.len()));
            fluids.tick(&world);
        }
        assert_eq!(notices.first().map(String::as_str), Some("The ground trembles: a volcano is waking"), "{notices:?}");
        assert!(notices.iter().any(|n| n == "The volcano erupts!"), "{notices:?}");
        assert!(bombs_seen > 3, "{bombs_seen} bombs in the air at once");
        let lava: u64 = fluids
            .fluids
            .chunks
            .values()
            .flat_map(|ch| (0..asc_world::fluid::CELLS).filter(move |&i| ch.kind[i] == FluidKind::Lava as u8 && ch.source[i] == 0).map(move |i| ch.amount[i] as u64))
            .sum();
        assert!(lava > 50 * FULL as u64, "only {lava} of lava poured out");
    }
}
