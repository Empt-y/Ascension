//! Volcanoes over time: active cones erupt now and then, and calderas' geysers blow.
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
}

pub struct Volcanism {
    tracked: Vec<Tracked>,
    /// Where (and when) the tracked list was last looked up.
    refreshed: Option<(Vec2, u64)>,
    /// Eruptions brought forward (cheats): volcano key and the tick it started rumbling.
    forced: Vec<([i32; 2], u64)>,
    /// Geysers brought forward: vent and tick.
    forced_geysers: Vec<(IVec3, u64)>,
    bombs: Vec<Bomb>,
    rng: u64,
    seed: i32,
    /// Whether the last report had anything in it (so an empty one is sent once, to clear).
    reported: bool,
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
            bombs: Vec::new(),
            rng: seed as u32 as u64 ^ 0x6572_7570,
            seed,
            reported: false,
        }
    }

    fn random(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(0x1405_7B7E_F767_814F);
        unit(self.rng)
    }

    /// The phase of tracked volcano `t` at `ticks`.
    fn phase(&self, t: &Tracked, ticks: u64) -> (Phase, f32) {
        if t.volcano.kind != VolcanoKind::Active {
            return (Phase::Quiet, 0.0);
        }
        if let Some((_, start)) = self.forced.iter().find(|(k, _)| *k == t.key)
            && ticks >= *start
            && ticks - start < RUMBLE_TICKS + ERUPT_TICKS + WANE_TICKS
        {
            return phase_at(ticks - start);
        }
        let h = hash(self.seed, t.key, 0x6379_636c);
        let period = ERUPTION_CYCLE[0] + ((ERUPTION_CYCLE[1] - ERUPTION_CYCLE[0]) as f32 * unit(h)) as u64;
        let offset = (period as f32 * unit(h.rotate_left(17))) as u64;
        phase_at((ticks + offset) % period)
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
        let summit = Vec3::new(c.x, rim_height(0.0, v.crater_radius), c.y);
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

    /// Bring the nearest active volcano's eruption, and the nearest geysers, forward to now.
    pub fn force(&mut self, world: &World, player: Vec3, ticks: u64) -> Option<Vec3> {
        let near = world.worldgen.volcanoes_near(Vec2::new(player.x, player.z), 20_000.0);
        let p = Vec2::new(player.x, player.z);
        let active = near.iter().filter(|v| v.kind == VolcanoKind::Active).min_by(|a, b| a.center.distance(p).total_cmp(&b.center.distance(p)));
        let mut at = None;
        if let Some(v) = active {
            let key = [v.center.x.round() as i32, v.center.y.round() as i32];
            self.forced.retain(|(k, _)| *k != key);
            self.forced.push((key, ticks));
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
        self.forced.retain(|(_, s)| ticks < s + span);
        self.forced_geysers.retain(|(_, s)| ticks < s + GEYSER_BLOW);
        let busy = !(report.active.is_empty() && report.bombs.is_empty() && report.geysers.is_empty());
        let send = busy || self.reported;
        self.reported = busy;
        (events, send.then_some(report))
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
        volcanism.force(&world, player, 0);
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
