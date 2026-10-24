//! Herds, packs and schools acting together: the leader leads (where to stroll, where the water
//! is), alarm spreads, wariness of the player is shared, a pack picks its quarry and feeds
//! together, and a herd moves its range away from a volcano that stirs.

use std::collections::BTreeMap;

use glam::Vec2;

use super::brain::{Agent, Seen, Who, World};
use super::{Creature, CreatureId, HerdId, TICKS_PER_SECOND, hash, kinds, unit};
use crate::nav::{NavGrid, cell_at};

/// Herds are looked after this often (ticks).
pub const EVERY: u64 = 10;
/// The leader strolls somewhere new every this many seconds or so.
const STROLL_SECONDS: [f32; 2] = [20.0, 45.0];
/// A herd moves this far from a volcano that stirs within `VOLCANO_REACH` of it.
const VOLCANO_REACH: f32 = 1500.0;
const VOLCANO_MOVE: f32 = 400.0;
/// Water is looked for this far round (metres of walking).
const WATER_SEARCH: f32 = 140.0;
/// A pack looks for quarry this far off, gives up on a chase after this long (seconds) or this
/// far, and takes on the player only this near.
const HUNT_REACH: f32 = 90.0;
const CHASE_SECONDS: f32 = 25.0;
const CHASE_LOST: f32 = 120.0;
const PLAYER_REACH: f32 = 70.0;
/// An alarm is answered for this long (ticks).
const ALARM_TICKS: u64 = 100;

/// A pack's quarry, and since when.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Target {
    pub who: Who,
    pub since: u64,
}

pub struct Herd {
    pub id: HerdId,
    pub kind: u16,
    /// Where it lives (its range's middle), and where it first did (before any volcano moved it).
    pub home: Vec2,
    origin: Vec2,
    pub range: f32,
    pub members: Vec<CreatureId>,
    pub leader: Option<CreatureId>,
    /// The members' middle.
    pub middle: Vec2,
    /// Where the leader is strolling to, and until when.
    pub goal: Option<Vec2>,
    goal_until: u64,
    /// When the herd last took fright, and at what (where).
    pub alarm: Option<(u64, Vec2)>,
    /// Wariness of the player, shared (0..1).
    pub wary: f32,
    /// Where it drinks.
    pub water: Option<Vec2>,
    water_looked: bool,
    /// Packs: what they're hunting, and the kill they're feeding on.
    pub target: Option<Target>,
    pub feed: Option<CreatureId>,
}

impl Herd {
    pub fn new(id: HerdId, kind: u16, members: &[Creature]) -> Self {
        let middle = members.iter().map(|c| Vec2::new(c.at.x, c.at.z)).sum::<Vec2>() / members.len().max(1) as f32;
        let k = &kinds()[kind as usize];
        Self {
            id,
            kind,
            home: middle,
            origin: middle,
            range: k.spread * 3.0 + 30.0,
            members: members.iter().map(|c| c.id).collect(),
            leader: members.first().map(|c| c.id),
            middle,
            goal: None,
            goal_until: 0,
            alarm: None,
            wary: 0.0,
            water: None,
            water_looked: false,
            target: None,
            feed: None,
        }
    }
}

pub fn update(h: &mut Herd, agents: &BTreeMap<CreatureId, Agent>, seen: &Seen, nav: &NavGrid, w: &World) {
    let k = &kinds()[h.kind as usize];
    let ticks = w.ticks;
    let dt = EVERY as f32 / TICKS_PER_SECOND;
    let alive: Vec<&Agent> = h.members.iter().filter_map(|m| agents.get(m)).filter(|a| !a.c.dead()).collect();
    if alive.is_empty() {
        h.leader = None;
        return;
    }
    // The first still standing leads.
    h.leader = alive.first().map(|a| a.c.id);
    h.middle = alive.iter().map(|a| a.xz()).sum::<Vec2>() / alive.len() as f32;
    h.wary *= (-dt / 300.0).exp();
    h.wary = h.wary.max(alive.iter().map(|a| a.wary).fold(0.0, f32::max) * 0.9);
    if h.alarm.is_some_and(|(t, _)| ticks > t + ALARM_TICKS) {
        h.alarm = None;
    }
    // A volcano stirring near: the herd moves off, and back when it's quiet again.
    let threat = w.hazards.volcanoes.iter().map(|(s, _)| Vec2::new(s.x, s.z)).find(|s| s.distance(h.origin) < VOLCANO_REACH);
    h.home = match threat {
        Some(s) => h.origin + (h.origin - s).normalize_or(Vec2::X) * VOLCANO_MOVE,
        None => h.origin,
    };
    let leader = alive[0];
    let profile = k.profile();
    // Somewhere new to stroll to, within the range (on ground it can stand on).
    if h.goal.is_none() || ticks >= h.goal_until {
        let r = hash(w.seed, h.id.0 as i64, ticks as i64, 0x7374);
        let mut spot = None;
        for i in 0..8u32 {
            let a = unit(r.rotate_left(i * 7)) * std::f32::consts::TAU;
            let d = h.range * 0.75 * unit(r.rotate_left(i * 7 + 3)).sqrt();
            let p = h.home + Vec2::from_angle(a) * d;
            if nav.cell(cell_at(p)).is_some_and(|c| profile.fits(c)) {
                spot = Some(p);
                break;
            }
        }
        h.goal = spot.or(h.goal);
        let secs = STROLL_SECONDS[0] + (STROLL_SECONDS[1] - STROLL_SECONDS[0]) * unit(r.rotate_left(50));
        h.goal_until = ticks + (secs * TICKS_PER_SECOND) as u64;
    }
    // Water: found once the herd gets thirsty (and again if what it had has gone).
    if !k.swims() {
        let thirsty = alive.iter().any(|a| a.thirst > 0.45);
        let still_wet = h.water.is_some_and(|p| {
            crate::nav::NEIGHBOURS.iter().any(|(d, _)| nav.cell(cell_at(p) + *d).is_some_and(|c| c.depth() > 0.1))
                || nav.cell(cell_at(p)).is_some_and(|c| c.depth() > 0.1)
        });
        if h.water.is_some() && !still_wet && nav.is_built(h.water.unwrap()) {
            h.water = None;
            h.water_looked = false;
        }
        if thirsty && h.water.is_none() && !h.water_looked && nav.is_built(leader.xz()) {
            h.water_looked = true;
            h.water = nav
                .nearest(leader.xz(), &profile, WATER_SEARCH, |_, c| c.depth() > 0.15 && c.depth() < 3.0)
                .map(crate::nav::centre);
        }
        if !thirsty {
            h.water_looked = false;
        }
    }
    if k.predator() {
        hunt(h, &alive, seen, w);
    }
}

/// A pack's hunting: keep after its quarry while it's worth it, or pick some.
fn hunt(h: &mut Herd, alive: &[&Agent], seen: &Seen, w: &World) {
    let k = &kinds()[h.kind as usize];
    let ticks = w.ticks;
    let leader = alive[0];
    let hunger = alive.iter().map(|a| a.hunger).sum::<f32>() / alive.len() as f32;
    let angry = alive.iter().any(|a| a.anger > 0.3 && a.foe == Some(Who::Player));
    let hurt = alive.iter().filter(|a| (a.c.health as f32) < k.health as f32 * 0.4).count();
    // A kill to feed on: the pack's own (lying near), while it's hungry.
    if let Some(c) = h.feed
        && seen.get(c).is_none_or(|o| o.alive || hunger < 0.05)
    {
        h.feed = None;
    }
    if h.feed.is_none() && hunger > 0.15 {
        h.feed = seen
            .near(leader.xz(), 40.0)
            .find(|o| !o.alive && (k.prey.contains(&o.kind) || k.big_prey.contains(&o.kind)))
            .map(|o| o.id);
        if h.feed.is_some() {
            h.target = None;
        }
    }
    if let Some(t) = h.target {
        let at = super::brain::whereabouts(t.who, seen, w.player);
        let gone = at.is_none_or(|p| p.distance(leader.xz()) > CHASE_LOST);
        let long = ticks > t.since + ((CHASE_SECONDS + 15.0) * TICKS_PER_SECOND) as u64;
        let fed = hunger < 0.2 && !angry;
        if gone || long || fed || hurt >= 2 || h.feed.is_some() {
            h.target = None;
        }
        return;
    }
    if h.feed.is_some() || hurt >= 2 {
        return;
    }
    let p = w.player;
    let near_player = p.alive && !p.in_water && p.xz().distance(leader.xz()) < PLAYER_REACH;
    if angry && near_player {
        h.target = Some(Target { who: Who::Player, since: ticks });
        return;
    }
    if hunger < 0.5 || !w.awake(k.activity) && hunger < 0.85 {
        return;
    }
    // The best of what's about: near, weak and on its own.
    let mut best: Option<(f32, Who)> = None;
    for o in seen.near(leader.xz(), HUNT_REACH) {
        if !o.alive || o.herd == h.id {
            continue;
        }
        let big = k.big_prey.contains(&o.kind);
        if !(k.prey.contains(&o.kind) || big && hunger > 0.85) {
            continue;
        }
        let d = o.xz().distance(leader.xz());
        let alone = seen.near(o.xz(), 12.0).filter(|x| x.herd == o.herd && x.alive).count() as f32;
        let score = (1.0 - d / HUNT_REACH) + (1.0 - o.health) + 0.5 / alone;
        if best.is_none_or(|(b, _)| score > b) {
            best = Some((score, Who::Creature(o.id)));
        }
    }
    // The player, at night, alone in the open, when the pack is hungry enough (and not carrying a
    // torch: they keep off fire).
    if near_player && !p.torch && w.night() && hunger > 0.6 {
        let d = p.xz().distance(leader.xz());
        let score = (1.0 - d / PLAYER_REACH) + (1.0 - p.health) + 0.3;
        if best.is_none_or(|(b, _)| score > b) {
            best = Some((score, Who::Player));
        }
    }
    h.target = best.map(|(_, who)| Target { who, since: ticks });
}
