//! What an animal does: it notices the player, predators, prey and danger; gets hungry, thirsty and
//! sleepy; remembers being chased; and picks the behaviour that matters most just now (each scored
//! from its drives and what it sees, the current one favoured a little so it doesn't flit). The
//! behaviour sets a goal (somewhere to go, how fast, what to face) that `motion` carries out.

use std::collections::BTreeMap;

use glam::{Vec2, Vec3};

use super::group::Herd;
use super::{Act, Activity, Changes, Creature, CreatureId, Diet, HerdId, Hazards, Motion, PlayerView, TICKS_PER_SECOND, hash, unit};
use crate::nav::{FORAGE, NavGrid, cell_at, centre};
use crate::path::{Path, Prefs};

/// Within this of the player animals think every `NEAR_EVERY` ticks; further off, every
/// `FAR_EVERY`.
const NEAR: f32 = 120.0;
const NEAR_EVERY: u64 = 4;
const FAR_EVERY: u64 = 20;
/// An idle clip is kept this long (seconds) before choosing another.
const IDLE_SECONDS: [f32; 2] = [6.0, 15.0];
/// Hunger and thirst rise from nothing to full in these many seconds, and eating and drinking
/// bring them down at these rates (per second).
const HUNGRY_IN: f32 = 600.0;
const THIRSTY_IN: f32 = 420.0;
const GRAZE_RATE: f32 = 1.0 / 45.0;
const FEED_RATE: f32 = 1.0 / 20.0;
const DRINK_RATE: f32 = 1.0 / 8.0;
/// Fear fades in a few seconds, anger in tens, wariness of the player over minutes.
const FEAR_FADE: f32 = 4.0;
const ANGER_FADE: f32 = 20.0;
const WARY_FADE: f32 = 300.0;
/// A herd's alarm spreads this far, and reaches each member after up to this long (seconds).
const ALARM_REACH: f32 = 60.0;
const ALARM_DELAY: f32 = 0.8;
/// How far to run when fleeing (m).
const FLEE_RUN: [f32; 2] = [40.0, 80.0];
/// Predators: stalk until this near, then run in; give a chase up after this long (seconds).
const STALK_UNTIL: f32 = 35.0;

/// Who or what an animal is reacting to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Who {
    Player,
    Creature(CreatureId),
    /// Something its herd ran from (it didn't see what).
    Somewhere,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Gait {
    Walk,
    Trot,
    Run,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Behaviour {
    /// Standing about, an idle clip.
    Idle,
    /// The herd's leader strolling somewhere; or keeping up with the herd.
    Wander,
    Graze,
    Drink,
    Sleep,
    /// Watching something.
    Alert,
    Flee,
    /// Getting away from fire, lava, a flow or a blast.
    Escape,
    /// Turning on what hurt it, or (a guard) on a threat to the herd.
    Fight,
    /// Predators: closing in on prey unseen, then running it down.
    Stalk,
    Chase,
    /// At a carcass.
    Feed,
    /// Strayed too far: back to the herd's range.
    Home,
    /// Swimmers going round with their school.
    School,
    /// Coming to look at the player.
    Curious,
}

/// Where it's going.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Goal {
    pub to: Vec2,
    pub gait: Gait,
    /// Close enough when this near.
    pub within: f32,
    /// What to face on arriving (or while standing).
    pub face: Option<Vec2>,
}

/// A path wanted: for whom, to where, how.
#[derive(Clone, Copy, Debug)]
pub struct Ask {
    pub id: CreatureId,
    pub to: Vec2,
    pub prefs: Prefs,
}

/// What thinking asks of the rest of the world.
#[derive(Clone, Copy, Debug)]
pub enum Event {
    Path(Ask),
    /// Its herd should run: from where.
    Alarm(HerdId, Vec2),
    /// It fed on this carcass.
    Ate(CreatureId),
}

/// The world as an animal sees it this tick.
pub struct World<'a> {
    pub player: &'a PlayerView,
    pub hazards: &'a Hazards,
    pub ticks: u64,
    pub time_of_day: f32,
    pub seed: i32,
}

impl World<'_> {
    /// 0 at night, 1 in daylight.
    pub fn daylight(&self) -> f32 {
        let sun = ((self.time_of_day - 0.25) * std::f32::consts::TAU).sin();
        (sun * 3.0 + 0.5).clamp(0.0, 1.0)
    }

    pub fn night(&self) -> bool {
        self.daylight() < 0.2
    }

    /// Whether `activity` is the time to be up and about.
    pub fn awake(&self, activity: Activity) -> bool {
        let t = self.time_of_day;
        match activity {
            Activity::Day => (0.24..0.82).contains(&t),
            Activity::Dusk => !(0.38..0.66).contains(&t),
            Activity::Night => !(0.30..0.72).contains(&t),
        }
    }
}

/// Every creature as the others see it this tick (sorted by id).
pub struct Seen {
    pub list: Vec<Other>,
}

#[derive(Clone, Copy, Debug)]
pub struct Other {
    pub id: CreatureId,
    pub kind: u16,
    pub herd: HerdId,
    pub at: Vec3,
    pub alive: bool,
    /// 0..1 of its health.
    pub health: f32,
    pub behaviour: Behaviour,
}

impl Other {
    pub fn xz(&self) -> Vec2 {
        Vec2::new(self.at.x, self.at.z)
    }
}

impl Seen {
    pub fn of(agents: &BTreeMap<CreatureId, Agent>) -> Self {
        Self {
            list: agents
                .values()
                .map(|a| Other {
                    id: a.c.id,
                    kind: a.c.kind,
                    herd: a.herd,
                    at: a.c.at,
                    alive: !a.c.dead(),
                    health: a.c.health as f32 / a.c.kind().health as f32,
                    behaviour: a.behaviour,
                })
                .collect(),
        }
    }

    pub fn get(&self, id: CreatureId) -> Option<&Other> {
        self.list.binary_search_by_key(&id, |o| o.id).ok().map(|i| &self.list[i])
    }

    pub fn near(&self, p: Vec2, r: f32) -> impl Iterator<Item = &Other> {
        self.list.iter().filter(move |o| o.xz().distance_squared(p) < r * r)
    }
}

pub struct Agent {
    pub c: Creature,
    pub herd: HerdId,
    /// Its place in the herd (for keeping a spot round the leader, and timing).
    pub index: usize,
    seed: i32,
    /// Horizontal velocity (m/s), for the others to lead it when chasing.
    pub velocity: Vec2,
    /// Swimmers: how far down the water it keeps (0 just under the surface, 1 just off the bed).
    pub swim_depth: f32,
    pub hunger: f32,
    pub thirst: f32,
    pub fear: f32,
    pub anger: f32,
    /// Wariness of the player (0..1): it runs from further off, and settles less.
    pub wary: f32,
    /// What scares it most, and where that was.
    pub threat: Option<(Who, Vec2)>,
    /// What hurt it (or it's after).
    pub foe: Option<Who>,
    /// 0..1 of its running.
    pub stamina: f32,
    pub behaviour: Behaviour,
    pub started: u64,
    pub goal: Option<Goal>,
    pub gait: Gait,
    /// The path it's following, and the point it's heading for.
    pub path: Vec<Vec3>,
    pub waypoint: usize,
    /// The goal the path (or its search) is for, and when it was asked.
    pub asked: Option<(Vec2, u64)>,
    /// Attacking: whom, the tick the blow lands, and the tick it's over.
    pub attack: Option<(Who, u64, u64)>,
    /// Not before this tick can it attack again.
    pub ready_at: u64,
    pub next_think: u64,
    last_think: u64,
    /// When the current one-shot act (hit, attack, jump) is over.
    pub act_until: Option<u64>,
    /// The herd alarm last answered.
    alarm_seen: u64,
    /// A carcass something has fed on.
    pub eaten: bool,
    /// What was last sent of where it is.
    sent: Option<(Vec3, f32, f32)>,
    /// Ticks it's been trying to go somewhere and getting nowhere.
    pub stuck: u32,
    /// How its last path was asked for (to ask again the same way).
    prefs: Prefs,
}

impl Agent {
    pub fn new(c: Creature, herd: HerdId, index: usize, seed: i32, ticks: u64) -> Self {
        let h = hash(seed, c.id.0 as i64, 0, 0x6167_656e);
        Self {
            c,
            herd,
            index,
            seed,
            velocity: Vec2::ZERO,
            swim_depth: unit(h),
            hunger: unit(h.rotate_left(11)) * 0.6,
            thirst: unit(h.rotate_left(23)) * 0.6,
            fear: 0.0,
            anger: 0.0,
            wary: 0.0,
            threat: None,
            foe: None,
            stamina: 1.0,
            behaviour: Behaviour::Idle,
            started: ticks,
            goal: None,
            gait: Gait::Walk,
            path: Vec::new(),
            waypoint: 0,
            asked: None,
            attack: None,
            ready_at: 0,
            next_think: ticks + h % NEAR_EVERY,
            last_think: ticks,
            act_until: None,
            alarm_seen: 0,
            eaten: false,
            sent: Some((c.at, c.yaw, 0.0)),
            stuck: 0,
            prefs: Prefs::default(),
        }
    }

    fn rand(&self, ticks: u64, salt: u64) -> f32 {
        unit(hash(self.seed, self.c.id.0 as i64, ticks as i64, salt))
    }

    pub fn xz(&self) -> Vec2 {
        Vec2::new(self.c.at.x, self.c.at.z)
    }

    /// Start a path (found for its goal).
    pub fn follow(&mut self, path: Path) {
        self.path = path.points;
        self.waypoint = 1.min(self.path.len().saturating_sub(1));
    }

    /// Its goal can't be reached from here: give it up and think again.
    pub fn no_way(&mut self) {
        self.goal = None;
        self.path.clear();
        self.asked = None;
        self.next_think = self.next_think.min(self.last_think + 2);
    }

    /// Set (or keep) a goal; ask for a path when it's new or has moved.
    fn go(&mut self, goal: Goal, prefs: Prefs, ticks: u64, events: &mut Vec<Event>) {
        let moved = self.asked.is_none_or(|(to, at)| to.distance(goal.to) > (goal.within * 0.5).max(1.5) || ticks > at + 200);
        self.goal = Some(goal);
        self.gait = goal.gait;
        self.prefs = prefs;
        if moved && self.xz().distance(goal.to) > goal.within {
            self.asked = Some((goal.to, ticks));
            self.path.clear();
            events.push(Event::Path(Ask { id: self.c.id, to: goal.to, prefs }));
        }
    }

    fn stop(&mut self) {
        self.goal = None;
        self.path.clear();
        self.asked = None;
    }

    /// Play `act` from now (if it isn't already).
    pub fn set_act(&mut self, act: Act, ticks: u64, changes: &mut Changes) {
        if self.c.act != act {
            self.c.act = act;
            self.c.since = ticks;
            changes.acts.push((self.c.id, act, ticks));
        }
    }

    /// Where it is, if that's changed enough since last sent.
    pub fn motion_to_send(&mut self, ticks: u64) -> Option<Motion> {
        let now = (self.c.at, self.c.yaw, self.c.speed);
        let changed = self.sent.is_none_or(|(at, yaw, speed)| {
            at.distance(now.0) > 0.05 || (yaw - now.1).abs() > 0.03 || (speed - now.2).abs() > 0.05
        });
        changed.then(|| {
            self.sent = Some(now);
            Motion { id: self.c.id, at: now.0, yaw: now.1, speed: now.2, tick: ticks }
        })
    }

    fn dist_to(&self, p: Vec2) -> f32 {
        self.xz().distance(p)
    }
}

/// Where `who` is now.
pub fn whereabouts(who: Who, seen: &Seen, player: &PlayerView) -> Option<Vec2> {
    match who {
        Who::Player => player.alive.then(|| player.xz()),
        Who::Creature(id) => seen.get(id).filter(|o| o.alive).map(|o| o.xz()),
        Who::Somewhere => None,
    }
}

/// Think: update drives, look about, and choose what to do.
pub fn think(
    a: &mut Agent,
    herd: Option<&Herd>,
    seen: &Seen,
    nav: &mut NavGrid,
    w: &World,
    events: &mut Vec<Event>,
    changes: &mut Changes,
) {
    let ticks = w.ticks;
    let dt = (ticks - a.last_think) as f32 / TICKS_PER_SECOND;
    a.last_think = ticks;
    let near = a.dist_to(w.player.xz()) < NEAR;
    a.next_think = ticks + if near { NEAR_EVERY } else { FAR_EVERY };
    if a.c.dead() {
        a.next_think = ticks + 100;
        return;
    }
    // Drives.
    a.hunger = (a.hunger + dt / HUNGRY_IN).min(1.0);
    a.thirst = (a.thirst + dt / THIRSTY_IN).min(1.0);
    a.fear *= (-dt / FEAR_FADE).exp();
    a.anger *= (-dt / ANGER_FADE).exp();
    a.wary *= (-dt / WARY_FADE).exp();
    if let Some(h) = herd {
        a.wary = a.wary.max(h.wary * 0.8);
    }
    if a.anger < 0.1 {
        a.foe = None;
    }
    if a.fear < 0.1 && a.behaviour != Behaviour::Flee {
        a.threat = None;
    }
    // A one-shot act runs its course.
    if a.act_until.is_some_and(|t| ticks >= t) {
        a.act_until = None;
        let idle = Act::Idle(0);
        a.set_act(idle, ticks, changes);
    }
    nav.want(a.xz(), 24.0);
    perceive(a, herd, seen, w);
    let choice = choose(a, herd, seen, w);
    if choice != a.behaviour {
        a.behaviour = choice;
        a.started = ticks;
        if choice == Behaviour::Flee
            && let Some((Who::Player | Who::Creature(_), from)) = a.threat
        {
            events.push(Event::Alarm(a.herd, from));
        }
    }
    act(a, herd, seen, nav, w, dt, events, changes);
    // Its path was lost (something in the way): find the way again.
    if let Some(g) = a.goal
        && a.path.is_empty()
        && a.asked.is_none()
        && a.dist_to(g.to) > g.within
    {
        a.asked = Some((g.to, ticks));
        events.push(Event::Path(Ask { id: a.c.id, to: g.to, prefs: a.prefs }));
    }
}

/// Look and listen: the player, predators, the herd's alarm, and (horses, donkeys) kick at
/// anything right behind.
fn perceive(a: &mut Agent, herd: Option<&Herd>, seen: &Seen, w: &World) {
    let k = a.c.kind();
    let me = a.xz();
    let ticks = w.ticks;
    let night_eyes = matches!(k.activity, Activity::Night | Activity::Dusk);
    let light = if night_eyes { 0.9 } else { 0.35 + 0.65 * w.daylight() };
    let p = w.player;
    // The player.
    if p.alive && (!k.swims() || p.in_water) {
        let d = me.distance(p.xz());
        let speed = Vec2::new(p.velocity.x, p.velocity.z).length();
        let sight = k.sight * light;
        let heard = p.noise * k.hearing * 40.0 > d;
        if d < sight || heard {
            // Wary animals, and a player coming on fast, are run from further off.
            let reach = k.flee * (1.0 + a.wary * 1.5) * (1.0 + (speed / 6.0).min(1.0) * 0.5) * if k.swims() { 0.4 } else { 1.0 };
            if d < reach && !k.predator() {
                let f = (1.0 - d / reach) * 0.6 + 0.5;
                if f > a.fear || matches!(a.threat, Some((Who::Player, _))) {
                    a.fear = a.fear.max(f);
                    a.threat = Some((Who::Player, p.xz()));
                }
                // Being run at teaches it to keep away.
                if speed > 3.0 && Vec2::new(p.velocity.x, p.velocity.z).dot(me - p.xz()) > 0.0 {
                    a.wary = (a.wary + 0.04 * k.wariness).min(1.0);
                }
            } else if d < sight && a.threat.is_none() && !k.predator() && a.fear < 0.3 {
                a.threat = Some((Who::Player, p.xz()));
                a.fear = a.fear.max(0.15);
            }
        }
    }
    // Predators (for prey: anything that hunts its kind).
    for o in seen.near(me, k.sight * light) {
        if !o.alive || o.herd == a.herd {
            continue;
        }
        let ok = &super::kinds()[o.kind as usize];
        let hunts = ok.prey.contains(&a.c.kind) || ok.big_prey.contains(&a.c.kind);
        if !hunts {
            continue;
        }
        let d = me.distance(o.xz());
        // A stalker creeps up unseen; one running in is seen a long way off.
        let (reach, alarm) = match o.behaviour {
            Behaviour::Chase => (k.flee * 2.0, 0.6),
            Behaviour::Stalk => (k.flee * 0.6, 0.6),
            _ => (k.flee, 0.4),
        };
        if d < reach {
            let f = (1.0 - d / reach) * 0.6 + alarm;
            if f > a.fear {
                a.fear = f;
                a.threat = Some((Who::Creature(o.id), o.xz()));
            }
        }
    }
    // The herd ran: each follows a moment later.
    if let Some(h) = herd
        && let Some((at, from)) = h.alarm
        && at > a.alarm_seen
    {
        let delay = (a.rand(at, 0x616c) * ALARM_DELAY * TICKS_PER_SECOND) as u64;
        if ticks >= at + delay {
            a.alarm_seen = at;
            if me.distance(from) < ALARM_REACH + k.flee {
                a.fear = a.fear.max(0.8);
                if a.threat.is_none_or(|(who, _)| who == Who::Somewhere) {
                    a.threat = Some((Who::Somewhere, from));
                }
            }
        }
    }
    // Keep the threat's position current.
    if let Some((who, _)) = a.threat
        && let Some(now) = whereabouts(who, seen, w.player)
    {
        a.threat = Some((who, now));
    }
}

/// Score each behaviour that could apply and pick the best.
fn choose(a: &Agent, herd: Option<&Herd>, seen: &Seen, w: &World) -> Behaviour {
    use Behaviour::*;
    let k = a.c.kind();
    let me = a.xz();
    let ticks = w.ticks;
    let mut best = (Idle, 0.3);
    let mut offer = |b: Behaviour, score: f32| {
        // What it's already doing is favoured, for a while at least.
        let held = (ticks - a.started) as f32 / TICKS_PER_SECOND;
        let bonus = if b == a.behaviour { if held < 3.0 { 0.6 } else { 0.25 } } else { 0.0 };
        if score + bonus > best.1 {
            best = (b, score + bonus);
        }
    };
    // Danger first.
    let danger_here = w.hazards.danger(me);
    let danger_near = (0..8).map(|i| w.hazards.danger(me + Vec2::from_angle(i as f32 * 0.785) * 12.0)).fold(0.0f32, f32::max);
    if danger_here > 0.0 || danger_near > 5.0 {
        offer(Escape, 10.0 + danger_here);
    }
    // A foe: fight or flight.
    if let Some(foe) = a.foe
        && a.anger > 0.3
        && k.attack.is_some()
        && let Some(at) = whereabouts(foe, seen, w.player)
        && me.distance(at) < 40.0
    {
        offer(Fight, 3.0 + a.anger * 3.0);
    }
    if a.fear > 0.35
        && let Some((_, from)) = a.threat
    {
        let d = me.distance(from);
        if a.fear > 0.55 || d < k.flee {
            offer(Flee, 2.0 + a.fear * 3.0);
        } else if !k.swims() {
            offer(Alert, 1.6 + a.fear);
        }
    } else if a.fear > 0.1 && a.threat.is_some() && !k.swims() {
        offer(Alert, 1.2 + a.fear);
    }
    // A guard: the herd is wary and the player near it.
    if k.guard
        && herd.is_some_and(|h| h.leader == Some(a.c.id) && h.wary > 0.3)
        && w.player.alive
        && me.distance(w.player.xz()) < 15.0
    {
        // Stands its ground: the herd's guard doesn't run from what it's guarding against.
        offer(Fight, 6.0);
    }
    // Predators: the pack's quarry, or a carcass to feed on.
    if k.predator()
        && let Some(h) = herd
    {
        if let Some(target) = h.target
            && let Some(at) = whereabouts(target.who, seen, w.player)
        {
            // Creep up while it hasn't noticed; once it runs (or the pack is close), run it down.
            let d = me.distance(at);
            let running = match target.who {
                Who::Creature(id) => seen.get(id).is_some_and(|o| o.behaviour == Flee),
                _ => Vec2::new(w.player.velocity.x, w.player.velocity.z).length() > 4.0,
            };
            let pack_running = seen.list.iter().any(|o| o.herd == a.herd && o.behaviour == Chase);
            offer(if d > STALK_UNTIL && !running && !pack_running && target.since + 400 > ticks { Stalk } else { Chase }, 3.5);
        }
        if let Some(c) = h.feed
            && seen.get(c).is_some_and(|o| !o.alive)
            && a.hunger > 0.1
        {
            offer(Feed, 3.0 + a.hunger);
        }
    }
    // Needs.
    let awake = w.awake(k.activity);
    if !k.swims() {
        if a.thirst > 0.55 && herd.is_some_and(|h| h.water.is_some()) {
            offer(Drink, 1.0 + a.thirst * 1.5);
        }
        if a.hunger > 0.35 && k.diet == Diet::Graze {
            offer(Graze, 0.8 + a.hunger * 1.2);
        }
        if !awake {
            offer(Sleep, 1.7);
        }
    }
    // The herd: keep up, and keep to its range.
    if let Some(h) = herd {
        let d_home = me.distance(h.home);
        if d_home > h.range {
            offer(Home, 1.0 + (d_home - h.range) / h.range);
        }
        if k.swims() {
            offer(School, 0.9);
        } else if h.leader == Some(a.c.id) {
            if awake && h.goal.is_some() {
                offer(Wander, 0.7);
            }
        } else if let Some(l) = h.leader.and_then(|l| seen.get(l)) {
            let d = me.distance(l.xz());
            if d > k.spread_keep() {
                offer(Wander, 0.9 + (d - k.spread_keep()) * 0.05);
            }
        }
    }
    if k.curious && w.player.alive && (!k.swims() || w.player.in_water) {
        let d = me.distance(w.player.xz());
        if (6.0..40.0).contains(&d) && a.fear < 0.2 && a.wary < 0.3 && awake {
            offer(Curious, 1.1);
        }
    }
    best.0
}

impl super::Kind {
    /// How far from the leader a herd member drifts before catching up.
    fn spread_keep(&self) -> f32 {
        self.spread * 1.2 + 3.0
    }
}

/// Carry out the behaviour: set the goal, the act and the drives it changes.
#[allow(clippy::too_many_arguments)]
fn act(a: &mut Agent, herd: Option<&Herd>, seen: &Seen, nav: &NavGrid, w: &World, dt: f32, events: &mut Vec<Event>, changes: &mut Changes) {
    use Behaviour::*;
    let k = a.c.kind();
    let me = a.xz();
    let ticks = w.ticks;
    let profile = k.profile();
    let calm = Prefs { cover: false, caution: 1.0, max_nodes: 0 };
    let arrived = |a: &Agent| a.goal.is_none_or(|g| a.dist_to(g.to) <= g.within + 0.3);
    let busy = a.attack.is_some() || a.act_until.is_some();
    match a.behaviour {
        Idle => {
            a.stop();
            idle_clip(a, ticks, changes);
        }
        Wander | Home => {
            let to = match (a.behaviour, herd) {
                (Home, Some(h)) => h.home + (Vec2::from_angle(a.index as f32 * 2.4) * h.range * 0.3),
                (_, Some(h)) if h.leader == Some(a.c.id) => h.goal.unwrap_or(me),
                (_, Some(h)) => {
                    let l = h.leader.and_then(|l| seen.get(l)).map_or(h.home, |o| o.xz());
                    l + Vec2::from_angle(a.index as f32 * 2.4 + 0.7) * k.spread * 0.6
                }
                _ => me,
            };
            let gait = if a.dist_to(to) > 25.0 && a.behaviour == Home { Gait::Trot } else { Gait::Walk };
            a.go(Goal { to, gait, within: 2.0, face: None }, calm, ticks, events);
            if arrived(a) {
                idle_clip(a, ticks, changes);
            } else if !busy {
                a.set_act(Act::Idle(0), ticks, changes);
            }
        }
        Graze => {
            if a.goal.is_none_or(|g| arrived(a) && nav.cell(cell_at(g.to)).is_none_or(|c| !c.has(FORAGE))) {
                let spot = nav
                    .nearest(me, &profile, 14.0, |_, c| c.has(FORAGE) && !c.has(crate::nav::BLOCKED))
                    .map_or(me, |c| centre(c) + Vec2::new(a.rand(ticks, 1) - 0.5, a.rand(ticks, 2) - 0.5));
                a.go(Goal { to: spot, gait: Gait::Walk, within: 1.0, face: None }, calm, ticks, events);
            }
            if arrived(a) {
                a.set_act(Act::Eat, ticks, changes);
                a.hunger = (a.hunger - dt * GRAZE_RATE).max(0.0);
                // Full: move on a few steps now and then.
                if a.hunger < 0.05 {
                    a.behaviour = Idle;
                } else if a.rand(ticks, 3) < 0.02 {
                    a.goal = None;
                }
            } else if !busy {
                a.set_act(Act::Idle(0), ticks, changes);
            }
        }
        Drink => {
            // To the water's edge (the path ends beside the water, where it can stand).
            let water = herd.and_then(|h| h.water).unwrap_or(me);
            let within = 1.6 + a.index as f32 * 0.4;
            a.go(Goal { to: water, gait: Gait::Walk, within, face: Some(water) }, calm, ticks, events);
            if arrived(a) || a.dist_to(water) < within + 0.8 || a.path.last().is_some_and(|p| a.dist_to(Vec2::new(p.x, p.z)) < 0.8) {
                a.stop();
                face(a, water);
                a.set_act(Act::Eat, ticks, changes);
                a.thirst = (a.thirst - dt * DRINK_RATE).max(0.0);
                if a.thirst < 0.05 {
                    a.behaviour = Idle;
                }
            } else if !busy {
                a.set_act(Act::Idle(0), ticks, changes);
            }
        }
        Sleep => {
            let bed = herd.map_or(me, |h| h.home + Vec2::from_angle(a.index as f32 * 2.4) * k.spread * 0.4);
            a.go(Goal { to: bed, gait: Gait::Walk, within: 2.5, face: None }, calm, ticks, events);
            if arrived(a) {
                a.stop();
                a.set_act(Act::Sleep, ticks, changes);
            }
        }
        Alert => {
            a.stop();
            if let Some((_, from)) = a.threat {
                face(a, from);
            }
            if !busy {
                a.set_act(Act::Alert, ticks, changes);
            }
            kick_behind(a, w, changes);
        }
        Flee => {
            let from = a.threat.map_or(me - a.c.forward(), |t| t.1);
            let away = (me - from).normalize_or(a.c.forward());
            let need_new = a.goal.is_none_or(|g| arrived(a) || (g.to - from).length() < me.distance(from) + 10.0);
            if need_new {
                let run = FLEE_RUN[0] + (FLEE_RUN[1] - FLEE_RUN[0]) * a.rand(ticks, 4);
                let to = flee_spot(nav, &profile, me, away, run).unwrap_or(me + away * run);
                a.go(Goal { to, gait: Gait::Run, within: 3.0, face: None }, Prefs { cover: true, caution: 2.0, max_nodes: 3000 }, ticks, events);
            }
            a.gait = Gait::Run;
            if !busy {
                a.set_act(Act::Idle(0), ticks, changes);
            }
            // Cornered (nowhere to run) and it has horns or hooves: it turns.
            if (a.goal.is_none() || a.stuck > 20) && k.attack.is_some() && me.distance(from) < 8.0 {
                a.anger = a.anger.max(0.6);
                a.foe = a.threat.map(|t| t.0).filter(|w| *w != Who::Somewhere);
            }
            kick_behind(a, w, changes);
            // Far enough: it calms down (wary still).
            if me.distance(from) > k.flee * (2.0 + a.wary) {
                a.fear *= 0.5;
            }
        }
        Escape => {
            // Away from the worst danger round about, or across a flow's way.
            let mut best = (me, f32::MAX);
            for i in 0..16 {
                let dir = Vec2::from_angle(i as f32 * std::f32::consts::TAU / 16.0);
                for r in [20.0, 45.0, 80.0] {
                    let p = me + dir * r;
                    let d = w.hazards.danger(p) * 10.0 + r * 0.05;
                    if d < best.1 && nav.cell(cell_at(p)).is_none_or(|c| profile.fits(c)) {
                        best = (p, d);
                    }
                }
            }
            a.go(Goal { to: best.0, gait: Gait::Run, within: 4.0, face: None }, Prefs { cover: false, caution: 6.0, max_nodes: 3000 }, ticks, events);
            a.gait = Gait::Run;
            if !busy {
                a.set_act(Act::Idle(0), ticks, changes);
            }
        }
        Fight | Chase | Stalk => {
            let who = match a.behaviour {
                Fight => a.foe.or_else(|| herd.filter(|h| h.leader == Some(a.c.id)).map(|_| Who::Player)),
                _ => herd.and_then(|h| h.target).map(|t| t.who),
            };
            let Some(who) = who else {
                a.behaviour = Idle;
                return;
            };
            let Some(at) = whereabouts(who, seen, w.player) else {
                a.behaviour = Idle;
                a.foe = None;
                return;
            };
            // Each of a pack comes at it from its own side.
            let flank = if a.behaviour == Chase && a.dist_to(at) > 6.0 {
                Vec2::from_angle(a.index as f32 * 2.1) * 3.0
            } else {
                Vec2::ZERO
            };
            let gait = if a.behaviour == Stalk { Gait::Walk } else { Gait::Run };
            let reach = k.attack.as_ref().map_or(1.5, |at| at.reach);
            let prefs = Prefs { cover: a.behaviour == Stalk, caution: 1.0, max_nodes: 2500 };
            a.go(Goal { to: at + flank, gait, within: reach * 0.6, face: Some(at) }, prefs, ticks, events);
            if !busy {
                a.set_act(Act::Idle(0), ticks, changes);
            }
            try_attack(a, who, at, seen, w, changes);
        }
        Feed => {
            let Some(c) = herd.and_then(|h| h.feed).and_then(|c| seen.get(c)) else {
                a.behaviour = Idle;
                return;
            };
            let at = c.xz() + Vec2::from_angle(a.index as f32 * 2.4) * 1.2;
            a.go(Goal { to: at, gait: Gait::Walk, within: 1.2, face: Some(c.xz()) }, calm, ticks, events);
            if arrived(a) {
                a.stop();
                face(a, c.xz());
                a.set_act(Act::Eat, ticks, changes);
                a.hunger = (a.hunger - dt * FEED_RATE).max(0.0);
                events.push(Event::Ate(c.id));
                if a.hunger < 0.05 {
                    a.behaviour = Idle;
                }
            }
        }
        School => {
            let Some(h) = herd else { return };
            let middle = h.middle;
            // Round the school's middle, each on its own circle.
            let t = ticks as f32 / TICKS_PER_SECOND * 0.08 + a.index as f32 * 2.4;
            let r = 1.5 + k.spread * 0.5 * (0.4 + 0.6 * ((a.index as f32 * 0.37).fract()));
            let to = middle + Vec2::from_angle(t) * r;
            a.go(Goal { to, gait: Gait::Walk, within: 1.0, face: None }, calm, ticks, events);
            a.gait = Gait::Walk;
        }
        Curious => {
            let p = w.player.xz();
            let d = a.dist_to(p);
            if d > 8.0 {
                let to = p + (me - p).normalize_or(Vec2::X) * 6.0;
                a.go(Goal { to, gait: Gait::Walk, within: 1.5, face: Some(p) }, calm, ticks, events);
            } else {
                a.stop();
                face(a, p);
                if !busy {
                    a.set_act(Act::Alert, ticks, changes);
                }
            }
        }
    }
}

/// Stand facing `p` (turning there over the next ticks).
fn face(a: &mut Agent, p: Vec2) {
    if let Some(g) = &mut a.goal {
        g.face = Some(p);
    } else {
        a.goal = Some(Goal { to: a.xz(), gait: Gait::Walk, within: 0.5, face: Some(p) });
    }
}

/// Cycle the idle clips every few seconds.
fn idle_clip(a: &mut Agent, ticks: u64, changes: &mut Changes) {
    let k = a.c.kind();
    let n = k.clips.len() as u8;
    let held = ticks.saturating_sub(a.c.since) as f32 / TICKS_PER_SECOND;
    let length = IDLE_SECONDS[0] + (IDLE_SECONDS[1] - IDLE_SECONDS[0]) * a.rand(a.c.since, 0x6c65);
    match a.c.act {
        Act::Idle(_) if held < length || n < 2 => {}
        Act::Idle(i) => {
            let next = (i + 1 + (a.rand(ticks, 0x6163) * (n - 1) as f32) as u8 % (n - 1)) % n;
            a.set_act(Act::Idle(next), ticks, changes);
        }
        _ if a.act_until.is_none() && a.attack.is_none() => a.set_act(Act::Idle(0), ticks, changes),
        _ => {}
    }
}

/// A place to run to: `run` metres off along `away`, or turned somewhat if that's no good (a cliff,
/// a lake).
fn flee_spot(nav: &NavGrid, profile: &crate::nav::Profile, me: Vec2, away: Vec2, run: f32) -> Option<Vec2> {
    for turn in [0.0f32, 0.4, -0.4, 0.8, -0.8, 1.2, -1.2] {
        let dir = Vec2::from_angle(turn).rotate(away);
        for r in [run, run * 0.6, run * 0.35] {
            let p = me + dir * r;
            if nav.cell(cell_at(p)).is_some_and(|c| profile.fits(c)) {
                return Some(p);
            }
        }
    }
    None
}

/// Strike at `who` (at `at`) if it's within reach and the last attack is over.
fn try_attack(a: &mut Agent, who: Who, at: Vec2, seen: &Seen, w: &World, changes: &mut Changes) {
    let k = a.c.kind();
    let Some(attack) = &k.attack else { return };
    if a.attack.is_some() || w.ticks < a.ready_at {
        return;
    }
    let target_radius = match who {
        Who::Creature(id) => seen.get(id).map_or(0.5, |o| super::kinds()[o.kind as usize].radius),
        _ => 0.35,
    };
    if a.dist_to(at) > attack.reach + target_radius + k.radius {
        return;
    }
    face(a, at);
    let hit = w.ticks + (attack.windup * TICKS_PER_SECOND) as u64;
    let end = hit + 6;
    a.attack = Some((who, hit, end));
    a.ready_at = w.ticks + (attack.cooldown * TICKS_PER_SECOND) as u64;
    a.act_until = Some(end);
    a.set_act(Act::Attack, w.ticks, changes);
}

/// Hooves for anything that comes up right behind.
fn kick_behind(a: &mut Agent, w: &World, changes: &mut Changes) {
    let k = a.c.kind();
    if !k.attack.as_ref().is_some_and(|at| at.clip.contains("Kick")) {
        return;
    }
    let Some((who, at)) = a.threat else { return };
    if who == Who::Somewhere || a.dist_to(at) > 2.8 {
        return;
    }
    let to = (at - a.xz()).normalize_or_zero();
    if to.dot(a.c.forward()) < -0.4 && a.attack.is_none() && w.ticks >= a.ready_at {
        // Kicks behind: the blow lands without turning round.
        let attack = k.attack.as_ref().unwrap();
        let hit = w.ticks + (attack.windup * TICKS_PER_SECOND) as u64;
        a.attack = Some((who, hit, hit + 6));
        a.ready_at = w.ticks + (attack.cooldown * TICKS_PER_SECOND) as u64;
        a.act_until = Some(hit + 6);
        a.set_act(Act::Attack, w.ticks, changes);
    }
}
