//! The client's copy of the wild creatures near the player, and drawing them.
//!
//! The server sends where each animal is a few times a second; it's drawn `DELAY` behind, between
//! the two positions either side (or a little ahead of the last, if the next is late), so it moves
//! smoothly. What it's doing (`Act`) and how fast it's going choose its clip: walking and galloping
//! play at the rate its speed calls for, and a change of clip crossfades. Far ones are posed less
//! often. Their bodies stop the player, and the crosshair finds them (to strike, or to butcher a
//! carcass).

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;

use asc_render::SkinnedInstance;
use asc_sim::creatures::{Act, Creature, CreatureId, Motion};
use glam::{Mat4, Vec2, Vec3};

use crate::animation::{Playing, Poser};
use crate::assets::{Animated, CreatureModels};

/// Beyond this creatures aren't drawn...
const DRAW: f32 = 220.0;
/// ...and beyond this they're posed every fourth frame.
const NEAR: f32 = 60.0;
/// A change of clip blends over this long (seconds).
const CROSSFADE: f32 = 0.3;
/// Drawn this far (seconds) behind the server, so there's usually a position either side...
const DELAY: f64 = 0.2;
/// ...and when there isn't, carried on along the last one's way for at most this long.
const EXTRAPOLATE: f64 = 0.5;
/// Positions kept per creature.
const KEEP: usize = 8;
/// Bodies within this of the player can stop them.
const COLLIDE_NEAR: f32 = 8.0;

/// Where it was at a time (seconds of game time).
#[derive(Clone, Copy, Debug)]
struct Snap {
    time: f64,
    at: Vec3,
    yaw: f32,
    speed: f32,
}

struct Entry {
    creature: Creature,
    snaps: VecDeque<Snap>,
    /// The clip playing, and the one fading out with how far the fade has got.
    playing: Option<Playing>,
    fading: Option<(Playing, f32)>,
    /// When it was last posed (game seconds), and the pose.
    posed: f64,
    palette: Option<Arc<[Mat4]>>,
    /// Seconds added to its looping clips, so a herd isn't in step.
    phase: f32,
    /// Where it's drawn this frame, which way it faces, how fast it's going.
    drawn: (Vec3, f32, f32),
}

/// A creature under the crosshair.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CreatureHit {
    pub id: CreatureId,
    /// How far along the ray.
    pub distance: f32,
    pub dead: bool,
}

/// A body: a capsule from `a` to `b`.
#[derive(Clone, Copy, Debug)]
struct Body {
    a: Vec3,
    b: Vec3,
    radius: f32,
}

impl Body {
    fn distance(&self, p: Vec3) -> f32 {
        let ab = self.b - self.a;
        let t = if ab.length_squared() > 0.0 { ((p - self.a).dot(ab) / ab.length_squared()).clamp(0.0, 1.0) } else { 0.0 };
        (p - (self.a + ab * t)).length() - self.radius
    }
}

#[derive(Default)]
pub struct ClientCreatures {
    creatures: BTreeMap<CreatureId, Entry>,
    poser: Poser,
    frame: u64,
    /// The live bodies near the player this frame.
    near: Vec<Body>,
}

fn body_of(c: &Creature, at: Vec3, yaw: f32) -> Body {
    let k = c.kind();
    let fwd = Vec3::new(yaw.sin(), 0.0, yaw.cos());
    if c.dead() {
        // Lying on its side.
        let mid = at + Vec3::Y * k.radius * 0.6;
        let half = (k.height * 0.5).max(k.radius);
        return Body { a: mid - fwd * half * 0.5, b: mid + fwd * half * 0.5, radius: k.radius * 0.9 };
    }
    let mid = if k.swims() { at } else { at + Vec3::Y * k.height * 0.55 };
    let radius = k.radius.max(k.height * 0.3);
    // Longer than wide: about as long as it's tall.
    let half = (k.height * 0.6 - radius).max(0.0);
    Body { a: mid - fwd * half, b: mid + fwd * half, radius }
}

/// The nearest point along a ray (origin, unit dir) where it meets the capsule, if it does.
fn ray_body(origin: Vec3, dir: Vec3, body: &Body, reach: f32) -> Option<f32> {
    // March: capsules are simple enough that sphere tracing converges in a few steps.
    let mut t = 0.0;
    for _ in 0..24 {
        let d = body.distance(origin + dir * t);
        if d < 0.01 {
            return Some(t);
        }
        t += d;
        if t > reach {
            return None;
        }
    }
    None
}

impl ClientCreatures {
    pub fn spawned(&mut self, list: &[Creature], clock: f64) {
        for c in list {
            let phase = (c.id.0 >> 40) as f32 / (1u64 << 24) as f32 * 30.0;
            let mut snaps = VecDeque::new();
            snaps.push_back(Snap { time: clock - DELAY, at: c.at, yaw: c.yaw, speed: c.speed });
            self.creatures.insert(c.id, Entry {
                creature: *c,
                snaps,
                playing: None,
                fading: None,
                posed: clock,
                palette: None,
                phase,
                drawn: (c.at, c.yaw, c.speed),
            });
        }
    }

    pub fn gone(&mut self, ids: &[CreatureId]) {
        for id in ids {
            self.creatures.remove(id);
        }
    }

    pub fn moves(&mut self, moves: &[Motion]) {
        for m in moves {
            if let Some(e) = self.creatures.get_mut(&m.id) {
                let time = m.tick as f64 / 20.0;
                // Out of order (or repeated): keep the times rising.
                while e.snaps.back().is_some_and(|s| s.time >= time) {
                    e.snaps.pop_back();
                }
                e.snaps.push_back(Snap { time, at: m.at, yaw: m.yaw, speed: m.speed });
                while e.snaps.len() > KEEP {
                    e.snaps.pop_front();
                }
                e.creature.at = m.at;
                e.creature.yaw = m.yaw;
                e.creature.speed = m.speed;
            }
        }
    }

    /// Creatures that started doing something else at `since` (ticks).
    pub fn acts(&mut self, acts: &[(CreatureId, Act, u64)]) {
        for &(id, act, since) in acts {
            if let Some(e) = self.creatures.get_mut(&id) {
                e.creature.act = act;
                e.creature.since = since;
            }
        }
    }

    pub fn hurt(&mut self, hurt: &[(CreatureId, i32)]) {
        for &(id, health) in hurt {
            if let Some(e) = self.creatures.get_mut(&id) {
                e.creature.health = health;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.creatures.len()
    }

    /// Where `id` is drawn this frame.
    pub fn drawn_at(&self, id: CreatureId) -> Option<Vec3> {
        self.creatures.get(&id).map(|e| e.drawn.0)
    }

    /// The creatures within `reach` of `p`: id, where drawn, and what it is.
    pub fn near(&self, p: Vec3, reach: f32) -> Vec<(CreatureId, Vec3, &Creature)> {
        self.creatures.values().filter(|e| e.drawn.0.distance(p) < reach).map(|e| (e.creature.id, e.drawn.0, &e.creature)).collect()
    }

    pub fn get(&self, id: CreatureId) -> Option<&Creature> {
        self.creatures.get(&id).map(|e| &e.creature)
    }

    /// The creature (or carcass) the ray from `origin` along `dir` meets first within `reach`.
    pub fn pick(&self, origin: Vec3, dir: Vec3, reach: f32) -> Option<CreatureHit> {
        self.creatures
            .values()
            .filter(|e| e.drawn.0.distance(origin) < reach + 6.0)
            .filter_map(|e| {
                let body = body_of(&e.creature, e.drawn.0, e.drawn.1);
                ray_body(origin, dir, &body, reach).map(|t| CreatureHit { id: e.creature.id, distance: t, dead: e.creature.dead() })
            })
            .min_by(|a, b| a.distance.total_cmp(&b.distance))
    }

    /// Signed distance to the nearest live body near the player (large when none is).
    pub fn sdf(&self, p: Vec3) -> f32 {
        self.near.iter().map(|b| b.distance(p)).fold(f32::MAX, f32::min)
    }

    /// Every creature near `camera`, posed at game time `clock` (seconds), as skinned instances;
    /// and the bodies near `player` for collision.
    pub fn instances(&mut self, models: &CreatureModels, clock: f64, camera: Vec3, player: Vec3, out: &mut Vec<SkinnedInstance>) {
        self.frame += 1;
        self.near.clear();
        let when = clock - DELAY;
        for (id, e) in &mut self.creatures {
            let (at, yaw, speed) = sample(&e.snaps, when);
            e.drawn = (at, yaw, speed);
            let c = e.creature;
            if !c.dead() && at.distance(player) < COLLIDE_NEAR {
                self.near.push(body_of(&c, at, yaw));
            }
            let d = at.distance(camera);
            let Some(model) = models.get(c.kind, c.model) else { continue };
            if d > DRAW {
                continue;
            }
            let due = d < NEAR || e.palette.is_none() || (self.frame + id.0).is_multiple_of(4);
            if due {
                let dt = (clock - e.posed).clamp(0.0, 0.5) as f32;
                e.posed = clock;
                let pose = choose(model, &c, speed, clock);
                advance(e, pose, dt);
                e.palette = Some(match e.playing {
                    Some(now) => self.poser.palette(model, now, e.fading),
                    None => self.poser.rest(model),
                });
            }
            let Some(palette) = &e.palette else { continue };
            out.push(SkinnedInstance {
                model: model.id,
                key: id.0,
                position: at,
                yaw,
                scale: c.kind().scale,
                palette: palette.clone(),
                shadow_only: false,
                casts_shadow: true,
            });
        }
    }
}

/// Where it is at `t`: between the positions either side, or on from the last.
fn sample(snaps: &VecDeque<Snap>, t: f64) -> (Vec3, f32, f32) {
    let first = snaps.front().expect("a creature always has a position");
    if t <= first.time {
        return (first.at, first.yaw, first.speed);
    }
    for w in snaps.iter().zip(snaps.iter().skip(1)) {
        let (a, b) = w;
        if t <= b.time {
            let f = ((t - a.time) / (b.time - a.time).max(1e-6)) as f32;
            let dyaw = (b.yaw - a.yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
            return (a.at.lerp(b.at, f), a.yaw + dyaw * f, a.speed + (b.speed - a.speed) * f);
        }
    }
    let last = snaps.back().unwrap();
    let ahead = (t - last.time).min(EXTRAPOLATE) as f32;
    let dir = Vec2::new(last.yaw.sin(), last.yaw.cos()) * last.speed * ahead;
    (last.at + Vec3::new(dir.x, 0.0, dir.y), last.yaw, last.speed)
}

/// What to play: the clip, how fast, whether it loops, and (one-shots) how far in it is by now.
struct Pose {
    clip: usize,
    rate: f32,
    looped: bool,
    at: Option<f32>,
}

fn first_clip(model: &Animated, names: &[&str]) -> Option<usize> {
    names.iter().find_map(|n| model.clip(n))
}

fn choose(model: &Animated, c: &Creature, speed: f32, clock: f64) -> Option<Pose> {
    let k = c.kind();
    let since = (clock - c.since as f64 / 20.0).max(0.0) as f32;
    let once = |names: &[&str]| first_clip(model, names).map(|clip| Pose { clip, rate: 1.0, looped: false, at: Some(since) });
    let looping = |name: &str, rate: f32| model.clip(name).map(|clip| Pose { clip, rate, looped: true, at: None });
    let moving = || {
        if k.swims() {
            return looping(&k.clips[0], (speed / k.stride[0]).clamp(0.4, 2.5));
        }
        if speed < 0.15 {
            return None;
        }
        if speed < (k.walk + k.run) * 0.35 {
            looping("Walk", (speed / k.stride[0]).clamp(0.4, 2.5))
        } else {
            looping("Gallop", (speed / k.stride[1]).clamp(0.5, 1.8))
        }
    };
    let pose = match c.act {
        Act::Die => once(&["Death"]),
        Act::Attack => k.attack.as_ref().and_then(|a| once(&[a.clip.as_str()])),
        Act::Hit => once(&["Idle_HitReact1", "Idle_HitReact_Left", "Idle_HitReact2"]),
        Act::Jump => once(&["Gallop_Jump"]),
        Act::Eat => looping(&k.eat_clip, 1.0),
        Act::Sleep => looping(&k.sleep_clip, 0.35),
        Act::Alert => moving().or_else(|| looping("Idle", 1.0)),
        Act::Idle(n) => moving().or_else(|| looping(&k.clips[n as usize % k.clips.len()], 1.0)),
    };
    pose.or_else(|| (!model.clips.is_empty()).then_some(Pose { clip: 0, rate: 1.0, looped: true, at: None }))
}

/// Step the clips on by `dt`: the same clip runs on (at its rate now); a new one starts, the old
/// fading out.
fn advance(e: &mut Entry, pose: Option<Pose>, dt: f32) {
    let Some(pose) = pose else { return };
    match &mut e.playing {
        Some(p) if p.clip == pose.clip && p.looped == pose.looped => {
            p.time = match pose.at {
                Some(at) => at,
                None => p.time + dt * pose.rate,
            };
        }
        _ => {
            e.fading = e.playing.map(|p| (p, 0.0));
            e.playing = Some(Playing { clip: pose.clip, time: pose.at.unwrap_or(e.phase), looped: pose.looped });
        }
    }
    if let Some((p, f)) = &mut e.fading {
        p.time += dt;
        *f += dt / CROSSFADE;
        if *f >= 1.0 {
            e.fading = None;
        }
    }
}
