//! Moving: following the path to the goal, keeping clear of the others and the player, turning and
//! speeding up as a body does, staying on ground it can stand on (or in water it swims in), and
//! running only as long as its wind lasts.

use glam::{Vec2, Vec3};

use super::brain::{Agent, Gait, Seen};
use super::{Act, Changes, PlayerView, TICKS_PER_SECOND, yaw_facing};
use crate::nav::{NavGrid, cell_at};

const DT: f32 = 1.0 / TICKS_PER_SECOND;
/// A waypoint is reached within this (m).
const WAYPOINT: f32 = 0.8;

fn turn_towards(yaw: f32, target: f32, max: f32) -> f32 {
    let diff = (target - yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
    yaw + diff.clamp(-max, max)
}

pub fn step(a: &mut Agent, nav: &NavGrid, player: &PlayerView, seen: &Seen, ticks: u64, changes: &mut Changes) {
    let k = a.c.kind();
    let me = a.xz();
    if a.c.dead() {
        a.c.speed = 0.0;
        a.velocity = Vec2::ZERO;
        return;
    }
    // Stood still for whatever it's doing.
    // (A kick is aimed standing; a bite or a butt comes on at a run.)
    let kicking = a.attack.is_some() && k.attack.as_ref().is_some_and(|at| at.clip.contains("Kick"));
    let standing = matches!(a.c.act, Act::Eat | Act::Sleep) || kicking;
    let mut want_speed = 0.0;
    let mut want_dir: Option<Vec2> = None;
    if let Some(goal) = a.goal {
        let to_goal = goal.to - me;
        if to_goal.length() > goal.within && !standing {
            // Head for the next waypoint, or straight at the goal while the path is coming.
            while a.waypoint < a.path.len() && Vec2::new(a.path[a.waypoint].x, a.path[a.waypoint].z).distance(me) < WAYPOINT {
                a.waypoint += 1;
            }
            let target = match a.path.get(a.waypoint) {
                Some(p) => Some(Vec2::new(p.x, p.z)),
                None if a.path.is_empty() && a.asked.is_some() && to_goal.length() < 4.0 => Some(goal.to),
                None if !a.path.is_empty() => Some(goal.to),
                None => None,
            };
            if let Some(t) = target {
                want_dir = (t - me).try_normalize();
                let gait = match a.gait {
                    Gait::Run if a.stamina <= 0.0 => Gait::Trot,
                    g => g,
                };
                want_speed = match gait {
                    Gait::Walk => k.walk,
                    Gait::Trot => (k.walk + k.run) * 0.45,
                    Gait::Run => k.run,
                };
                // Slow coming up to the goal.
                want_speed = want_speed.min(to_goal.length() * 1.5 + 0.5);
            }
        } else if let Some(face) = goal.face {
            want_dir = (face - me).try_normalize();
        }
    }
    // Keep clear of the herd close by (and of anything it's bumping into, except what it's after),
    // and of the player.
    let mut push = Vec2::ZERO;
    let after = a.foe.or(a.attack.map(|x| x.0));
    if want_speed > 0.0 {
        for o in seen.near(me, k.radius * 2.0 + 2.0) {
            if o.id == a.c.id || !o.alive || after == Some(super::brain::Who::Creature(o.id)) {
                continue;
            }
            let other = super::kinds()[o.kind as usize].radius;
            let off = me - o.xz();
            let gap = off.length() - k.radius - other;
            let room = if o.herd == a.herd { 0.8 } else { 0.1 };
            if gap < room {
                push += off.normalize_or_zero() * (room - gap);
            }
        }
        let off = me - player.xz();
        let gap = off.length() - k.radius - 0.35;
        if player.alive && gap < 1.0 && a.attack.is_none() {
            push += off.normalize_or_zero() * (1.0 - gap) * 2.0;
        }
    }
    let dir = match want_dir {
        Some(d) => (d + push * 0.5).normalize_or(d),
        None => a.c.forward(),
    };
    // Turn: quickly when slow, more widely when fast.
    let turn = k.turn * (1.0 + 2.0 / (1.0 + a.c.speed)) * DT;
    a.c.yaw = turn_towards(a.c.yaw, yaw_facing(dir), turn);
    let accel = if want_speed > a.c.speed { if a.gait == Gait::Run { 6.0 } else { 2.5 } } else { 8.0 };
    a.c.speed = if want_speed > a.c.speed { (a.c.speed + accel * DT).min(want_speed) } else { (a.c.speed - accel * DT).max(want_speed) };
    // Running winds it; walking and standing, it gets its breath back.
    if a.c.speed > (k.walk + k.run) * 0.5 {
        a.stamina = (a.stamina - DT / k.stamina).max(0.0);
    } else {
        a.stamina = (a.stamina + DT / (k.stamina * 2.0)).min(1.0);
    }
    if a.c.speed < 0.01 {
        a.c.speed = 0.0;
        a.velocity = Vec2::ZERO;
        return;
    }
    let profile = k.profile();
    let here_cell = nav.cell(cell_at(me));
    let can_go = |next: Vec2| {
        let next_cell = nav.cell(cell_at(next));
        match (here_cell, next_cell) {
            (_, None) => false,
            (Some(h), Some(n)) if cell_at(me) != cell_at(next) => profile.steps(h, n, (cell_at(next) - cell_at(me)).as_vec2().length()),
            (None, Some(n)) => profile.fits(n),
            // Within its cell: fine if it fits (or it was already somewhere it doesn't, and is
            // getting out).
            (Some(h), Some(n)) => profile.fits(n) || !profile.fits(h),
        }
    };
    // On the way it's facing; or, swinging wide into something, the way it wants to go, or round
    // what's in the way.
    let step = a.c.speed * DT;
    let want = want_dir.unwrap_or(a.c.forward());
    let mut moved = None;
    for dir in [a.c.forward(), want, Vec2::from_angle(0.5).rotate(want), Vec2::from_angle(-0.5).rotate(want), Vec2::from_angle(1.0).rotate(want), Vec2::from_angle(-1.0).rotate(want)] {
        if can_go(me + dir * step) {
            moved = Some(dir);
            break;
        }
    }
    let Some(forward) = moved else {
        // Hemmed in (or the ground not ready): stop, and find the way again.
        a.stuck += 1;
        a.c.speed = 0.0;
        a.velocity = Vec2::ZERO;
        a.path.clear();
        a.asked = None;
        return;
    };
    if forward != a.c.forward() {
        a.c.yaw = turn_towards(a.c.yaw, yaw_facing(forward), turn * 2.0);
    }
    let next = me + forward * step;
    let next_cell = nav.cell(cell_at(next));
    a.stuck = 0;
    let y = if k.swims() {
        let c = next_cell.unwrap();
        let surface = c.height + c.depth();
        let room = (c.depth() - 1.0).max(0.0);
        surface - 0.5 - room * a.swim_depth.clamp(0.1, 0.9)
    } else {
        nav.height_at(next).unwrap_or(a.c.at.y)
    };
    // Up a ledge at a run: a leap (not just a steep stride).
    let rise = y - a.c.at.y;
    if !k.swims() && rise > 0.5 && rise > step * 0.9 && a.c.speed > k.walk * 2.0 && a.act_until.is_none() && a.attack.is_none() {
        a.act_until = Some(ticks + 12);
        a.set_act(Act::Jump, ticks, changes);
    }
    a.velocity = forward * a.c.speed;
    a.c.at = Vec3::new(next.x, if k.swims() { a.c.at.y + (y - a.c.at.y) * 0.1 } else { y }, next.y);
}
