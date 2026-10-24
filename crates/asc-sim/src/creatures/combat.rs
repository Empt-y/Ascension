//! Blows and wounds: an attack's blow lands when its clip gets there (if the target is still in
//! reach), the hurt flinch, fear or turn on whoever did it, and the dead lie as carcasses.

use std::collections::BTreeMap;

use glam::{Vec2, Vec3};

use super::brain::{Agent, Who, World};
use super::{Act, Changes, CreatureId, HerdId, hash, unit};
use crate::vitals::Cause;

/// The player hits what's within this of their eye (plus the creature's size)...
pub const PLAYER_REACH: f32 = 2.6;
/// ...no more often than every this many ticks...
pub const STRIKE_TICKS: u64 = 9;
/// ...and butchers a carcass within this of their feet.
pub const BUTCHER_REACH: f32 = 3.0;
/// How long the flinch of being hit lasts (ticks).
const FLINCH_TICKS: u64 = 10;

/// Hurt `a` by `damage` (`by` whom, if anyone): it flinches, and takes fright or (by its nature,
/// or with nowhere to go) turns on them. True if that killed it.
pub fn hurt(a: &mut Agent, damage: i32, by: Option<Who>, ticks: u64, changes: &mut Changes) -> bool {
    if a.c.dead() {
        return false;
    }
    let k = a.c.kind();
    a.c.health = (a.c.health - damage).max(0);
    changes.hurt.push((a.c.id, a.c.health));
    if a.c.health == 0 {
        a.goal = None;
        a.path.clear();
        a.attack = None;
        a.act_until = None;
        a.c.speed = 0.0;
        a.set_act(Act::Die, ticks, changes);
        return true;
    }
    a.act_until = Some(ticks + FLINCH_TICKS);
    a.set_act(Act::Hit, ticks, changes);
    a.next_think = ticks;
    if let Some(who) = by {
        let fights = k.attack.is_some()
            && (unit(hash(0, a.c.id.0 as i64, ticks as i64, 0x6669)) < k.aggression || k.predator() || k.guard);
        if fights {
            a.anger = (a.anger + 0.7).min(1.0);
            a.foe = Some(who);
        } else {
            a.fear = 1.0;
        }
        if who == Who::Player {
            a.wary = (a.wary + 0.5 * k.wariness.max(0.3)).min(1.0);
        }
        a.threat = Some((who, a.xz()));
    }
    false
}

/// Blows that land this tick, and attacks that are over. The herds (and whether by the player) of
/// those killed.
pub fn resolve(agents: &mut BTreeMap<CreatureId, Agent>, w: &World, changes: &mut Changes) -> Vec<(HerdId, bool)> {
    let ticks = w.ticks;
    let mut blows: Vec<(Who, CreatureId, Vec3, f32, i32)> = Vec::new();
    for a in agents.values_mut() {
        let Some((who, hit, end)) = a.attack else { continue };
        if ticks == hit
            && let Some(attack) = &a.c.kind().attack
        {
            blows.push((who, a.c.id, a.c.body(), attack.reach, attack.damage));
        }
        if ticks >= end {
            a.attack = None;
        }
    }
    let mut killed = Vec::new();
    for (who, by, from, reach, damage) in blows {
        let kind = agents.get(&by).map_or(0, |a| a.c.kind);
        match who {
            Who::Player => {
                let p = w.player;
                let body = p.feet + Vec3::Y * 0.9;
                if p.alive && body.distance(from) < reach + 1.0 {
                    changes.player.push((damage, Cause::Animal(kind)));
                }
            }
            Who::Creature(id) => {
                let Some(t) = agents.get_mut(&id) else { continue };
                let tk = t.c.kind();
                let gap = Vec2::new(t.c.at.x - from.x, t.c.at.z - from.z).length() - tk.radius;
                if gap < reach + 0.5 && hurt(t, damage, Some(Who::Creature(by)), ticks, changes) {
                    killed.push((t.herd, false));
                }
            }
            Who::Somewhere => {}
        }
    }
    killed
}
