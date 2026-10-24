//! The client's copy of the wild creatures near the player, and drawing them: each plays the clip
//! for what it's doing (see `asc_sim::creatures`), crossfading when that changes; swimmers move
//! round their circles. Far ones are posed less often.

use std::collections::BTreeMap;
use std::sync::Arc;

use asc_render::SkinnedInstance;
use asc_sim::creatures::{Creature, CreatureId};
use glam::{Mat4, Vec3};

use crate::animation::{Playing, Poser, seconds_since};
use crate::assets::CreatureModels;

/// Beyond this creatures aren't drawn...
const DRAW: f32 = 220.0;
/// ...and beyond this they're posed every fourth frame.
const NEAR: f32 = 60.0;
/// A change of what an animal's doing blends over this long (seconds).
const CROSSFADE: f32 = 0.35;

struct Entry {
    creature: Creature,
    /// What it was doing before, and the game time (seconds) it changed.
    was: Option<(u8, u64, f64)>,
    palette: Option<Arc<[Mat4]>>,
    /// Seconds added to its clips' time, so a herd isn't in step.
    phase: f32,
}

#[derive(Default)]
pub struct ClientCreatures {
    creatures: BTreeMap<CreatureId, Entry>,
    poser: Poser,
    frame: u64,
}

impl ClientCreatures {
    pub fn spawned(&mut self, list: &[Creature]) {
        for c in list {
            let phase = (c.id.0 >> 40) as f32 / (1u64 << 24) as f32 * 30.0;
            self.creatures.insert(c.id, Entry { creature: *c, was: None, palette: None, phase });
        }
    }

    pub fn gone(&mut self, ids: &[CreatureId]) {
        for id in ids {
            self.creatures.remove(id);
        }
    }

    /// Creatures that started doing something else at `since` (ticks); `clock` is now (seconds).
    pub fn acts(&mut self, acts: &[(CreatureId, u8, u64)], clock: f64) {
        for &(id, act, since) in acts {
            if let Some(e) = self.creatures.get_mut(&id) {
                e.was = Some((e.creature.act, e.creature.since, clock));
                e.creature.act = act;
                e.creature.since = since;
            }
        }
    }

    pub fn len(&self) -> usize {
        self.creatures.len()
    }

    /// Every creature near `camera`, posed at game time `clock` (seconds), as skinned instances.
    pub fn instances(&mut self, models: &CreatureModels, clock: f64, camera: Vec3, out: &mut Vec<SkinnedInstance>) {
        self.frame += 1;
        for (id, e) in &mut self.creatures {
            let c = &e.creature;
            let Some(model) = models.get(c.kind, c.model) else { continue };
            let (at, yaw) = c.pose_at(clock as f32);
            let d = at.distance(camera);
            if d > DRAW {
                continue;
            }
            let kind = c.kind();
            let (phase, was) = (e.phase, e.was);
            let play = |act: u8, since: u64| {
                let name = &kind.clips[act as usize % kind.clips.len()];
                model.clip(name).or_else(|| (!model.clips.is_empty()).then_some(0)).map(|clip| Playing {
                    clip,
                    time: seconds_since(clock, since) + phase,
                    looped: true,
                })
            };
            let due = d < NEAR || e.palette.is_none() || (self.frame + id.0).is_multiple_of(4);
            if due {
                e.palette = Some(match play(c.act, c.since) {
                    Some(now) => {
                        let from = was.and_then(|(act, since, changed)| {
                            let w = ((clock - changed) as f32 / CROSSFADE).min(1.0);
                            (w < 1.0).then(|| play(act, since).map(|p| (p, w))).flatten()
                        });
                        self.poser.palette(model, now, from)
                    }
                    None => self.poser.rest(model),
                });
            }
            let Some(palette) = &e.palette else { continue };
            out.push(SkinnedInstance {
                model: model.id,
                key: id.0,
                position: at,
                yaw,
                scale: kind.scale,
                palette: palette.clone(),
                shadow_only: false,
            });
        }
    }
}
