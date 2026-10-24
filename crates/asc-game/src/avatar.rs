//! The player's own body: what it's doing (from the player's movement and actions) chooses a clip
//! from the animation libraries, crossfading from the last, and the body turns to face the way
//! it's going.

use std::sync::Arc;

use glam::{Mat4, Vec2, Vec3};

use crate::animation::{Playing, Poser};
use crate::assets::PlayerModel;
use crate::player::Mode;

/// A change of clip blends over this long (seconds).
const CROSSFADE: f32 = 0.2;
/// How fast the body turns to face its way (radians a second).
const TURN_RATE: f32 = 10.0;

/// What the player is doing this frame.
#[derive(Clone, Copy, Debug)]
pub struct State {
    pub mode: Mode,
    pub grounded: bool,
    pub velocity: Vec3,
    /// 0..1 how deep in water the body is.
    pub submerged: f32,
    /// The view's yaw (game convention: forward is (cos, 0, sin)).
    pub look_yaw: f32,
    /// Holding the mine button on something.
    pub mining: bool,
    /// Seconds since last placing something, and since dying (if recent).
    pub since_placed: Option<f32>,
    pub since_died: Option<f32>,
}

#[derive(Default)]
pub struct Avatar {
    playing: Option<Playing>,
    /// The clip being faded out, and how far the fade has got (0..1).
    fading: Option<(Playing, f32)>,
    /// The body's yaw (model convention: see `asc_sim::creatures::yaw_facing`).
    yaw: f32,
    /// Seconds off the ground, and seconds since landing.
    airborne: f32,
    landed: f32,
    poser: Poser,
}

/// The clip a state calls for, and its playback rate (looped clips) or `None` for a one-shot.
fn choose(model: &PlayerModel, s: &State, airborne: f32, landed: f32) -> (Option<usize>, Option<f32>) {
    let speed = Vec2::new(s.velocity.x, s.velocity.z).length();
    let clip = |name: &str| model.body.clip(name);
    if s.since_died.is_some_and(|t| t < 4.0) {
        return (clip("Death01"), None);
    }
    if s.mode == Mode::Fly {
        return (clip("Jump_Loop"), Some(1.0));
    }
    if s.submerged > 0.35 && !s.grounded {
        return if speed > 0.6 { (clip("Swim_Fwd_Loop"), Some((speed / 3.0).clamp(0.7, 1.4))) } else { (clip("Swim_Idle_Loop"), Some(1.0)) };
    }
    if !s.grounded && airborne > 0.12 {
        return if airborne < 0.35 && s.velocity.y > 0.0 { (clip("Jump_Start"), None) } else { (clip("Jump_Loop"), Some(1.0)) };
    }
    if landed < 0.25 && speed < 1.0 {
        return (clip("Jump_Land"), None);
    }
    if s.since_placed.is_some_and(|t| t < 0.6) && speed < 1.0 {
        return (clip("Interact"), None);
    }
    if s.mining && speed < 0.5 {
        return (clip("TreeChopping_Loop"), Some(1.0));
    }
    match speed {
        v if v < 0.3 => (clip("Idle_Loop"), Some(1.0)),
        v if v < 2.6 => (clip("Walk_Loop"), Some((v / 1.6).clamp(0.6, 1.5))),
        v if v < 6.2 => (clip("Jog_Fwd_Loop"), Some((v / 4.5).clamp(0.7, 1.4))),
        v => (clip("Sprint_Loop"), Some((v / 7.5).clamp(0.8, 1.3))),
    }
}

impl Avatar {
    /// Step `dt` seconds on from the last frame in `state`; the body's yaw and joint palette.
    pub fn update(&mut self, model: &PlayerModel, state: &State, dt: f32) -> (f32, Arc<[Mat4]>) {
        if state.grounded || state.mode == Mode::Fly {
            if self.airborne > 0.4 {
                self.landed = 0.0;
            }
            self.airborne = 0.0;
        } else {
            self.airborne += dt;
        }
        self.landed += dt;
        let (want, rate) = choose(model, state, self.airborne, self.landed);
        if let Some(want) = want {
            match &mut self.playing {
                Some(p) if p.clip == want => p.time += dt * rate.unwrap_or(1.0),
                _ => {
                    self.fading = self.playing.map(|p| (p, 0.0));
                    self.playing = Some(Playing { clip: want, time: 0.0, looped: rate.is_some() });
                }
            }
        }
        if let Some((p, f)) = &mut self.fading {
            p.time += dt;
            *f += dt / CROSSFADE;
            if *f >= 1.0 {
                self.fading = None;
            }
        }
        // Facing the way it moves, or (standing, working) the way it looks.
        let flat = Vec2::new(state.velocity.x, state.velocity.z);
        let facing = if flat.length() > 0.8 && !state.mining {
            flat
        } else {
            Vec2::new(state.look_yaw.cos(), state.look_yaw.sin())
        };
        let target = asc_sim::creatures::yaw_facing(facing);
        let diff = (target - self.yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
        self.yaw += diff.clamp(-TURN_RATE * dt, TURN_RATE * dt);
        let palette = match self.playing {
            Some(now) => self.poser.palette(&model.body, now, self.fading),
            None => self.poser.rest(&model.body),
        };
        (self.yaw, palette)
    }
}
