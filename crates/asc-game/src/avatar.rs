//! The player's own body: what it's doing (from the player's movement and actions) chooses a clip
//! from the animation libraries, crossfading from the last, and the body turns to face the way
//! it's going.

use std::sync::Arc;

use glam::{Mat4, Vec2, Vec3};

use crate::animation::{Layer, Playing, Poser};
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
    /// Seconds since last striking (and how many strikes there have been: hands take turns),
    /// since last hurt, and whether butchering a carcass.
    pub since_struck: Option<(f32, u32)>,
    pub since_hurt: Option<f32>,
    pub butchering: bool,
    /// What's in the hands, and how the view is pitched (radians, up positive).
    pub held: Held,
    pub look_pitch: f32,
}

/// What the hands are doing with what they hold (played over the arms and chest, so the legs walk
/// on underneath).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub enum Held {
    #[default]
    Nothing,
    /// Carrying it (a beacon, an idle scanner).
    Carry,
    /// Aiming it along the view (the multitool, the manipulator).
    Aim,
    /// Seconds since a shot.
    Shoot(f32),
    /// Seconds since a scan.
    Scan(f32),
    Torch,
    /// Seconds since starting an injection.
    Inject(f32),
}

impl Held {
    /// Whether the body turns to face where the player looks.
    fn faces_view(self) -> bool {
        !matches!(self, Self::Nothing | Self::Carry | Self::Torch)
    }
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
    /// The arms' clip, how much of it shows (0..1), and the joints it's on (from the chest up).
    layer: Option<Playing>,
    layer_weight: f32,
    upper: Vec<bool>,
}

/// The arms' clip for what's held: the clip, whether it loops, and (one-shots) how far in.
fn layer_clip(model: &PlayerModel, held: Held) -> Option<(usize, bool, Option<f32>)> {
    let clip = |n: &str| model.body.clip(n);
    let (name, once) = match held {
        Held::Nothing => return None,
        Held::Carry => ("Pistol_Idle_Loop", None),
        Held::Aim => ("Pistol_Aim_Neutral", None),
        Held::Shoot(t) if t < 0.35 => ("Pistol_Shoot", Some(t)),
        Held::Shoot(_) => ("Pistol_Aim_Neutral", None),
        Held::Scan(t) if t < 0.9 => ("Spell_Simple_Shoot", Some(t)),
        Held::Scan(_) => ("Spell_Simple_Idle_Loop", None),
        Held::Torch => ("Idle_Torch_Loop", None),
        Held::Inject(t) => ("Consume", Some(t)),
    };
    clip(name).map(|c| (c, once.is_none(), once))
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
    if let Some((t, n)) = s.since_struck
        && t < 0.45
    {
        return (clip(if n % 2 == 0 { "Punch_Cross" } else { "Punch_Jab" }), None);
    }
    if s.since_hurt.is_some_and(|t| t < 0.35) && speed < 2.0 {
        return (clip("Hit_Chest"), None);
    }
    if s.butchering && speed < 0.5 {
        return (clip("Farm_Harvest"), Some(1.0));
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
    /// Step `dt` seconds on from the last frame in `state`: the body's yaw, its joint palette, and
    /// where its right hand is (model space).
    pub fn update(&mut self, model: &PlayerModel, state: &State, dt: f32) -> (f32, Arc<[Mat4]>, Mat4) {
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
        let facing = if state.held.faces_view() && flat.length() < 5.0 { Vec2::new(state.look_yaw.cos(), state.look_yaw.sin()) } else { facing };
        let target = asc_sim::creatures::yaw_facing(facing);
        let diff = (target - self.yaw + std::f32::consts::PI).rem_euclid(std::f32::consts::TAU) - std::f32::consts::PI;
        self.yaw += diff.clamp(-TURN_RATE * dt, TURN_RATE * dt);
        // The arms: what's held, eased in and out.
        let skeleton = &model.body.skeleton;
        if self.upper.is_empty() {
            self.upper = skeleton.index("spine_02").map_or_else(|| vec![false; skeleton.len()], |j| skeleton.below(j));
        }
        let want = layer_clip(model, state.held);
        self.layer_weight = (self.layer_weight + if want.is_some() { dt } else { -dt } / 0.2).clamp(0.0, 1.0);
        if let Some((clip, looped, at)) = want {
            match &mut self.layer {
                Some(p) if p.clip == clip => p.time = at.unwrap_or(p.time + dt),
                _ => self.layer = Some(Playing { clip, time: at.unwrap_or(0.0), looped }),
            }
        }
        // Aiming: the neutral aim blended towards up or down with the view's pitch.
        let with = (matches!(state.held, Held::Aim | Held::Shoot(_)))
            .then(|| {
                let name = if state.look_pitch > 0.0 { "Pistol_Aim_Up" } else { "Pistol_Aim_Down" };
                model.body.clip(name).map(|c| (Playing { clip: c, time: 0.0, looped: true }, (state.look_pitch.abs() / 1.1).min(1.0)))
            })
            .flatten()
            .filter(|_| !matches!(state.held, Held::Shoot(t) if t < 0.35));
        let layer = self.layer.filter(|_| self.layer_weight > 0.0).map(|now| Layer { now, with, weight: self.layer_weight, mask: &self.upper });
        let palette = match self.playing {
            Some(now) => self.poser.layered(&model.body, now, self.fading, layer),
            None => self.poser.rest(&model.body),
        };
        let hand = skeleton.index("hand_r").map_or(Mat4::IDENTITY, |j| palette[j] * skeleton.inverse_bind[j].inverse());
        (self.yaw, palette, hand)
    }
}
