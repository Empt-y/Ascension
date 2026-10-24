//! Handheld tools: what each does, and the heat and charge that limit them.
//!
//! A tool is an item with a `tool` kind (`items.toml`); holding it means having it selected in the
//! hotbar. The client says what it does with it (`ToolUse`) and the server checks it: that the
//! tool is the one held, the target in reach, the rate fair, and the tool not overheated or out of
//! charge. The laser, flamer and manipulator heat while they're used and overheat (locking the
//! tool a few seconds) if held on too long; the rifle's shots come from a charge that refills.

use glam::Vec3;
use serde::{Deserialize, Serialize};

use crate::creatures::CreatureId;

/// What a tool item is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ToolKind {
    Scanner,
    Multitool,
    Manipulator,
    Torch,
    Beacon,
    Injector,
}

impl ToolKind {
    pub const ALL: [ToolKind; 6] = [Self::Scanner, Self::Multitool, Self::Manipulator, Self::Torch, Self::Beacon, Self::Injector];

    /// Its model's name (`assets/models/tools/<name>.glb`) and its item's id.
    pub fn name(self) -> &'static str {
        match self {
            Self::Scanner => "scanner",
            Self::Multitool => "multitool",
            Self::Manipulator => "manipulator",
            Self::Torch => "torch",
            Self::Beacon => "beacon",
            Self::Injector => "injector",
        }
    }

    /// Seconds of use from cold until it overheats (`None`: it doesn't heat).
    pub fn heats(self) -> Option<f32> {
        match self {
            Self::Multitool => Some(HEAT_SECONDS),
            Self::Manipulator => Some(MANIPULATOR_HEAT_SECONDS),
            _ => None,
        }
    }
}

/// The multitool's modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Mode {
    #[default]
    Laser,
    Rifle,
    Flamer,
}

impl Mode {
    pub fn next(self) -> Self {
        match self {
            Self::Laser => Self::Rifle,
            Self::Rifle => Self::Flamer,
            Self::Flamer => Self::Laser,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Laser => "Mining laser",
            Self::Rifle => "Rifle",
            Self::Flamer => "Flamethrower",
        }
    }
}

/// The laser mines this far off, this many times faster than hands, and burns creatures for this
/// much a hit (it hits ten times a second).
pub const LASER_REACH: f32 = 20.0;
pub const LASER_SPEED: f32 = 2.5;
pub const LASER_DAMAGE: i32 = 12;
/// The rifle: reach, damage, the most charge (shots), its refill (shots a second), and the
/// quickest it fires (seconds).
pub const RIFLE_REACH: f32 = 80.0;
pub const RIFLE_DAMAGE: i32 = 140;
pub const RIFLE_CHARGE: f32 = 10.0;
pub const RIFLE_REFILL: f32 = 1.2;
pub const RIFLE_INTERVAL: f32 = 0.15;
/// The flamer: reach, the half-angle of its cone (radians), damage a burst (ten a second).
pub const FLAME_REACH: f32 = 8.0;
pub const FLAME_ANGLE: f32 = 0.3;
pub const FLAME_DAMAGE: i32 = 15;
/// The manipulator's reach, and its brush's radius range.
pub const TERRAFORM_REACH: f32 = 12.0;
pub const BRUSH: [f32; 2] = [0.9, 2.5];
/// Heat: seconds of use from cold to overheating (the multitool, the manipulator), seconds to cool
/// right down, and how long overheating locks the tool.
pub const HEAT_SECONDS: f32 = 7.0;
pub const MANIPULATOR_HEAT_SECONDS: f32 = 10.0;
pub const COOL_SECONDS: f32 = 4.0;
pub const LOCK_SECONDS: f32 = 3.0;
/// Uses that heat a tool keep it heating this long after each (seconds): they come ten a second
/// while the trigger is held.
pub const HEATING_HOLD: f32 = 0.15;
/// The injector: how much it heals, over how long (seconds), and what it uses up.
pub const INJECT_HEAL: i32 = 300;
pub const INJECT_SECONDS: f32 = 5.0;
pub const INJECT_COST: &str = "raw_meat";
/// Beacons are picked up from this near.
pub const BEACON_REACH: f32 = 4.0;

/// What the player does with the tool they hold.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub enum ToolUse {
    /// The trigger is held on a tool that heats (sent ten times a second while it is).
    Hold,
    /// A finished laser bite at `point` on a surface facing `normal`.
    Laser { point: Vec3, normal: Vec3 },
    /// The laser on a creature (each tenth of a second it's held there).
    Zap { id: CreatureId },
    /// A rifle shot, and a burst of flame (each tenth of a second), along `dir` from the eye.
    Shoot { dir: Vec3 },
    Flame { dir: Vec3 },
    /// The manipulator digs (or fills) a ball of `radius` at `point`.
    Terraform { point: Vec3, normal: Vec3, radius: f32, fill: bool },
    PlaceBeacon { at: Vec3 },
    TakeBeacon { id: u32 },
    Inject,
}

/// A player's tools' heat, the rifle's charge, and when each tool's overheating lock ends.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct ToolState {
    /// 0..1 by tool kind (`ToolKind::ALL` order).
    pub heat: [f32; 6],
    pub charge: f32,
    /// Ticks until each is usable again (0: it is).
    pub locked: [u32; 6],
}

impl Default for ToolState {
    fn default() -> Self {
        Self { heat: [0.0; 6], charge: RIFLE_CHARGE, locked: [0; 6] }
    }
}

fn index(kind: ToolKind) -> usize {
    ToolKind::ALL.iter().position(|k| *k == kind).unwrap_or(0)
}

impl ToolState {
    pub fn heat(&self, kind: ToolKind) -> f32 {
        self.heat[index(kind)]
    }

    pub fn locked(&self, kind: ToolKind) -> bool {
        self.locked[index(kind)] > 0
    }

    /// One step of `dt` seconds: tools in use (`heating`) warm, the rest cool, and one that reaches
    /// full heat locks; the rifle's charge refills.
    pub fn step(&mut self, dt: f32, heating: Option<ToolKind>) {
        for (i, kind) in ToolKind::ALL.into_iter().enumerate() {
            let Some(seconds) = kind.heats() else { continue };
            self.locked[i] = self.locked[i].saturating_sub(1);
            if heating == Some(kind) && self.locked[i] == 0 {
                self.heat[i] = (self.heat[i] + dt / seconds).min(1.0);
                if self.heat[i] >= 1.0 {
                    self.locked[i] = (LOCK_SECONDS * 20.0) as u32;
                }
            } else {
                self.heat[i] = (self.heat[i] - dt / COOL_SECONDS).max(0.0);
            }
        }
        self.charge = (self.charge + dt * RIFLE_REFILL).min(RIFLE_CHARGE);
    }

    /// Take a shot from the charge, if there's one.
    pub fn shoot(&mut self) -> bool {
        if self.charge < 1.0 {
            return false;
        }
        self.charge -= 1.0;
        true
    }
}

/// Whether `dir` (unit) from `from` points at `p` within `angle` (radians) and `reach`.
pub fn in_cone(from: Vec3, dir: Vec3, angle: f32, reach: f32, p: Vec3) -> bool {
    let to = p - from;
    let d = to.length();
    d < reach && (d < 0.5 || to.dot(dir) / d > angle.cos())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn held_on_it_overheats_and_locks_then_cools() {
        let mut s = ToolState::default();
        let mut ticks = 0;
        while !s.locked(ToolKind::Multitool) {
            s.step(0.05, Some(ToolKind::Multitool));
            ticks += 1;
            assert!(ticks < 1000);
        }
        assert!((ticks as f32 * 0.05 - HEAT_SECONDS).abs() < 0.2, "overheated after {} s", ticks as f32 * 0.05);
        // Locked: holding the trigger does nothing, and it cools.
        for _ in 0..(LOCK_SECONDS * 20.0) as u32 - 1 {
            s.step(0.05, Some(ToolKind::Multitool));
        }
        assert!(s.locked(ToolKind::Multitool) && s.heat(ToolKind::Multitool) < 0.4);
        s.step(0.05, None);
        assert!(!s.locked(ToolKind::Multitool));
        // The manipulator's heat is its own.
        assert_eq!(s.heat(ToolKind::Manipulator), 0.0);
    }

    #[test]
    fn the_rifle_runs_dry_and_refills() {
        let mut s = ToolState::default();
        let shots = (0..20).filter(|_| s.shoot()).count();
        assert_eq!(shots, RIFLE_CHARGE as usize);
        assert!(!s.shoot());
        for _ in 0..20 {
            s.step(0.05, None);
        }
        assert!(s.shoot(), "a shot back after a second");
    }

    #[test]
    fn the_flame_reaches_what_is_in_front() {
        let (o, d) = (Vec3::ZERO, Vec3::Z);
        assert!(in_cone(o, d, FLAME_ANGLE, FLAME_REACH, Vec3::new(0.5, 0.0, 5.0)));
        assert!(!in_cone(o, d, FLAME_ANGLE, FLAME_REACH, Vec3::new(4.0, 0.0, 5.0)));
        assert!(!in_cone(o, d, FLAME_ANGLE, FLAME_REACH, Vec3::new(0.0, 0.0, 9.0)));
        assert!(!in_cone(o, d, FLAME_ANGLE, FLAME_REACH, Vec3::new(0.0, 0.0, -3.0)));
    }
}
