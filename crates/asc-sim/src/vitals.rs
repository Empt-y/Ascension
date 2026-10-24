//! A player's health, air and body temperature, and what hurts them.
//!
//! Integers throughout (the simulation's rule): health in points (`MAX_HEALTH` = 20 hearts of 50),
//! air in ticks, temperature in hundredths of a degree Celsius. The server owns these; the client
//! only shows them.

use serde::{Deserialize, Serialize};

pub const MAX_HEALTH: i32 = 1000;
/// Air lasts this many ticks underwater (15 s at 20 Hz).
pub const MAX_AIR: i32 = 300;
/// Normal body temperature, hundredths of a degree.
pub const BODY_TEMPERATURE: i32 = 3700;
/// Falls up to this far are free; each metre beyond costs `FALL_DAMAGE` (so about 24 m kills).
pub const SAFE_FALL: f32 = 4.0;
pub const FALL_DAMAGE: f32 = 50.0;
/// Health comes back this fast once nothing has hurt the player for `REGEN_DELAY` ticks.
const REGEN_PER_TICK: i32 = 2;
const REGEN_DELAY: u32 = 100;
/// Drowning: this much damage every `DROWN_INTERVAL` ticks once the air has run out.
const DROWN_DAMAGE: i32 = 50;
const DROWN_INTERVAL: u32 = 20;

/// What hurt a player.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Cause {
    Fall,
    Drowning,
    Lava,
    Cold,
    Heat,
    Lightning,
    Eruption,
    /// Fell out of the world.
    Void,
    /// Bitten, butted or kicked by a creature of this kind (`creatures::kinds()`).
    Animal(u16),
}

impl Cause {
    pub fn describe(self) -> &'static str {
        match self {
            Self::Fall => "fell too far",
            Self::Drowning => "drowned",
            Self::Lava => "burned in lava",
            Self::Cold => "froze",
            Self::Heat => "overheated",
            Self::Lightning => "was struck by lightning",
            Self::Eruption => "was caught in an eruption",
            Self::Void => "fell out of the world",
            Self::Animal(k) => crate::creatures::kinds().get(k as usize).map_or("was killed by an animal", |k| k.killed.as_str()),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vitals {
    pub health: i32,
    pub air: i32,
    pub temperature: i32,
    /// Ticks since anything hurt the player.
    #[serde(default)]
    pub calm: u32,
    /// Ticks spent without air.
    #[serde(default)]
    pub drowning: u32,
}

impl Default for Vitals {
    fn default() -> Self {
        Self { health: MAX_HEALTH, air: MAX_AIR, temperature: BODY_TEMPERATURE, calm: REGEN_DELAY, drowning: 0 }
    }
}

/// What a tick or a hit did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    Fine,
    Hurt(i32, Cause),
    Died(Cause),
}

impl Vitals {
    pub fn alive(&self) -> bool {
        self.health > 0
    }

    /// Take `amount` of damage.
    pub fn hurt(&mut self, amount: i32, cause: Cause) -> Outcome {
        if amount <= 0 || !self.alive() {
            return Outcome::Fine;
        }
        self.health = (self.health - amount).max(0);
        self.calm = 0;
        if self.alive() { Outcome::Hurt(amount, cause) } else { Outcome::Died(cause) }
    }

    /// Damage for landing after falling `height` metres.
    pub fn fall_damage(height: f32) -> i32 {
        ((height - SAFE_FALL).max(0.0) * FALL_DAMAGE) as i32
    }

    /// One tick: air runs down underwater (then drowning starts) and refills above; health comes
    /// back after a while unhurt.
    pub fn tick(&mut self, underwater: bool) -> Outcome {
        if !self.alive() {
            return Outcome::Fine;
        }
        let mut outcome = Outcome::Fine;
        if underwater {
            if self.air > 0 {
                self.air -= 1;
            } else {
                self.drowning += 1;
                if self.drowning.is_multiple_of(DROWN_INTERVAL) {
                    outcome = self.hurt(DROWN_DAMAGE, Cause::Drowning);
                }
            }
        } else {
            self.air = (self.air + 5).min(MAX_AIR);
            self.drowning = 0;
        }
        self.calm = self.calm.saturating_add(1);
        if self.calm >= REGEN_DELAY && self.alive() {
            self.health = (self.health + REGEN_PER_TICK).min(MAX_HEALTH);
        }
        outcome
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_falls_are_free_and_long_ones_kill() {
        assert_eq!(Vitals::fall_damage(3.5), 0);
        assert_eq!(Vitals::fall_damage(6.0), 100);
        let mut v = Vitals::default();
        assert_eq!(v.hurt(Vitals::fall_damage(12.0), Cause::Fall), Outcome::Hurt(400, Cause::Fall));
        assert_eq!(v.hurt(Vitals::fall_damage(30.0), Cause::Fall), Outcome::Died(Cause::Fall));
        assert_eq!(v.health, 0);
        assert_eq!(v.hurt(100, Cause::Fall), Outcome::Fine, "the dead can't be hurt again");
    }

    #[test]
    fn health_comes_back_after_a_while_unhurt() {
        let mut v = Vitals::default();
        v.hurt(500, Cause::Fall);
        for _ in 0..REGEN_DELAY - 1 {
            v.tick(false);
        }
        assert_eq!(v.health, 500, "not yet");
        for _ in 0..300 {
            v.tick(false);
        }
        assert!(v.health > 500 && v.health <= MAX_HEALTH);
    }

    #[test]
    fn air_runs_out_then_drowning_hurts() {
        let mut v = Vitals::default();
        let mut hurt = 0;
        for _ in 0..(MAX_AIR as u32 + DROWN_INTERVAL * 3) {
            if let Outcome::Hurt(n, Cause::Drowning) = v.tick(true) {
                hurt += n;
            }
        }
        assert_eq!(v.air, 0);
        assert_eq!(hurt, DROWN_DAMAGE * 3);
        for _ in 0..100 {
            v.tick(false);
        }
        assert_eq!(v.air, MAX_AIR, "air refills above water");
    }
}
