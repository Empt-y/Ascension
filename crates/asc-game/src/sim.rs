//! The fixed-rate simulation. Runs at 20 ticks per second regardless of frame rate; machines, networks
//! and everything else that must be deterministic will live here. For now it only keeps time.

use glam::Vec3;

/// Ticks in one in-game day: 20 real minutes, as in Minecraft.
pub const DAY_TICKS: u64 = 24_000;

pub struct Sim {
    pub ticks: u64,
    /// 0 = midnight, 0.25 = sunrise, 0.5 = noon, 0.75 = sunset.
    pub time_of_day: f32,
}

impl Default for Sim {
    fn default() -> Self {
        Self { ticks: 0, time_of_day: 0.34 }
    }
}

impl Sim {
    pub fn tick(&mut self) {
        self.ticks += 1;
        self.time_of_day = (self.time_of_day + 1.0 / DAY_TICKS as f32).rem_euclid(1.0);
    }

    /// Unit vector towards the sun. It rises in the east (+X), sets in the west, tilted to the south.
    pub fn sun_dir(&self) -> Vec3 {
        let angle = (self.time_of_day - 0.25) * std::f32::consts::TAU;
        Vec3::new(angle.cos(), angle.sin(), 0.35).normalize()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sun_is_up_at_noon_and_down_at_midnight() {
        let noon = Sim { ticks: 0, time_of_day: 0.5 };
        let midnight = Sim { ticks: 0, time_of_day: 0.0 };
        assert!(noon.sun_dir().y > 0.9);
        assert!(midnight.sun_dir().y < -0.9);
    }

    #[test]
    fn a_day_wraps_after_day_ticks() {
        let mut sim = Sim { ticks: 0, time_of_day: 0.5 };
        for _ in 0..DAY_TICKS {
            sim.tick();
        }
        assert!((sim.time_of_day - 0.5).abs() < 1e-3);
    }
}
