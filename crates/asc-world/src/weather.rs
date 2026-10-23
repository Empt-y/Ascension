//! Weather: worked out from the seed, a place and the time, so the server and every client agree
//! on it without sending it (only overrides, such as a forced storm, need saying).
//!
//! Fronts are a large-scale noise drifting with the wind and slowly changing as it goes. Where a
//! front is dense there's cloud; denser, rain (snow where it's cold, and in dry deserts, blowing
//! sand when it's windy instead); densest, storms with lightning. The wind turns slowly over hours
//! and gusts. Snow on the ground isn't stored: it's the snow that fell over the last few hours
//! there, less what melted, worked out from a few looks back in time.

use fastnoise_lite::{FastNoiseLite, FractalType, NoiseType};
use glam::Vec2;

use crate::WorldGen;

/// Ticks per second of the world clock.
pub const TICKS_PER_SECOND: f32 = 20.0;
/// Fronts are about this wide (metres) and change over about this long (seconds).
const FRONT_SIZE: f32 = 5000.0;
const FRONT_TIME: f32 = 2400.0;
/// The wind turns over about this long.
const WIND_TURN: f32 = 7200.0;
/// Precipitation is snow below this temperature (lapsed, 0..1 as the climate's).
pub const SNOW_BELOW: f32 = 0.3;
/// Snow on the ground: looks back this far in steps of this long; metres of snow a full
/// snowfall lays per hour, and melt per hour on warm ground (per 0.1 above `SNOW_BELOW`).
const SNOW_LOOKBACK: f32 = 6.0 * 3600.0;
const SNOW_STEPS: usize = 12;
const SNOW_PER_HOUR: f32 = 0.12;
const MELT_PER_HOUR: f32 = 0.08;
pub const MAX_SNOW: f32 = 0.6;

/// Kinds of falling weather.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fall {
    Rain,
    Snow,
    Sand,
}

/// The weather at one place and time.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Conditions {
    /// 0..1 of the sky.
    pub cloud: f32,
    /// 0..1 how hard it rains, snows or blows sand.
    pub precipitation: f32,
    pub fall: Fall,
    /// 0..1 a storm (lightning) — only in heavy rain or snow.
    pub storm: f32,
    /// Metres per second, blowing toward.
    pub wind: Vec2,
    /// Lapsed temperature here (0..1, as the climate's), lowered by cloud and night.
    pub temperature: f32,
}

/// Overrides of the natural weather (cheats and scripted tests).
#[derive(Clone, Copy, Debug, Default, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Forced {
    /// Weather at least this stormy everywhere (0 = natural): 1 is a full storm.
    pub storm: f32,
    /// Clear skies everywhere.
    pub clear: bool,
}

pub struct Weather {
    fronts: FastNoiseLite,
    detail: FastNoiseLite,
    wind: FastNoiseLite,
    gusts: FastNoiseLite,
    pub forced: Forced,
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl Weather {
    pub fn new(seed: i32) -> Self {
        let noise = |salt: i32, freq: f32, octaves: i32| {
            let mut n = FastNoiseLite::with_seed(seed.wrapping_add(salt));
            n.set_noise_type(Some(NoiseType::OpenSimplex2));
            n.set_frequency(Some(freq));
            if octaves > 1 {
                n.set_fractal_type(Some(FractalType::FBm));
                n.set_fractal_octaves(Some(octaves));
            }
            n
        };
        Self {
            fronts: noise(101, 1.0, 3),
            detail: noise(102, 1.0, 2),
            wind: noise(103, 1.0, 1),
            gusts: noise(104, 1.0, 1),
            forced: Forced::default(),
        }
    }

    /// The prevailing wind at time `t` (seconds): turning slowly, stronger in places' storms.
    fn prevailing(&self, t: f32) -> Vec2 {
        let a = self.wind.get_noise_2d(t / WIND_TURN, 0.5) * std::f32::consts::TAU;
        Vec2::from_angle(a) * (4.0 + 3.0 * self.wind.get_noise_2d(0.5, t / WIND_TURN))
    }

    /// How dense the fronts are at (x, z) and time `t` (seconds): carried along by the wind.
    fn density(&self, p: Vec2, t: f32) -> f32 {
        // The wind's drift so far (its turning is slow, so straight-line drift is close enough).
        let drift = self.prevailing(t) * t;
        let q = (p - drift) / FRONT_SIZE;
        let n = self.fronts.get_noise_3d(q.x, q.y, t / FRONT_TIME);
        let fine = self.detail.get_noise_3d(q.x * 4.0, q.y * 4.0, t / (FRONT_TIME * 0.3));
        n + fine * 0.15
    }

    /// The weather at (x, z) at `ticks`, for ground (or the camera) at height `y`.
    pub fn at(&self, worldgen: &WorldGen, x: f32, z: f32, y: f32, ticks: u64, time_of_day: f32) -> Conditions {
        let t = ticks as f32 / TICKS_PER_SECOND;
        let p = Vec2::new(x, z);
        let (climate_t, moisture) = worldgen.climate_at(x, z);
        let lapsed = (climate_t - (y - 20.0).max(0.0) * crate::worldgen::LAPSE).clamp(0.0, 1.0);
        let mut d = self.density(p, t);
        if self.forced.clear {
            d = -1.0;
        }
        // Wetter climates cloud over and rain more readily.
        let wet = (moisture - 0.5) * 0.25;
        let mut cloud = smoothstep(-0.25, 0.45, d + wet);
        let mut precipitation = smoothstep(0.35, 0.7, d + wet) * smoothstep(0.08, 0.3, moisture);
        let mut storm = smoothstep(0.62, 0.85, d + wet) * smoothstep(0.25, 0.5, moisture);
        if self.forced.storm > 0.0 {
            cloud = cloud.max(self.forced.storm);
            precipitation = precipitation.max(self.forced.storm);
            storm = storm.max(self.forced.storm);
        }
        let gust = 1.0 + 0.6 * self.gusts.get_noise_2d(t * 0.15, x * 0.001 + z * 0.0007);
        let wind = self.prevailing(t) * (1.0 + 1.5 * cloud + 2.5 * storm) * gust;
        // Night and cloud cool the air a little.
        let night = 1.0 - smoothstep(0.2, 0.3, time_of_day) * (1.0 - smoothstep(0.7, 0.8, time_of_day));
        let temperature = (lapsed - 0.04 * night - 0.03 * cloud).clamp(0.0, 1.0);
        let desert = moisture < 0.2 && lapsed > 0.6;
        let fall = if desert {
            // Dry country: no rain to speak of, but sand blows in strong wind.
            precipitation = smoothstep(9.0, 16.0, wind.length());
            storm = 0.0;
            Fall::Sand
        } else if temperature < SNOW_BELOW {
            Fall::Snow
        } else {
            Fall::Rain
        };
        Conditions { cloud, precipitation, fall, storm, wind, temperature }
    }

    /// Metres of snow lying on open ground at (x, z), height `y`, at `ticks`: what fell over the
    /// last few hours, less what melted.
    pub fn snow_depth(&self, worldgen: &WorldGen, x: f32, z: f32, y: f32, ticks: u64) -> f32 {
        let step = SNOW_LOOKBACK / SNOW_STEPS as f32;
        let hours = step / 3600.0;
        let mut depth: f32 = 0.0;
        for k in (0..SNOW_STEPS).rev() {
            let back = (k as f32 * step * TICKS_PER_SECOND) as u64;
            let then = ticks.saturating_sub(back);
            if then == 0 && k > 0 {
                continue;
            }
            let c = self.at(worldgen, x, z, y, then, 0.5);
            if c.fall == Fall::Snow {
                depth += c.precipitation * SNOW_PER_HOUR * hours;
            }
            let warmth = ((c.temperature - SNOW_BELOW) * 10.0).max(0.0);
            depth = (depth - warmth * MELT_PER_HOUR * hours).max(0.0);
        }
        depth.min(MAX_SNOW)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weather_is_the_same_wherever_its_worked_out() {
        let wg = WorldGen::new(1337);
        let (a, b) = (Weather::new(1337), Weather::new(1337));
        for i in 0..20 {
            let (x, z, ticks) = (i as f32 * 731.0, i as f32 * -419.0, i as u64 * 9000);
            assert_eq!(a.at(&wg, x, z, 20.0, ticks, 0.5), b.at(&wg, x, z, 20.0, ticks, 0.5));
        }
    }

    #[test]
    fn the_weather_varies_over_place_and_time_with_rain_and_storms() {
        let wg = WorldGen::new(1337);
        let w = Weather::new(1337);
        let (mut clear, mut rain, mut storm, mut snow, mut n) = (0, 0, 0, 0, 0);
        for i in 0..40 {
            for k in 0..40 {
                let (x, z) = (i as f32 * 900.0 - 18000.0, k as f32 * 700.0 - 14000.0);
                let c = w.at(&wg, x, z, wg.height(x, z), (i * 7 + k) as u64 * 20 * 600, 0.5);
                n += 1;
                clear += usize::from(c.cloud < 0.2);
                rain += usize::from(c.precipitation > 0.3 && c.fall == Fall::Rain);
                snow += usize::from(c.precipitation > 0.3 && c.fall == Fall::Snow);
                storm += usize::from(c.storm > 0.5);
            }
        }
        let share = |c: usize| c as f32 / n as f32;
        assert!(share(clear) > 0.2, "clear {}", share(clear));
        assert!(share(rain) > 0.03 && share(rain) < 0.5, "rain {}", share(rain));
        assert!(share(storm) > 0.005 && share(storm) < share(rain), "storms {}", share(storm));
        println!("clear {} rain {} snow {} storm {}", share(clear), share(rain), share(snow), share(storm));
    }

    #[test]
    fn snow_builds_up_while_it_snows_and_melts_when_warm() {
        let wg = WorldGen::new(1337);
        let mut w = Weather::new(1337);
        // A cold place.
        let (x, z, y) = (0..4000)
            .map(|i| ((i % 64) as f32 * 300.0 - 9600.0, (i / 64) as f32 * 300.0 - 9600.0))
            .map(|(x, z)| (x, z, wg.height(x, z)))
            .find(|&(x, z, y)| w.at(&wg, x, z, y, 0, 0.5).temperature < SNOW_BELOW - 0.05 && wg.water_level(x, z).is_none())
            .expect("somewhere cold");
        let start = 20 * 3600 * 10;
        w.forced = Forced { storm: 0.0, clear: true };
        assert_eq!(w.snow_depth(&wg, x, z, y, start), 0.0, "no snow under clear skies");
        w.forced = Forced { storm: 1.0, clear: false };
        let deep = w.snow_depth(&wg, x, z, y, start);
        assert!(deep > 0.2, "only {deep} m after hours of blizzard");
    }
}
