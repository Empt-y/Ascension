//! Particle effects the client makes from what the server says: a volcano's smoke billowing out of
//! its crater (pale steam while it rumbles, dark ash clouds while it erupts), lava spraying up in
//! the crater, bombs glowing as they fly, and flames over burning ground.

use asc_render::{PARTICLE_GLOW, PARTICLE_SMOKE, Particle};
use asc_sim::volcanism::{Phase, Report};
use glam::{IVec2, Vec2, Vec3};

/// Smoke puffs a second from an erupting crater (at full strength), and lava sparks.
const SMOKE_PER_SECOND: f32 = 22.0;
const SPARKS_PER_SECOND: f32 = 60.0;
/// How long a puff lives, how fast it rises (at first, m/s) and how big it grows (radius, m).
const PUFF_LIFE: f32 = 45.0;
const PUFF_RISE: f32 = 16.0;
const PUFF_GROWTH: f32 = 1.4;
const PUFF_START: f32 = 9.0;
/// Puffs stop growing here: a plume is many billows, not a few huge blobs.
const PUFF_MAX: f32 = 38.0;
/// The wind carries the plume this way (m/s per 100 m above the crater: stronger higher up, so
/// the plume leans and spreads).
const DRIFT: Vec2 = Vec2::new(3.5, 1.2);

struct Puff {
    at: Vec3,
    /// The height it set off from (the wind grows above it).
    from: f32,
    vel: Vec3,
    size: f32,
    age: f32,
    /// 0 pale steam .. 1 dark ash.
    dark: f32,
    opacity: f32,
    seed: f32,
    spin: f32,
    /// The slowest it rises (m/s; a pyroclastic flow's cloud hugs the ground), how big it grows,
    /// and how much it glows while young (lit from below by lava).
    lift: f32,
    max: f32,
    hot: f32,
}

struct Spark {
    at: Vec3,
    vel: Vec3,
    age: f32,
    life: f32,
    size: f32,
}

/// Fire: flames a second per burning cell (catching, then burning hard), drawn within `FIRE_REACH`.
const FLAMES_PER_SECOND: [f32; 2] = [2.0, 5.0];
const FIRE_REACH: f32 = 300.0;

/// Ground-hugging cloud puffs a second per pyroclastic flow, and no more puffs in all than this
/// while they make them.
const FLOW_PUFFS_PER_SECOND: f32 = 40.0;
const FLOW_PUFFS_CAP: usize = 3000;

#[derive(Default)]
pub struct Effects {
    puffs: Vec<Puff>,
    /// Flow puffs still owed.
    owed_flow: f32,
    sparks: Vec<Spark>,
    /// Burning ground: fire cell and stage (1 catching, 2 burning hard), and the part of a flame
    /// still owed to it.
    burning: std::collections::HashMap<IVec2, (u8, f32)>,
    /// Fractions of a puff or spark still owed to each volcano (by index) at its rate.
    owed: Vec<(f32, f32)>,
    rng: u64,
}

impl Effects {
    fn random(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(0x1405_7B7E_F767_814F);
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Fire cells whose stage changed (burnt-out ones stop burning).
    pub fn set_fire(&mut self, changed: &[([i32; 2], u8)]) {
        for &(c, stage) in changed {
            let cell = IVec2::from_array(c);
            if (1..=2).contains(&stage) {
                self.burning.entry(cell).or_insert((stage, 0.0)).0 = stage;
            } else {
                self.burning.remove(&cell);
            }
        }
    }

    /// Lights for the burning ground nearest `camera` (at most `count`).
    pub fn fire_lights(&self, worldgen: &asc_world::WorldGen, camera: Vec3, count: usize) -> Vec<asc_render::PointLight> {
        let mut near: Vec<(f32, IVec2, u8)> = self
            .burning
            .iter()
            .map(|(c, (stage, _))| {
                let centre = (c.as_vec2() + 0.5) * asc_sim::fire::CELL;
                (centre.distance_squared(Vec2::new(camera.x, camera.z)), *c, *stage)
            })
            .filter(|(d, _, _)| *d < FIRE_REACH * FIRE_REACH)
            .collect();
        near.sort_by(|a, b| a.0.total_cmp(&b.0));
        near.truncate(count);
        near.into_iter()
            .map(|(_, c, stage)| {
                let centre = (c.as_vec2() + 0.5) * asc_sim::fire::CELL;
                let y = worldgen.height(centre.x, centre.y) + 1.5;
                asc_render::PointLight {
                    position: Vec3::new(centre.x, y, centre.y),
                    radius: 10.0 + 6.0 * stage as f32,
                    color: Vec3::new(1.0, 0.45, 0.12) * (1.5 + 1.5 * stage as f32),
                }
            })
            .collect()
    }

    /// Flames over the burning ground near `camera` (no smoke: up close it filled the screen).
    fn fire(&mut self, dt: f32, worldgen: &asc_world::WorldGen, camera: Vec3) {
        let cells: Vec<IVec2> = self.burning.keys().copied().collect();
        for cell in cells {
            let centre = (cell.as_vec2() + 0.5) * asc_sim::fire::CELL;
            if centre.distance_squared(Vec2::new(camera.x, camera.z)) > FIRE_REACH * FIRE_REACH {
                continue;
            }
            let (stage, owed_flame) = self.burning[&cell];
            let mut flames = owed_flame + FLAMES_PER_SECOND[stage as usize - 1] * dt;
            while flames >= 1.0 {
                flames -= 1.0;
                let at = centre + Vec2::new(self.random() - 0.5, self.random() - 0.5) * asc_sim::fire::CELL;
                let ground = worldgen.height(at.x, at.y);
                // Flames lick up to a few metres (higher through burning trees).
                let spark = Spark {
                    at: Vec3::new(at.x, ground + self.random() * 1.5 * stage as f32, at.y),
                    vel: Vec3::new(self.random() - 0.5, 2.5 + 2.5 * self.random() * stage as f32, self.random() - 0.5),
                    age: 0.0,
                    life: 0.6 + 0.7 * self.random(),
                    size: 0.5 + 0.6 * self.random() * stage as f32,
                };
                self.sparks.push(spark);
            }
            self.burning.insert(cell, (stage, flames));
        }
    }

    /// Pyroclastic flows: a churning dark cloud boiling along the ground behind each front.
    fn flows(&mut self, dt: f32, report: &Report, worldgen: &asc_world::WorldGen) {
        // While it's still moving: it slows as its front nears its length (with the last 5% to go
        // it has all but stopped, though the server keeps it a while longer).
        let flows: Vec<_> = report.flows.iter().filter(|f| f.front < f.length * 0.95).copied().collect();
        if flows.is_empty() {
            self.owed_flow = 0.0;
            return;
        }
        let rate = FLOW_PUFFS_PER_SECOND * flows.len() as f32;
        self.owed_flow = (self.owed_flow + rate * dt).min(FLOW_PUFFS_CAP.saturating_sub(self.puffs.len()) as f32);
        while self.owed_flow >= 1.0 {
            self.owed_flow -= 1.0;
            let f = flows[((self.random() * flows.len() as f32) as usize).min(flows.len() - 1)];
            // Anywhere in the deadly body behind the front, across its width.
            let along = (f.front - self.random() * 350.0).max(0.0);
            let half = f.width * 0.5 * (0.25 + 0.75 * (along / f.length).clamp(0.0, 1.0));
            let p = Vec2::new(f.origin.x, f.origin.z) + f.dir * along + f.dir.perp() * (self.random() - 0.5) * 2.0 * half;
            let ground = worldgen.height(p.x, p.y);
            let forward = f.dir * (20.0 + 15.0 * self.random());
            let size = 25.0 + 25.0 * self.random();
            let puff = Puff {
                at: Vec3::new(p.x, ground + 5.0 + self.random() * 20.0, p.y),
                from: ground,
                vel: Vec3::new(forward.x, 1.0 + 2.0 * self.random(), forward.y),
                size,
                age: 0.0,
                dark: 0.75 + 0.25 * self.random(),
                opacity: 0.35 + 0.2 * self.random(),
                seed: self.random() * 100.0,
                spin: self.random() * std::f32::consts::TAU,
                lift: 0.3,
                max: size * 1.8,
                hot: 0.35,
            };
            self.puffs.push(puff);
        }
    }

    /// Step `dt` seconds on from what the server last reported.
    pub fn update(&mut self, dt: f32, report: &Report, worldgen: &asc_world::WorldGen, camera: Vec3) {
        self.fire(dt, worldgen, camera);
        self.flows(dt, report, worldgen);
        self.owed.resize(report.active.len(), (0.0, 0.0));
        for (i, a) in report.active.iter().enumerate() {
            let (smoke, dark, sparks) = match a.phase {
                Phase::Rumbling => (0.35 + 0.4 * a.strength, 0.1, 0.0),
                Phase::Erupting => (0.6 + 0.4 * a.strength, 1.0, a.strength),
                Phase::Waning => (0.3 + 0.5 * a.strength, 0.7, 0.3 * a.strength),
                Phase::Quiet => (0.0, 0.0, 0.0),
            };
            self.owed[i].0 += smoke * SMOKE_PER_SECOND * dt;
            self.owed[i].1 += sparks * SPARKS_PER_SECOND * dt;
            // The crater's middle (its rim's height) and roughly its radius.
            let crater = (a.radius * 0.1).max(20.0);
            while self.owed[i].0 >= 1.0 {
                self.owed[i].0 -= 1.0;
                let (r, ang) = (self.random().sqrt() * crater * 0.5, self.random() * std::f32::consts::TAU);
                let at = a.summit + Vec3::new(ang.cos() * r, -10.0, ang.sin() * r);
                let rise = PUFF_RISE * (0.6 + 0.6 * self.random()) * (0.5 + 0.5 * smoke);
                let side = Vec2::from_angle(self.random() * std::f32::consts::TAU) * 2.0;
                let puff = Puff {
                    at,
                    from: at.y,
                    vel: Vec3::new(side.x, rise, side.y),
                    size: PUFF_START * (0.7 + 0.6 * self.random()),
                    age: 0.0,
                    dark: (dark * (0.8 + 0.4 * self.random())).min(1.0),
                    opacity: 0.18 + 0.12 * dark + 0.1 * self.random(),
                    seed: self.random() * 100.0,
                    spin: self.random() * std::f32::consts::TAU,
                    lift: 2.5,
                    max: PUFF_MAX,
                    hot: 1.0,
                };
                self.puffs.push(puff);
            }
            while self.owed[i].1 >= 1.0 {
                self.owed[i].1 -= 1.0;
                let ang = self.random() * std::f32::consts::TAU;
                let r = self.random().sqrt() * crater * 0.35;
                let out = 3.0 + 7.0 * self.random();
                let spark = Spark {
                    at: a.summit + Vec3::new(ang.cos() * r, -22.0, ang.sin() * r),
                    vel: Vec3::new(ang.cos() * out, 18.0 + 22.0 * self.random(), ang.sin() * out),
                    age: 0.0,
                    life: 2.5 + 2.0 * self.random(),
                    size: 0.8 + 1.6 * self.random(),
                };
                self.sparks.push(spark);
            }
        }
        for p in &mut self.puffs {
            p.age += dt;
            // Rising slows as the puff spreads; the wind carries it off.
            p.vel.y = (p.vel.y - 0.35 * dt).max(p.lift);
            let drift = DRIFT * ((p.at.y - p.from) / 100.0).clamp(0.0, 4.0);
            p.at += Vec3::new(p.vel.x + drift.x, p.vel.y, p.vel.z + drift.y) * dt;
            p.vel.x *= 1.0 - 0.4 * dt;
            p.vel.z *= 1.0 - 0.4 * dt;
            p.size = (p.size + PUFF_GROWTH * dt * (1.0 + p.age / 20.0)).min(p.max);
        }
        self.puffs.retain(|p| p.age < PUFF_LIFE);
        for s in &mut self.sparks {
            s.age += dt;
            s.vel.y -= 9.8 * dt;
            s.at += s.vel * dt;
        }
        self.sparks.retain(|s| s.age < s.life);
    }

    /// This frame's particles (plus a glow for each bomb in the air).
    pub fn particles(&self, report: &Report) -> Vec<Particle> {
        let mut out = Vec::with_capacity(self.puffs.len() + self.sparks.len() + report.bombs.len());
        for p in &self.puffs {
            let fade_in = (p.age / 0.8).min(1.0);
            let fade_out = 1.0 - ((p.age - PUFF_LIFE * 0.6) / (PUFF_LIFE * 0.4)).clamp(0.0, 1.0);
            let shade = 0.6 - 0.32 * p.dark;
            // Young puffs over an erupting crater are lit from below by the lava.
            let glow = p.hot * p.dark * (1.0 - (p.age / 8.0).min(1.0)) * 0.8;
            out.push(Particle {
                position: p.at.into(),
                size: p.size,
                color: [shade, shade * 0.97, shade * 0.93, p.opacity * fade_in * fade_out],
                params: [PARTICLE_SMOKE, glow, p.spin + p.age * 0.05, p.seed],
            });
        }
        for s in &self.sparks {
            let cool = s.age / s.life;
            let heat = 1.0 - cool * 0.6;
            out.push(Particle {
                position: s.at.into(),
                size: s.size,
                color: [3.0 * heat, 0.9 * heat * heat, 0.15 * heat * heat, 2.5 * (1.0 - cool * cool)],
                params: [PARTICLE_GLOW, 0.0, 0.0, s.age * 13.0],
            });
        }
        // A hot glow round each bomb (the rock itself is a model: see `bombs`).
        for &(b, size) in &report.bombs {
            out.push(Particle {
                position: b.into(),
                size: 1.2 + 0.7 * size as f32,
                color: [3.0, 0.8, 0.12, 1.2],
                params: [PARTICLE_GLOW, 0.0, 0.0, b.x * 0.1],
            });
        }
        out
    }

    /// The bombs in the air as tumbling models (`models`: one per size, smallest first).
    pub fn bombs(&self, report: &Report, models: &[asc_render::ModelId], time: f32) -> Vec<asc_render::PartInstance> {
        report
            .bombs
            .iter()
            .filter_map(|&(at, size)| {
                let model = *models.get(size as usize)?;
                // Each spins at its own rate (seeded by where it is, which is smooth in time).
                let spin = 1.5 + (at.x * 0.37 + at.z * 0.23).sin().abs() * 3.0;
                Some(asc_render::PartInstance { model, position: at, yaw: time * spin + at.x, scale: 1.0, sway: 0.0 })
            })
            .collect()
    }

    /// How deep in the smoke `at` is (0..1): ash in the air only there.
    pub fn smoke_at(&self, at: Vec3) -> f32 {
        let mut ash: f32 = 0.0;
        for p in &self.puffs {
            let d = p.at.distance(at) / p.size;
            if d < 1.0 {
                ash = ash.max((1.0 - d) * p.dark * 2.0);
            }
        }
        ash.min(1.0)
    }
}
