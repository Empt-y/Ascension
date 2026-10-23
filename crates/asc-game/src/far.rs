//! Keeps the far-terrain clipmap's height textures filled as the camera moves.
//!
//! One low-priority worker computes the samples that enter each level's window (whole steps at a
//! time) and the main thread uploads them and switches the level to its new window in the same
//! frame, so a level never draws half-updated data. Coarse levels are refreshed first so the horizon
//! appears before the detail.
//!
//! The far terrain is worldgen's, except where something huge has been dug out of it (a volcano
//! blowing its flank away): digs at least `BIG_DIG` across are cut out of it too. Smaller edits
//! only show in the chunks near by.

use std::sync::{Arc, RwLock};
use std::thread;
use std::time::Instant;

use asc_render::Renderer;
use asc_render::clipmap::{self, GridRect, LEVELS, Samples};
use asc_world::WorldGen;
use crossbeam_channel::{Receiver, Sender};
use glam::{IVec2, Vec2};

/// Digs of at least this radius (m) are cut out of the far terrain.
const BIG_DIG: f32 = 25.0;

struct Job {
    level: usize,
    from: Option<IVec2>,
    to: IVec2,
}

pub struct Strips {
    pub level: usize,
    pub to: IVec2,
    pub rects: Vec<(GridRect, Samples)>,
    samples: usize,
    seconds: f32,
}

#[derive(Clone, Copy, Default)]
struct Level {
    drawn: Option<IVec2>,
    in_flight: bool,
    /// Which set of big digs it was drawn with (see `FarStreamer::digs_version`).
    digs: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FarStats {
    pub in_flight: usize,
    /// Height samples computed per second of worker time, and total samples.
    pub samples_per_sec: f32,
    pub samples: u64,
}

pub struct FarStreamer {
    jobs: Sender<(Job, u32)>,
    results: Receiver<(Strips, u32)>,
    levels: [Level; LEVELS],
    stats: FarStats,
    /// The big digs (shared with the worker), and a count of changes to them.
    digs: Arc<RwLock<Vec<asc_world::EditOp>>>,
    digs_version: u32,
}

/// Where dug spheres have lowered the ground at `(x, z)` from `height`, the new height. A sphere
/// cuts the surface when the surface is inside it, so one under the ground can be reached through
/// another's hole: it goes round again until nothing cuts deeper (in any order of digs).
fn dug(digs: &[asc_world::EditOp], x: f32, z: f32, height: f32) -> Option<f32> {
    let mut lowest: Option<f32> = None;
    loop {
        let mut deeper = false;
        for op in digs {
            let d2 = (x - op.center.x).powi(2) + (z - op.center.z).powi(2);
            let r2 = op.radius * op.radius;
            if d2 >= r2 {
                continue;
            }
            let half = (r2 - d2).sqrt();
            let (bottom, top) = (op.center.y - half, op.center.y + half);
            let h = lowest.unwrap_or(height);
            if h > bottom && h <= top {
                lowest = Some(bottom);
                deeper = true;
            }
        }
        if !deeper {
            return lowest;
        }
    }
}

/// Heights and surface materials for every sample of `rect` on `level`, row-major. Heights are
/// computed with a one-sample border so each sample's slope comes from its grid neighbours.
pub fn compute(worldgen: &WorldGen, digs: &[asc_world::EditOp], level: usize, rect: GridRect) -> Samples {
    let s = clipmap::spacing(level);
    let size = rect.size();
    let (w, h) = (size.x as usize + 2, size.y as usize + 2);
    let mut border = Vec::with_capacity(w * h);
    let mut water = Vec::with_capacity(w * h);
    let mut ice = Vec::with_capacity(w * h);
    for z in rect.min.y - 1..=rect.max.y + 1 {
        for x in rect.min.x - 1..=rect.max.x + 1 {
            let (height, level, lava, ice_top) = worldgen.height_water_lava(x as f32 * s, z as f32 * s);
            // Blown away: bare ground, its lake (or lava) gone with it.
            if let Some(lowered) = dug(digs, x as f32 * s, z as f32 * s, height) {
                border.push(lowered);
                ice.push(None);
                water.push(None);
                continue;
            }
            border.push(height);
            ice.push(ice_top);
            // Lava is drawn by the far water too (flagged), in its place.
            water.push(lava.map(|l| (l, true)).or(level.map(|l| (l, false))));
        }
    }
    let mut out = Samples {
        heights: Vec::with_capacity(rect.len()),
        materials: Vec::with_capacity(rect.len()),
        water: Vec::with_capacity(rect.len()),
    };
    for (j, z) in (rect.min.y..=rect.max.y).enumerate() {
        for (i, x) in (rect.min.x..=rect.max.x).enumerate() {
            let at = |di: usize, dj: usize| border[(j + dj) * w + i + di];
            let height = at(1, 1);
            let gx = (at(2, 1) - at(0, 1)) / (2.0 * s);
            let gz = (at(1, 2) - at(1, 0)) / (2.0 * s);
            let slope = (gx * gx + gz * gz).sqrt();
            let (level, lava) = match water[(j + 1) * w + i + 1] {
                Some((l, lava)) => (Some(l), lava),
                None => (None, false),
            };
            out.heights.push(height);
            let (fx, fz) = (x as f32 * s, z as f32 * s);
            let flow = if level.is_some() && !lava { worldgen.water_flow(fx, fz) } else { glam::Vec2::ZERO };
            out.water.push([level.unwrap_or(asc_render::clipmap::NO_WATER), flow.x, flow.y, if lava { 1.0 } else { 0.0 }]);
            // Glaciers' snow and ice, else the ground's own surface.
            let material = ice[(j + 1) * w + i + 1].unwrap_or_else(|| worldgen.surface_material_at(fx, fz, height, slope, level));
            out.materials.push(material as u8);
        }
    }
    out
}

impl FarStreamer {
    pub fn new(worldgen: Arc<WorldGen>, edits: &[asc_world::EditOp]) -> Self {
        let (job_tx, job_rx) = crossbeam_channel::unbounded::<(Job, u32)>();
        let (done_tx, done_rx) = crossbeam_channel::unbounded::<(Strips, u32)>();
        let digs = Arc::new(RwLock::new(edits.iter().copied().filter(is_big_dig).collect::<Vec<_>>()));
        let worker_digs = digs.clone();
        thread::Builder::new()
            .name("far-terrain".into())
            .spawn(move || {
                let _ = thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Min);
                for (job, version) in job_rx {
                    let t = Instant::now();
                    let digs = worker_digs.read().expect("far digs").clone();
                    let rects: Vec<(GridRect, Samples)> = clipmap::dirty(job.from, job.to)
                        .into_iter()
                        .map(|r| (r, compute(&worldgen, &digs, job.level, r)))
                        .collect();
                    let samples = rects.iter().map(|(r, _)| r.len()).sum();
                    let strips = Strips {
                        level: job.level,
                        to: job.to,
                        rects,
                        samples,
                        seconds: t.elapsed().as_secs_f32(),
                    };
                    if done_tx.send((strips, version)).is_err() {
                        break;
                    }
                }
            })
            .expect("spawning far-terrain worker");
        Self {
            jobs: job_tx,
            results: done_rx,
            levels: [Level::default(); LEVELS],
            stats: FarStats::default(),
            digs,
            digs_version: 0,
        }
    }

    /// An edit has been made: a big dig is cut out of the far terrain (every level is redrawn).
    pub fn edited(&mut self, op: &asc_world::EditOp) {
        if is_big_dig(op) {
            self.digs.write().expect("far digs").push(*op);
            self.digs_version += 1;
        }
    }

    pub fn stats(&self) -> FarStats {
        FarStats { in_flight: self.levels.iter().filter(|l| l.in_flight).count(), ..self.stats }
    }

    pub fn update(&mut self, camera: Vec2, renderer: &mut Renderer) {
        for (strips, version) in self.results.try_iter() {
            renderer.far_upload(strips.level, strips.to, &strips.rects);
            let level = &mut self.levels[strips.level];
            level.drawn = Some(strips.to);
            level.in_flight = false;
            level.digs = version;
            if strips.seconds > 0.0 {
                let rate = strips.samples as f32 / strips.seconds;
                self.stats.samples_per_sec += (rate - self.stats.samples_per_sec) * 0.2;
            }
            self.stats.samples += strips.samples as u64;
        }
        // Coarse levels first: the horizon matters more than the detail just past the chunks.
        for level in (0..LEVELS).rev() {
            let state = &mut self.levels[level];
            let target = clipmap::keep_or_recenter(level, camera, state.drawn);
            let stale = state.digs != self.digs_version;
            if state.in_flight || (state.drawn == Some(target) && !stale) {
                continue;
            }
            state.in_flight = true;
            // New digs: the whole window again.
            let from = if stale { None } else { state.drawn };
            log::debug!("far level {level}: {from:?} -> {target} (camera {camera})");
            let _ = self.jobs.send((Job { level, from, to: target }, self.digs_version));
        }
    }
}

fn is_big_dig(op: &asc_world::EditOp) -> bool {
    op.kind == asc_world::EditKind::Dig && op.radius >= BIG_DIG
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computed_samples_are_worldgen_heights_at_their_grid_points() {
        let wg = WorldGen::new(1337);
        let level = 3;
        let rect = GridRect { min: IVec2::new(-5, 120), max: IVec2::new(4, 130) };
        let data = compute(&wg, &[], level, rect);
        assert_eq!(data.heights.len(), rect.len());
        assert_eq!(data.materials.len(), rect.len());
        let s = clipmap::spacing(level);
        let width = rect.size().x;
        for (i, h) in data.heights.iter().enumerate() {
            let (x, z) = (rect.min.x + i as i32 % width, rect.min.y + i as i32 / width);
            assert_eq!(*h, wg.height(x as f32 * s, z as f32 * s));
        }
    }

    #[test]
    fn a_big_dig_lowers_the_far_terrain_and_a_small_one_doesnt() {
        let wg = WorldGen::new(1337);
        let level = 2;
        let s = clipmap::spacing(level);
        let rect = GridRect { min: IVec2::new(0, 0), max: IVec2::new(9, 9) };
        let (x, z) = (5.0 * s, 5.0 * s);
        let ground = wg.height(x, z);
        let dig = |radius| asc_world::EditOp {
            center: glam::Vec3::new(x, ground, z),
            radius,
            kind: asc_world::EditKind::Dig,
            material: asc_world::Material::Air,
        };
        let big = dig(200.0);
        assert!(is_big_dig(&big) && !is_big_dig(&dig(3.0)));
        let data = compute(&wg, &[big], level, rect);
        let i = (5 * rect.size().x + 5) as usize;
        assert!((data.heights[i] - (ground - 200.0)).abs() < 0.01, "{} vs {}", data.heights[i], ground - 200.0);
        // A sphere wholly under the ground is reached through the big one's hole, listed either way.
        let under = asc_world::EditOp { center: glam::Vec3::new(x, ground - 260.0, z), ..dig(100.0) };
        for digs in [[under, big], [big, under]] {
            assert_eq!(dug(&digs, x, z, ground), Some(ground - 360.0));
        }
    }
}
