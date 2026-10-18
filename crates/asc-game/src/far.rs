//! Keeps the far-terrain clipmap's height textures filled as the camera moves.
//!
//! One low-priority worker computes the samples that enter each level's window (whole steps at a
//! time) and the main thread uploads them and switches the level to its new window in the same
//! frame, so a level never draws half-updated data. Coarse levels are refreshed first so the horizon
//! appears before the detail.

use std::sync::Arc;
use std::thread;
use std::time::Instant;

use asc_render::Renderer;
use asc_render::clipmap::{self, GridRect, LEVELS, Samples};
use asc_world::WorldGen;
use crossbeam_channel::{Receiver, Sender};
use glam::{IVec2, Vec2};

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
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FarStats {
    pub in_flight: usize,
    /// Height samples computed per second of worker time, and total samples.
    pub samples_per_sec: f32,
    pub samples: u64,
}

pub struct FarStreamer {
    jobs: Sender<Job>,
    results: Receiver<Strips>,
    levels: [Level; LEVELS],
    stats: FarStats,
}

/// Heights and surface materials for every sample of `rect` on `level`, row-major. Heights are
/// computed with a one-sample border so each sample's slope comes from its grid neighbours.
pub fn compute(worldgen: &WorldGen, level: usize, rect: GridRect) -> Samples {
    let s = clipmap::spacing(level);
    let size = rect.size();
    let (w, h) = (size.x as usize + 2, size.y as usize + 2);
    let mut border = Vec::with_capacity(w * h);
    for z in rect.min.y - 1..=rect.max.y + 1 {
        for x in rect.min.x - 1..=rect.max.x + 1 {
            border.push(worldgen.height(x as f32 * s, z as f32 * s));
        }
    }
    let mut out =
        Samples { heights: Vec::with_capacity(rect.len()), materials: Vec::with_capacity(rect.len()) };
    for (j, z) in (rect.min.y..=rect.max.y).enumerate() {
        for (i, x) in (rect.min.x..=rect.max.x).enumerate() {
            let at = |di: usize, dj: usize| border[(j + dj) * w + i + di];
            let height = at(1, 1);
            let gx = (at(2, 1) - at(0, 1)) / (2.0 * s);
            let gz = (at(1, 2) - at(1, 0)) / (2.0 * s);
            let slope = (gx * gx + gz * gz).sqrt();
            out.heights.push(height);
            out.materials.push(worldgen.surface_material_at(x as f32 * s, z as f32 * s, height, slope) as u8);
        }
    }
    out
}

impl FarStreamer {
    pub fn new(worldgen: Arc<WorldGen>) -> Self {
        let (job_tx, job_rx) = crossbeam_channel::unbounded::<Job>();
        let (done_tx, done_rx) = crossbeam_channel::unbounded::<Strips>();
        thread::Builder::new()
            .name("far-terrain".into())
            .spawn(move || {
                let _ = thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Min);
                for job in job_rx {
                    let t = Instant::now();
                    let rects: Vec<(GridRect, Samples)> = clipmap::dirty(job.from, job.to)
                        .into_iter()
                        .map(|r| (r, compute(&worldgen, job.level, r)))
                        .collect();
                    let samples = rects.iter().map(|(r, _)| r.len()).sum();
                    let strips = Strips {
                        level: job.level,
                        to: job.to,
                        rects,
                        samples,
                        seconds: t.elapsed().as_secs_f32(),
                    };
                    if done_tx.send(strips).is_err() {
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
        }
    }

    pub fn stats(&self) -> FarStats {
        FarStats { in_flight: self.levels.iter().filter(|l| l.in_flight).count(), ..self.stats }
    }

    pub fn update(&mut self, camera: Vec2, renderer: &mut Renderer) {
        for strips in self.results.try_iter() {
            renderer.far_upload(strips.level, strips.to, &strips.rects);
            let level = &mut self.levels[strips.level];
            level.drawn = Some(strips.to);
            level.in_flight = false;
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
            if state.in_flight || state.drawn == Some(target) {
                continue;
            }
            state.in_flight = true;
            log::debug!("far level {level}: {:?} -> {target} (camera {camera})", state.drawn);
            let _ = self.jobs.send(Job { level, from: state.drawn, to: target });
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computed_samples_are_worldgen_heights_at_their_grid_points() {
        let wg = WorldGen::new(1337);
        let level = 3;
        let rect = GridRect { min: IVec2::new(-5, 120), max: IVec2::new(4, 130) };
        let data = compute(&wg, level, rect);
        assert_eq!(data.heights.len(), rect.len());
        assert_eq!(data.materials.len(), rect.len());
        let s = clipmap::spacing(level);
        let width = rect.size().x;
        for (i, h) in data.heights.iter().enumerate() {
            let (x, z) = (rect.min.x + i as i32 % width, rect.min.y + i as i32 / width);
            assert_eq!(*h, wg.height(x as f32 * s, z as f32 * s));
        }
    }
}
