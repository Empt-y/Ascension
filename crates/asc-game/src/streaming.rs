//! Keeps the chunks around the camera generated, meshed and uploaded.
//!
//! The main thread decides what is wanted and in which order; a pool of worker threads builds chunks
//! (worldgen + replayed edits + meshing) and sends meshes back. Only a few jobs per worker are in
//! flight at once so priorities stay fresh as the camera moves. Every request carries a version; edits
//! bump the version, and a result is uploaded only if it is the newest one seen for that chunk.

use std::collections::VecDeque;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ahash::{AHashMap, AHashSet};
use asc_render::{Frustum, Renderer};
use asc_world::{CHUNK, ChunkCoord, ChunkData, ChunkMesh, EditOp, World, WorldGen, chunk_of, mesh_chunk};
use crossbeam_channel::{Receiver, Sender};
use glam::{IVec2, IVec3, Vec3};

const JOBS_PER_WORKER: usize = 3;
/// Chunks are dropped this many chunks beyond the view radius, so small moves don't thrash.
const UNLOAD_MARGIN: i32 = 2;
/// Vertical chunks kept around the camera regardless of terrain (caves, flying high).
const CAMERA_Y_REACH: i32 = 1;
/// Upload budget per frame.
const MAX_UPLOADS_PER_FRAME: usize = 48;

struct Job {
    coord: ChunkCoord,
    version: u32,
    ops: Vec<EditOp>,
}

struct Built {
    coord: ChunkCoord,
    version: u32,
    mesh: ChunkMesh,
    build_time: Duration,
    mesh_time: Duration,
}

#[derive(Default)]
struct Slot {
    /// Latest version requested.
    wanted: u32,
    /// Version currently on the GPU (or known empty), if any.
    shown: Option<u32>,
    /// Jobs for this chunk currently with workers.
    in_flight: u32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct StreamStats {
    pub loaded: usize,
    pub in_flight: usize,
    pub queued: usize,
    pub avg_build_ms: f32,
    pub avg_mesh_ms: f32,
    /// Time from the most recent edit until all chunks it touched were back on screen.
    pub last_edit_ms: Option<f32>,
}

pub struct Streamer {
    jobs: Sender<Job>,
    results: Receiver<Built>,
    workers: usize,
    slots: AHashMap<ChunkCoord, Slot>,
    /// Chunks waiting for a worker, best first. Rebuilt when the camera changes chunk.
    queue: VecDeque<ChunkCoord>,
    /// Edited chunks jump the queue.
    urgent: VecDeque<ChunkCoord>,
    in_flight: usize,
    uploads: VecDeque<Built>,
    column_range: AHashMap<IVec2, (i32, i32)>,
    worldgen: Arc<WorldGen>,
    pub view_radius: i32,
    last_center: Option<IVec3>,
    last_radius: i32,
    stats: StreamStats,
    pending_edit: Option<(Instant, AHashSet<ChunkCoord>)>,
}

impl Streamer {
    pub fn new(worldgen: Arc<WorldGen>, view_radius: i32) -> Self {
        let workers = thread::available_parallelism().map_or(4, |n| n.get()).saturating_sub(2).max(1);
        let (job_tx, job_rx) = crossbeam_channel::unbounded::<Job>();
        let (done_tx, done_rx) = crossbeam_channel::unbounded::<Built>();
        for i in 0..workers {
            let (rx, tx, wg) = (job_rx.clone(), done_tx.clone(), worldgen.clone());
            thread::Builder::new()
                .name(format!("chunk-worker-{i}"))
                .spawn(move || {
                    for job in rx {
                        let t0 = Instant::now();
                        let data = ChunkData::build(&wg, job.coord, &job.ops);
                        let t1 = Instant::now();
                        let mesh = mesh_chunk(&data);
                        let built = Built {
                            coord: job.coord,
                            version: job.version,
                            mesh,
                            build_time: t1 - t0,
                            mesh_time: t1.elapsed(),
                        };
                        if tx.send(built).is_err() {
                            break;
                        }
                    }
                })
                .expect("spawning chunk worker");
        }
        log::info!("chunk streaming with {workers} workers");
        Self {
            jobs: job_tx,
            results: done_rx,
            workers,
            slots: AHashMap::new(),
            queue: VecDeque::new(),
            urgent: VecDeque::new(),
            in_flight: 0,
            uploads: VecDeque::new(),
            column_range: AHashMap::new(),
            worldgen,
            view_radius,
            last_center: None,
            last_radius: view_radius,
            stats: StreamStats::default(),
            pending_edit: None,
        }
    }

    pub fn stats(&self) -> StreamStats {
        StreamStats {
            loaded: self.slots.len(),
            in_flight: self.in_flight,
            queued: self.queue.len() + self.urgent.len(),
            ..self.stats
        }
    }

    /// Chunk-y range that can hold the terrain surface in a column (with margin for overhangs).
    fn surface_band(&mut self, col: IVec2) -> (i32, i32) {
        *self.column_range.entry(col).or_insert_with(|| {
            let base = (col * CHUNK).as_vec2();
            let s = CHUNK as f32;
            let (mut lo, mut hi) = (f32::MAX, f32::MIN);
            for (dx, dz) in [(0.0, 0.0), (s, 0.0), (0.0, s), (s, s), (s * 0.5, s * 0.5)] {
                let h = self.worldgen.height(base.x + dx, base.y + dz);
                lo = lo.min(h);
                hi = hi.max(h);
            }
            // Slopes inside the column can exceed the sampled corners; caves can open just below.
            (chunk_of(Vec3::splat(lo - 18.0)).y, chunk_of(Vec3::splat(hi + 12.0)).y)
        })
    }

    /// Mark chunks touched by an edit for an immediate rebuild.
    pub fn invalidate(&mut self, coords: &[ChunkCoord]) {
        let mut waiting = AHashSet::new();
        for c in coords {
            if let Some(slot) = self.slots.get_mut(c) {
                slot.wanted += 1;
                self.urgent.push_back(*c);
                waiting.insert(*c);
            }
        }
        if !waiting.is_empty() {
            self.pending_edit = Some((Instant::now(), waiting));
        }
    }

    pub fn update(&mut self, world: &World, camera: Vec3, frustum: &Frustum, renderer: &mut Renderer) {
        let center = chunk_of(camera);
        if self.last_center != Some(center) || self.last_radius != self.view_radius {
            self.retarget(center, camera, frustum, renderer);
            self.last_center = Some(center);
            self.last_radius = self.view_radius;
        }
        self.dispatch(world);
        self.receive(renderer);
    }

    /// Recompute which chunks are wanted and in what order; drop the ones that fell out of range.
    fn retarget(&mut self, center: IVec3, camera: Vec3, frustum: &Frustum, renderer: &mut Renderer) {
        let r = self.view_radius;
        let mut wanted: Vec<(f32, ChunkCoord)> = Vec::new();
        for dz in -r..=r {
            for dx in -r..=r {
                if dx * dx + dz * dz > r * r {
                    continue;
                }
                let col = IVec2::new(center.x + dx, center.z + dz);
                let (band_lo, band_hi) = self.surface_band(col);
                let lo = band_lo.min(center.y - CAMERA_Y_REACH).max(center.y - r);
                let hi = band_hi.max(center.y + CAMERA_Y_REACH).min(center.y + r);
                for y in lo..=hi {
                    // Below the surface band only chunks near the camera are worth filling (caves).
                    let underground = y < band_lo;
                    if underground && (y - center.y).abs() > CAMERA_Y_REACH {
                        continue;
                    }
                    let coord = IVec3::new(col.x, y, col.y);
                    let rel_min = (coord * CHUNK).as_vec3() - camera;
                    let visible = frustum.intersects_box(rel_min, rel_min + CHUNK as f32);
                    let dist = ((coord - center).as_vec3() * Vec3::new(1.0, 1.5, 1.0)).length();
                    wanted.push((if visible { dist } else { dist + 6.0 }, coord));
                }
            }
        }
        wanted.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        self.queue.clear();
        for (_, coord) in wanted {
            let slot = self.slots.entry(coord).or_default();
            if slot.shown.is_none() && slot.in_flight == 0 {
                self.queue.push_back(coord);
            }
        }

        let keep = r + UNLOAD_MARGIN;
        self.slots.retain(|coord, slot| {
            let d = *coord - center;
            let inside = d.x * d.x + d.z * d.z <= keep * keep && d.y.abs() <= keep;
            if !inside && slot.in_flight == 0 {
                renderer.remove_chunk(*coord);
            }
            inside || slot.in_flight > 0
        });
        self.column_range.retain(|col, _| {
            let d = *col - IVec2::new(center.x, center.z);
            d.length_squared() <= (keep + 4) * (keep + 4)
        });
    }

    fn dispatch(&mut self, world: &World) {
        let capacity = self.workers * JOBS_PER_WORKER;
        while self.in_flight < capacity {
            let Some(coord) = self.urgent.pop_front().or_else(|| self.queue.pop_front()) else { break };
            let Some(slot) = self.slots.get_mut(&coord) else { continue };
            slot.in_flight += 1;
            self.in_flight += 1;
            let ops = world.edits.ops_for(coord).copied().collect();
            // The workers live as long as the streamer, so this cannot fail.
            let _ = self.jobs.send(Job { coord, version: slot.wanted, ops });
        }
    }

    fn receive(&mut self, renderer: &mut Renderer) {
        for built in self.results.try_iter() {
            self.in_flight -= 1;
            let alpha = 0.05;
            let (b, m) = (built.build_time.as_secs_f32() * 1e3, built.mesh_time.as_secs_f32() * 1e3);
            self.stats.avg_build_ms += (b - self.stats.avg_build_ms) * alpha;
            self.stats.avg_mesh_ms += (m - self.stats.avg_mesh_ms) * alpha;
            self.uploads.push_back(built);
        }
        // Edited chunks go first so digging feels immediate.
        if let Some((_, waiting)) = &self.pending_edit {
            let (mut first, rest): (VecDeque<_>, VecDeque<_>) =
                self.uploads.drain(..).partition(|b| waiting.contains(&b.coord));
            first.extend(rest);
            self.uploads = first;
        }
        for _ in 0..MAX_UPLOADS_PER_FRAME {
            let Some(built) = self.uploads.pop_front() else { break };
            let Some(slot) = self.slots.get_mut(&built.coord) else { continue };
            slot.in_flight -= 1;
            if slot.shown.is_some_and(|v| v > built.version) {
                continue;
            }
            slot.shown = Some(built.version);
            renderer.upload_chunk(built.coord, &built.mesh);
            if let Some((started, waiting)) = &mut self.pending_edit
                && built.version == slot.wanted
            {
                waiting.remove(&built.coord);
                if waiting.is_empty() {
                    self.stats.last_edit_ms = Some(started.elapsed().as_secs_f32() * 1e3);
                    self.pending_edit = None;
                }
            }
        }
    }
}
