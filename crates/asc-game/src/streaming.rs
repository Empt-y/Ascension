//! Keeps the chunks around the camera generated, meshed and uploaded.
//!
//! The main thread decides what is wanted and in which order; a pool of worker threads builds chunks
//! (worldgen + replayed edits + meshing, and the mesh of the chunk's natural water) and sends meshes
//! back. Only a few jobs per worker are in
//! flight at once so priorities stay fresh as the camera moves. Every request carries a version; edits
//! bump the version, and a result is uploaded only if it is the newest one seen for that chunk.

use std::collections::VecDeque;
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use ahash::{AHashMap, AHashSet};
use asc_render::{Frustum, Renderer};
use asc_world::fluid;

use crate::fluids::ClientFluids;
use asc_world::{CHUNK, ChunkCoord, ChunkMesh, EditOp, World, WorldGen, chunk_of, mesh_chunk};
use crossbeam_channel::{Receiver, Sender};
use glam::{IVec2, IVec3, Vec3};

const JOBS_PER_WORKER: usize = 3;
/// Chunks are dropped this many chunks beyond the view radius, so small moves don't thrash.
const UNLOAD_MARGIN: i32 = 2;
/// Vertical chunks kept around the camera regardless of terrain (caves, flying through hills)...
const CAMERA_Y_REACH: i32 = 1;
/// ...within this many chunks horizontally.
const CAMERA_XZ_REACH: i32 = 4;
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
    /// Its natural water and lava (drawn unless the chunk's fluid is materialised).
    fluid: ChunkMesh,
    /// The light of its natural lava, if any.
    lava_light: Option<asc_render::PointLight>,
    /// The surface leaves through the chunk's top / bottom face, so the chunk above / below is needed.
    spills_up: bool,
    spills_down: bool,
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
    /// Out of range now: kept only until its jobs come back, then dropped (not drawn).
    unwanted: bool,
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
    /// Worst main-thread cost of retargeting / uploading since the stats were last taken.
    pub worst_retarget_ms: f32,
    pub worst_upload_ms: f32,
}

pub struct Streamer {
    jobs: Sender<Job>,
    results: Receiver<Built>,
    workers: usize,
    slots: AHashMap<ChunkCoord, Slot>,
    /// Lights of the loaded chunks' natural lava.
    lights: AHashMap<ChunkCoord, asc_render::PointLight>,
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
        // Saturating the cores starves the render thread and the GPU driver's own threads: with 14
        // workers frames spiked to 15-35 ms while flying, with 7 to ~16 ms, with 4 they stay under ~10 ms
        // and still keep up at 40 m/s. A quarter of the cores, at low priority. ASC_WORKERS overrides.
        let cores = thread::available_parallelism().map_or(4, |n| n.get());
        let workers =
            std::env::var("ASC_WORKERS").ok().and_then(|v| v.parse().ok()).unwrap_or(cores / 4).max(2);
        let (job_tx, job_rx) = crossbeam_channel::unbounded::<Job>();
        let (done_tx, done_rx) = crossbeam_channel::unbounded::<Built>();
        for i in 0..workers {
            let (rx, tx, wg) = (job_rx.clone(), done_tx.clone(), worldgen.clone());
            thread::Builder::new()
                .name(format!("chunk-worker-{i}"))
                .spawn(move || {
                    if let Err(err) =
                        thread_priority::set_current_thread_priority(thread_priority::ThreadPriority::Min)
                    {
                        log::warn!("could not lower chunk worker priority: {err:?}");
                    }
                    for job in rx {
                        let t0 = Instant::now();
                        let natural = wg.generate(job.coord);
                        let mut data = natural.clone();
                        data.apply_ops(&job.ops);
                        data.update_uniform();
                        let fluid = fluid::natural_mesh(&wg, &natural, &data);
                        let lava_light = lava_light(job.coord, &fluid);
                        let t1 = Instant::now();
                        let mesh = mesh_chunk(&data);
                        let (spills_up, spills_down) = spills(&mesh);
                        let built = Built {
                            coord: job.coord,
                            version: job.version,
                            mesh,
                            fluid,
                            lava_light,
                            spills_up,
                            spills_down,
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
            lights: AHashMap::new(),
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

    /// Read the stats and reset the worst-case timers.
    pub fn take_worst(&mut self) -> (f32, f32) {
        let w = (self.stats.worst_retarget_ms, self.stats.worst_upload_ms);
        self.stats.worst_retarget_ms = 0.0;
        self.stats.worst_upload_ms = 0.0;
        w
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
                let (h, water) = self.worldgen.height_and_water(base.x + dx, base.y + dz);
                lo = lo.min(h);
                // Up to the water's surface too (deep water's surface is in a chunk of its own).
                hi = hi.max(water.map_or(h, |w| w.max(h)));
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

    /// `fluids`: the simulated water, which draws its chunks (and their neighbours) itself.
    /// Lights of natural lava within `reach` of the camera, leaving out materialised chunks (they
    /// light themselves).
    pub fn lava_lights(&self, fluids: &ClientFluids, camera: Vec3, reach: f32) -> Vec<asc_render::PointLight> {
        self.lights
            .iter()
            .filter(|(c, l)| !fluids.fluids.chunks.contains_key(*c) && l.position.distance(camera) < reach)
            .map(|(_, l)| *l)
            .collect()
    }

    pub fn update(
        &mut self,
        world: &World,
        fluids: &ClientFluids,
        camera: Vec3,
        frustum: &Frustum,
        renderer: &mut Renderer,
    ) {
        let center = chunk_of(camera);
        if self.last_center != Some(center) || self.last_radius != self.view_radius {
            let t = Instant::now();
            self.retarget(center, camera, frustum, fluids, renderer);
            self.stats.worst_retarget_ms = self.stats.worst_retarget_ms.max(t.elapsed().as_secs_f32() * 1e3);
            self.last_center = Some(center);
            self.last_radius = self.view_radius;
        }
        self.dispatch(world);
        let t = Instant::now();
        self.receive(fluids, renderer);
        self.stats.worst_upload_ms = self.stats.worst_upload_ms.max(t.elapsed().as_secs_f32() * 1e3);
    }

    /// Recompute which chunks are wanted and in what order; drop the ones that fell out of range.
    fn retarget(&mut self, center: IVec3, camera: Vec3, frustum: &Frustum, fluids: &ClientFluids, renderer: &mut Renderer) {
        let r = self.view_radius;
        let mut wanted: Vec<(f32, ChunkCoord)> = Vec::new();
        for dz in -r..=r {
            for dx in -r..=r {
                if dx * dx + dz * dz > r * r {
                    continue;
                }
                let col = IVec2::new(center.x + dx, center.z + dz);
                let (band_lo, band_hi) = self.surface_band(col);
                // The surface band (whatever the camera's height: seen from high up, that's the
                // ground below) and the chunks round the camera, but not the air between them.
                // (Round the camera only close by: caves, and flying through nearby terrain.)
                let close = dx * dx + dz * dz <= CAMERA_XZ_REACH * CAMERA_XZ_REACH;
                #[allow(clippy::reversed_empty_ranges)]
                let near_camera = if close { (center.y - CAMERA_Y_REACH)..=(center.y + CAMERA_Y_REACH) } else { 1..=0 };
                let ys = (band_lo..=band_hi).chain(near_camera.filter(|y| *y < band_lo || *y > band_hi));
                for y in ys {
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
        let mut queued = AHashSet::new();
        for (_, coord) in wanted {
            let slot = self.slots.entry(coord).or_default();
            slot.unwanted = false;
            if slot.shown.is_none() && slot.in_flight == 0 {
                self.queue.push_back(coord);
                queued.insert(coord);
            }
        }
        // Chunks added by following the surface aren't in the band estimate; keep them queued too.
        for (coord, slot) in &self.slots {
            if slot.shown.is_none() && slot.in_flight == 0 && !queued.contains(coord) {
                self.queue.push_back(*coord);
            }
        }

        let keep = r + UNLOAD_MARGIN;
        let column_range = &self.column_range;
        self.slots.retain(|coord, slot| {
            let d = *coord - center;
            let in_band = column_range
                .get(&IVec2::new(coord.x, coord.z))
                .is_some_and(|(lo, hi)| (lo - 1..=hi + 1).contains(&coord.y));
            let inside = d.x * d.x + d.z * d.z <= keep * keep && (d.y.abs() <= keep || in_band);
            if !inside {
                // Whatever it had on screen goes now; a job still out is dropped when it's back.
                renderer.remove_chunk(*coord);
                self.lights.remove(coord);
                if !fluids.owns(*coord) {
                    renderer.remove_fluid(*coord);
                }
                slot.unwanted = true;
                slot.shown = None;
            }
            inside || slot.in_flight > 0
        });
        self.column_range.retain(|col, _| {
            let d = *col - IVec2::new(center.x, center.z);
            d.length_squared() <= (keep + 4) * (keep + 4)
        });
    }

    fn follow(&mut self, coord: ChunkCoord) {
        let Some(center) = self.last_center else { return };
        let d = coord - center;
        let r = self.view_radius;
        if self.slots.contains_key(&coord) || d.x * d.x + d.z * d.z > r * r || d.y.abs() > r {
            return;
        }
        self.slots.insert(coord, Slot::default());
        self.queue.push_front(coord);
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

    fn receive(&mut self, fluids: &ClientFluids, renderer: &mut Renderer) {
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
            if slot.unwanted {
                if slot.in_flight == 0 {
                    self.slots.remove(&built.coord);
                }
                continue;
            }
            if slot.shown.is_some_and(|v| v > built.version) {
                continue;
            }
            slot.shown = Some(built.version);
            let current = built.version == slot.wanted;
            renderer.upload_chunk(built.coord, &built.mesh);
            if !fluids.owns(built.coord) {
                renderer.upload_fluid(built.coord, &built.fluid);
            }
            match built.lava_light {
                Some(l) => self.lights.insert(built.coord, l),
                None => self.lights.remove(&built.coord),
            };
            // Follow the surface vertically: the band estimate misses peaks and steep cliffs.
            if built.spills_up {
                self.follow(built.coord + IVec3::Y);
            }
            if built.spills_down {
                self.follow(built.coord - IVec3::Y);
            }
            if let Some((started, waiting)) = &mut self.pending_edit
                && current
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

/// A light at the middle of a chunk's natural lava surface, brighter and reaching further the more
/// of it there is (as for simulated lava).
fn lava_light(coord: ChunkCoord, mesh: &ChunkMesh) -> Option<asc_render::PointLight> {
    let (mut sum, mut count) = (Vec3::ZERO, 0u32);
    for v in mesh.vertices.iter().filter(|v| v.weights[1] > 128 && v.normal[1] > 64) {
        sum += Vec3::from(v.position);
        count += 1;
    }
    (count > 0).then(|| crate::fluids::lava_light_at(coord.as_vec3() * CHUNK as f32 + sum / count as f32, count))
}

/// Whether any vertex sits on the chunk's top or bottom face, i.e. the surface continues past it.
fn spills(mesh: &ChunkMesh) -> (bool, bool) {
    let top = CHUNK as f32 - 0.5;
    let (mut up, mut down) = (false, false);
    for v in &mesh.vertices {
        up |= v.position[1] >= top;
        down |= v.position[1] <= 0.5;
    }
    (up, down)
}
