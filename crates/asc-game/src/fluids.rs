//! The client's copy of the simulated water, and its meshes.
//!
//! The server sends every materialised fluid chunk (quantised levels) as it changes. Here they're
//! kept for swimming and drawn: each chunk's cells become a density field (half a cell minus the
//! fill), meshed with the terrain's surface nets. Neighbours that aren't materialised read as their
//! pristine water, worked out once per chunk. The sea plane is told which chunk columns to leave
//! to these meshes.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use ahash::AHashMap;
use asc_render::Renderer;
use asc_world::fluid::{self, CELLS, FULL, FluidChunk, FluidKind, PackedFluid, cell_index};
use asc_world::{CHUNK, ChunkCoord, ChunkData, SAMPLES, World, mesh_chunk};
use glam::{IVec3, Vec3};

/// A chunk is remeshed at most this often while its water moves.
const REMESH_INTERVAL: Duration = Duration::from_millis(100);
/// Remeshes per frame.
const REMESH_PER_FRAME: usize = 4;

pub struct ClientFluids {
    pub fluids: fluid::Fluids,
    /// Pristine water of chunks next to materialised ones, for meshing their edges.
    ghosts: AHashMap<ChunkCoord, FluidChunk>,
    /// Chunks waiting to be remeshed, and when each was last.
    dirty: BTreeMap<[i32; 3], Option<Instant>>,
    last_mesh: AHashMap<ChunkCoord, Instant>,
    columns_changed: bool,
    /// Each chunk's lava light, worked out when it last changed (None: no lava).
    lights: AHashMap<ChunkCoord, Option<asc_render::PointLight>>,
}

impl ClientFluids {
    pub fn new() -> Self {
        Self {
            fluids: fluid::Fluids::default(),
            ghosts: AHashMap::new(),
            dirty: BTreeMap::new(),
            last_mesh: AHashMap::new(),
            columns_changed: true,
            lights: AHashMap::new(),
        }
    }

    /// A chunk from the server.
    pub fn apply(&mut self, coord: ChunkCoord, packed: &PackedFluid) {
        let (kind, level) = packed.unpack();
        let mut ch = FluidChunk::empty();
        for i in 0..CELLS {
            ch.kind[i] = kind[i];
            ch.amount[i] = (level[i] as u32 * FULL as u32 / 255) as u16;
        }
        if !self.fluids.chunks.contains_key(&coord) {
            self.columns_changed = true;
        }
        self.fluids.chunks.insert(coord, ch);
        self.lights.remove(&coord);
        // Its own mesh, and its neighbours' edges.
        self.dirty.entry(coord.to_array()).or_insert(None);
        for d in [IVec3::X, IVec3::NEG_X, IVec3::Y, IVec3::NEG_Y, IVec3::Z, IVec3::NEG_Z] {
            if self.fluids.chunks.contains_key(&(coord + d)) {
                self.dirty.entry((coord + d).to_array()).or_insert(None);
            }
        }
    }

    /// How much water is at a point, 0..1 (for swimming).
    pub fn fraction_at(&self, world: &World, p: Vec3) -> f32 {
        match self.fluids.fraction_at(&world.worldgen, p) {
            (FluidKind::Water, f) => f,
            _ => 0.0,
        }
    }

    /// The fill (0..=FULL) of a cell, materialised or pristine.
    fn amount(&mut self, world: &World, c: IVec3) -> u16 {
        let (coord, local) = fluid::split(c);
        let i = cell_index(local.x as usize, local.y as usize, local.z as usize);
        if let Some(ch) = self.fluids.chunks.get(&coord) {
            return ch.amount[i];
        }
        let ghost = self
            .ghosts
            .entry(coord)
            .or_insert_with(|| fluid::pristine_chunk(&world.worldgen, &world.build_chunk(coord)));
        ghost.amount[i]
    }

    /// The fluid kind of a cell, materialised or pristine.
    fn kind(&mut self, world: &World, c: IVec3) -> FluidKind {
        let (coord, local) = fluid::split(c);
        let i = cell_index(local.x as usize, local.y as usize, local.z as usize);
        if let Some(ch) = self.fluids.chunks.get(&coord) {
            return FluidKind::from_u8(ch.kind[i]);
        }
        let ghost = self
            .ghosts
            .entry(coord)
            .or_insert_with(|| fluid::pristine_chunk(&world.worldgen, &world.build_chunk(coord)));
        FluidKind::from_u8(ghost.kind[i])
    }

    /// A light for every materialised chunk of lava near the camera: at the middle of its surface, brighter and
    /// reaching further the more lava there is.
    pub fn lava_lights(&mut self, camera: Vec3) -> Vec<asc_render::PointLight> {
        let mut lights = Vec::new();
        let coords: Vec<ChunkCoord> = self
            .fluids
            .chunks
            .keys()
            .copied()
            .filter(|c| (c.as_vec3() * CHUNK as f32 + Vec3::splat(16.0)).distance(camera) < 160.0)
            .collect();
        for coord in coords {
            if let Some(light) = self.lights.get(&coord) {
                if let Some(l) = light {
                    lights.push(*l);
                }
                continue;
            }
            let Some(ch) = self.fluids.chunks.get(&coord) else { continue };
            let (mut sum, mut count) = (Vec3::ZERO, 0u32);
            for i in 0..CELLS {
                if ch.kind[i] != FluidKind::Lava as u8 || ch.amount[i] == 0 {
                    continue;
                }
                // Surface cells: lava with no lava above it.
                let y = (i / 32) % 32;
                if y + 1 < 32 && ch.kind[i + 32] == FluidKind::Lava as u8 {
                    continue;
                }
                sum += Vec3::new((i % 32) as f32, y as f32, (i / 1024) as f32);
                count += 1;
            }
            let light = (count > 0).then(|| lava_light_at(coord.as_vec3() * CHUNK as f32 + sum / count as f32, count));
            self.lights.insert(coord, light);
            if let Some(l) = light {
                lights.push(l);
            }
        }
        lights
    }

    fn mesh(&mut self, world: &World, coord: ChunkCoord) -> asc_world::ChunkMesh {
        let n = SAMPLES * SAMPLES * SAMPLES;
        let base = coord * CHUNK - IVec3::ONE;
        let mut density = vec![0.5f32; n];
        for z in 0..SAMPLES {
            for y in 0..SAMPLES {
                for x in 0..SAMPLES {
                    let c = base + IVec3::new(x as i32, y as i32, z as i32);
                    let a = self.amount(world, c).min(FULL) as f32 / FULL as f32;
                    density[ChunkData::index(x, y, z)] = 0.5 - a;
                }
            }
        }
        // The mesher blends materials per vertex; the fluid shader reads slot 1 (id 2) as "how much
        // of this is lava" (water is id 1).
        let mut material = vec![1u8; n];
        for z in 0..SAMPLES {
            for y in 0..SAMPLES {
                for x in 0..SAMPLES {
                    let c = base + IVec3::new(x as i32, y as i32, z as i32);
                    if self.kind(world, c) == FluidKind::Lava {
                        material[ChunkData::index(x, y, z)] = 2;
                    }
                }
            }
        }
        let mut data = ChunkData {
            coord,
            density,
            material,
            // Open sky everywhere: water isn't darkened as if buried.
            surface: vec![1e9; SAMPLES * SAMPLES],
            water: vec![f32::NAN; SAMPLES * SAMPLES],
            uniform: None,
        };
        data.update_uniform();
        mesh_chunk(&data)
    }

    /// Remesh what changed (a few chunks a frame, each at most every `REMESH_INTERVAL`) and tell the
    /// sea plane which columns to skip.
    pub fn update(&mut self, world: &World, renderer: &mut Renderer) {
        let now = Instant::now();
        let ready: Vec<[i32; 3]> = self
            .dirty
            .keys()
            .copied()
            .filter(|k| {
                self.last_mesh
                    .get(&IVec3::from_array(*k))
                    .is_none_or(|t| now.duration_since(*t) >= REMESH_INTERVAL)
            })
            .take(REMESH_PER_FRAME)
            .collect();
        for k in ready {
            self.dirty.remove(&k);
            let coord = IVec3::from_array(k);
            let mesh = self.mesh(world, coord);
            renderer.upload_fluid(coord, &mesh);
            self.last_mesh.insert(coord, now);
        }
        if self.columns_changed {
            self.columns_changed = false;
            // Only where the chunk at the sea's surface is materialised (one deep under the sea
            // floor must not punch a hole in it).
            // Only where the chunk at the natural water's surface (at the column's middle) is
            // materialised: one deep under a lake must not punch a hole in it.
            let mut columns: Vec<[i32; 2]> = self
                .fluids
                .chunks
                .keys()
                .filter(|c| {
                    let mid = (c.as_vec3() + Vec3::splat(0.5)) * CHUNK as f32;
                    world.worldgen.water_level(mid.x, mid.z).is_some_and(|l| (l / CHUNK as f32).floor() as i32 == c.y)
                })
                .map(|c| [c.x, c.z])
                .collect();
            columns.sort_unstable();
            columns.dedup();
            renderer.set_fluid_columns(&columns);
        }
    }
}

/// The light of `count` cells of lava surface centred on `at`.
pub fn lava_light_at(at: Vec3, count: u32) -> asc_render::PointLight {
    let size = (count as f32).sqrt();
    asc_render::PointLight {
        position: at + Vec3::Y * 1.5,
        radius: 8.0 + size * 1.6,
        color: Vec3::new(1.0, 0.38, 0.08) * (1.5 + size * 0.12),
    }
}

impl Default for ClientFluids {
    fn default() -> Self {
        Self::new()
    }
}
