//! The client's copy of the simulated water, and its meshes.
//!
//! The server sends every materialised fluid chunk (quantised levels) as it changes. Here they're
//! kept for swimming and drawn with `fluid::mesh_fluid`, as are the chunks next to them (so the
//! simulated water meets its surroundings without a seam; further out the streamer draws natural
//! water the same way). Cells of chunks that aren't materialised read as their pristine water,
//! worked out once per chunk. The sea plane is told which chunk columns to leave to these meshes.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use ahash::AHashMap;
use asc_render::Renderer;
use asc_world::fluid::{self, CELLS, FULL, FluidChunk, FluidKind, PackedFluid, MARGIN, RING, cell_index};
use asc_world::{CHUNK, ChunkCoord, ChunkData, SAMPLES, World};
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
    /// Built terrain of the chunks meshed here, and the edit count it was built at.
    terrain: AHashMap<ChunkCoord, (usize, Option<ChunkData>)>,
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
            terrain: AHashMap::new(),
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
        // Its own mesh, and its neighbours' (which meet it, so they're drawn from here too).
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    self.dirty.entry((coord + IVec3::new(dx, dy, dz)).to_array()).or_insert(None);
                }
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

    /// Whether this draws the chunk's fluid (rather than the streamer's natural mesh): it's
    /// materialised or next to a chunk that is, so its mesh has to meet the simulated water.
    pub fn owns(&self, coord: ChunkCoord) -> bool {
        (-1..=1).any(|dz| {
            (-1..=1).any(|dy| (-1..=1).any(|dx| self.fluids.chunks.contains_key(&(coord + IVec3::new(dx, dy, dz)))))
        })
    }

    /// The chunk's built terrain, kept until the world's edits change.
    fn terrain(&mut self, world: &World, coord: ChunkCoord) -> &ChunkData {
        let edits = world.edits.len();
        let entry = self.terrain.entry(coord).or_insert_with(|| (usize::MAX, None));
        if entry.0 != edits || entry.1.is_none() {
            *entry = (edits, Some(world.build_chunk(coord)));
        }
        entry.1.as_ref().expect("just built")
    }

    fn mesh(&mut self, world: &World, coord: ChunkCoord) -> asc_world::ChunkMesh {
        // Every cell the mesh reads: the padded samples, the ring of columns round them and a few
        // cells above and below (for the water surfaces).
        let base = coord * CHUNK;
        // Padded sample 0 is at base - 1; the ring reaches MARGIN columns further.
        let lo = base - IVec3::new(1 + MARGIN as i32, 7, 1 + MARGIN as i32);
        let size = IVec3::new(RING as i32, SAMPLES as i32 + 12, RING as i32);
        for cz in (lo.z.div_euclid(CHUNK))..=((lo.z + size.z - 1).div_euclid(CHUNK)) {
            for cy in (lo.y.div_euclid(CHUNK))..=((lo.y + size.y - 1).div_euclid(CHUNK)) {
                for cx in (lo.x.div_euclid(CHUNK))..=((lo.x + size.x - 1).div_euclid(CHUNK)) {
                    let c = IVec3::new(cx, cy, cz);
                    if !self.fluids.chunks.contains_key(&c) && !self.ghosts.contains_key(&c) {
                        let ghost = fluid::pristine_chunk(&world.worldgen, &world.build_chunk(c));
                        self.ghosts.insert(c, ghost);
                    }
                }
            }
        }
        let cells = size.x as usize * size.y as usize * size.z as usize;
        let (mut kind, mut amount) = (vec![0u8; cells], vec![0u16; cells]);
        let at = |p: IVec3| {
            let r = p - lo;
            r.x as usize + size.x as usize * (r.y as usize + size.y as usize * r.z as usize)
        };
        let mut last: Option<(ChunkCoord, &FluidChunk)> = None;
        for z in 0..size.z {
            for y in 0..size.y {
                for x in 0..size.x {
                    let c = lo + IVec3::new(x, y, z);
                    let (chunk, local) = fluid::split(c);
                    let ch = match last {
                        Some((at, ch)) if at == chunk => ch,
                        _ => {
                            let ch = self.fluids.chunks.get(&chunk).or_else(|| self.ghosts.get(&chunk)).expect("gathered");
                            last = Some((chunk, ch));
                            ch
                        }
                    };
                    let i = cell_index(local.x as usize, local.y as usize, local.z as usize);
                    let k = at(c);
                    kind[k] = ch.kind[i];
                    amount[k] = ch.amount[i];
                }
            }
        }
        let get = |c: IVec3| {
            let r = c - lo;
            if r.cmplt(IVec3::ZERO).any() || r.cmpge(size).any() {
                return (FluidKind::None, 0);
            }
            let k = at(c);
            (FluidKind::from_u8(kind[k]), amount[k])
        };
        let mut surface = vec![f32::NAN; RING * RING];
        for rz in 0..RING {
            for rx in 0..RING {
                let (x, z) = (lo.x + rx as i32, lo.z + rz as i32);
                surface[rx + RING * rz] =
                    fluid::column_surface(|y| get(IVec3::new(x, y, z)), lo.y + size.y - 1, lo.y);
            }
        }
        let n = SAMPLES * SAMPLES * SAMPLES;
        let (mut pk, mut pa) = (vec![0u8; n], vec![0u16; n]);
        for z in 0..SAMPLES {
            for y in 0..SAMPLES {
                for x in 0..SAMPLES {
                    let (k, a) = get(base - IVec3::ONE + IVec3::new(x as i32, y as i32, z as i32));
                    pk[ChunkData::index(x, y, z)] = k as u8;
                    pa[ChunkData::index(x, y, z)] = a;
                }
            }
        }
        let terrain = self.terrain(world, coord);
        fluid::mesh_fluid(&fluid::FluidInput { terrain, kind: &pk, amount: &pa, surface: &surface, flow: None })
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

#[cfg(test)]
mod tests {
    use super::*;

    /// A chunk next to simulated water is meshed here, from the simulation's cells and their
    /// neighbours' pristine water: it should show the same water as the chunk's natural mesh.
    #[test]
    fn chunks_next_to_simulated_water_mesh_their_water() {
        let world = World::new(1256132528);
        let mut fluids = ClientFluids::new();
        let materialised = IVec3::new(-74, -1, 54);
        let pristine = fluid::pristine_chunk(&world.worldgen, &world.build_chunk(materialised));
        fluids.apply(materialised, &PackedFluid::pack(&pristine));
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let coord = materialised + IVec3::new(dx, dy, dz);
                    let natural = world.worldgen.generate(coord);
                    let expected = fluid::natural_mesh(&world.worldgen, &natural, &natural);
                    let ours = fluids.mesh(&world, coord);
                    let water = |m: &asc_world::ChunkMesh| m.vertices.iter().filter(|v| v.weights[0] > 128).count();
                    // Vertices deep enough to be drawn (the shader fades depth 0 to clear).
                    let visible = |m: &asc_world::ChunkMesh| {
                        m.vertices.iter().filter(|v| v.weights[0] > 128 && v.ore[2] > 0 && v.ore[3] > 30).count()
                    };
                    let (e, o) = (water(&expected), water(&ours));
                    println!("{coord}: natural {e} water vertices ({} visible), meshed here {o} ({} visible)", visible(&expected), visible(&ours));
                    assert!(visible(&expected) == 0 || visible(&ours) * 2 > visible(&expected), "{coord}: ours aren't drawn");
                    assert!(e == 0 || o * 2 > e, "{coord}: natural mesh has {e} water vertices, ours {o}");
                }
            }
        }
    }
}
