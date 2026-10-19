//! Naive surface nets over a chunk's padded samples.
//!
//! One vertex is placed in every cell whose corners straddle the surface, at the average of its edge
//! crossings. One quad is emitted for every sample edge that crosses the surface, joining the four cells
//! around it. A chunk only emits quads for edges whose low end lies inside the chunk, so each edge in
//! the world belongs to exactly one chunk; the padding provides the neighbouring cells. Since every
//! chunk stores identical values for shared samples, vertices along chunk borders coincide exactly.

use glam::Vec3;

use crate::chunk::{CHUNK, ChunkData, SAMPLES};

/// Vertex positions are rounded to 1 / this many metres (see the comment where they're made).
pub const POSITION_GRID: f32 = 1024.0;

/// Materials that can be blended per vertex: slot `i` holds material id `i + 1` (0 is air).
pub const MATERIAL_SLOTS: usize = 12;

/// 28 bytes. The renderer reads `normal` as snorm8x4 and `weights` as three unorm8x4.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Vertex {
    /// Chunk-local position in metres.
    pub position: [f32; 3],
    /// xyz: unit normal. w: how much ambient light reaches this point (127 = open sky).
    pub normal: [i8; 4],
    /// Blend weights of the surrounding materials, summing to about 255.
    pub weights: [u8; MATERIAL_SLOTS],
}

impl Vertex {
    pub fn normal(&self) -> Vec3 {
        Vec3::new(self.normal[0] as f32, self.normal[1] as f32, self.normal[2] as f32) / 127.0
    }
}

#[derive(Default)]
pub struct ChunkMesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
}

impl ChunkMesh {
    pub fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }
}

/// Cells per edge in the padded grid.
const CELLS: usize = SAMPLES - 1;
const NO_VERTEX: u32 = u32::MAX;

/// Corner offsets of a cell, bit i of the index selects +1 on axis i.
const CORNERS: [[usize; 3]; 8] =
    [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]];
/// The 12 cell edges as corner index pairs.
const EDGES: [(usize, usize); 12] =
    [(0, 1), (2, 3), (4, 5), (6, 7), (0, 2), (1, 3), (4, 6), (5, 7), (0, 4), (1, 5), (2, 6), (3, 7)];

pub fn mesh_chunk(chunk: &ChunkData) -> ChunkMesh {
    let mut mesh = ChunkMesh::default();
    if chunk.uniform.is_some() {
        return mesh;
    }
    let d = &chunk.density;
    let idx = ChunkData::index;
    let cell_index = |x: usize, y: usize, z: usize| x + CELLS * (y + CELLS * z);
    let mut cell_vertex = vec![NO_VERTEX; CELLS * CELLS * CELLS];

    // Pass 1: a vertex in every cell that straddles the surface.
    for z in 0..CELLS {
        for y in 0..CELLS {
            for x in 0..CELLS {
                let mut corner = [0.0f32; 8];
                let mut solid_mask = 0u8;
                for (i, o) in CORNERS.iter().enumerate() {
                    corner[i] = d[idx(x + o[0], y + o[1], z + o[2])];
                    if corner[i] < 0.0 {
                        solid_mask |= 1 << i;
                    }
                }
                if solid_mask == 0 || solid_mask == 0xff {
                    continue;
                }
                let mut sum = Vec3::ZERO;
                let mut count = 0.0;
                for &(a, b) in &EDGES {
                    let (da, db) = (corner[a], corner[b]);
                    if (da < 0.0) != (db < 0.0) {
                        let t = da / (da - db);
                        let pa = Vec3::from(CORNERS[a].map(|v| v as f32));
                        let pb = Vec3::from(CORNERS[b].map(|v| v as f32));
                        sum += pa.lerp(pb, t);
                        count += 1.0;
                    }
                }
                let local = sum / count;
                let normal = gradient(&corner, local).normalize_or(Vec3::Y);

                // Blend the materials of the solid corners, deeper ones counting more, so material
                // boundaries fade across triangles instead of following them.
                let mut weight = [0.0f32; MATERIAL_SLOTS];
                for (i, o) in CORNERS.iter().enumerate() {
                    let m = chunk.material[idx(x + o[0], y + o[1], z + o[2])] as usize;
                    if corner[i] < 0.0 && (1..=MATERIAL_SLOTS).contains(&m) {
                        weight[m - 1] += (-corner[i]).min(2.0) + 0.05;
                    }
                }
                let total: f32 = weight.iter().sum::<f32>().max(1e-6);
                let weights = weight.map(|w| (w / total * 255.0).round() as u8);

                // Cell (x, y, z) has its low corner at sample (x, y, z), which sits at chunk-local x - 1.
                // On a 1/1024 m grid, so that chunk offset + local position is exact in f32 and the
                // two chunks sharing a border vertex put it in bit-identically the same place on the
                // GPU (otherwise hairline cracks open along chunk seams and sparkle). Only the offset
                // within the cell is rounded: it comes from the same samples in both chunks, so it
                // rounds the same way; the whole cell position is then added exactly.
                let within = (local * POSITION_GRID).round() / POSITION_GRID;
                let position = (Vec3::new(x as f32, y as f32, z as f32) - Vec3::ONE) + within;
                let column = (x + local.x.round() as usize) + SAMPLES * (z + local.z.round() as usize);
                let buried = chunk.surface[column] - (chunk.coord.y * CHUNK) as f32 - position.y;
                let occlusion = 1.0 - smoothstep(2.0, 12.0, buried) * 0.85;

                let n = (normal * 127.0).round();
                cell_vertex[cell_index(x, y, z)] = mesh.vertices.len() as u32;
                mesh.vertices.push(Vertex {
                    position: position.into(),
                    normal: [n.x as i8, n.y as i8, n.z as i8, (occlusion * 127.0).round() as i8],
                    weights,
                });
            }
        }
    }

    // Pass 2: a quad for every crossing edge owned by this chunk (low sample in [1, CHUNK]).
    let own = 1..=CHUNK as usize;
    for z in own.clone() {
        for y in own.clone() {
            for x in own.clone() {
                let here = d[idx(x, y, z)];
                // (axis step, first and second perpendicular axes) chosen so e1 × e2 = axis.
                for axis in 0..3 {
                    let (nx, ny, nz) = match axis {
                        0 => (x + 1, y, z),
                        1 => (x, y + 1, z),
                        _ => (x, y, z + 1),
                    };
                    let there = d[idx(nx, ny, nz)];
                    if (here < 0.0) == (there < 0.0) {
                        continue;
                    }
                    let (e1, e2) = match axis {
                        0 => ([0, 1, 0], [0, 0, 1]),
                        1 => ([0, 0, 1], [1, 0, 0]),
                        _ => ([1, 0, 0], [0, 1, 0]),
                    };
                    let cell = |a: usize, b: usize| {
                        let c =
                            [x - a * e1[0] - b * e2[0], y - a * e1[1] - b * e2[1], z - a * e1[2] - b * e2[2]];
                        cell_vertex[cell_index(c[0], c[1], c[2])]
                    };
                    // Counter-clockwise around +axis: (-1,-1), (0,-1), (0,0), (-1,0) in (e1, e2).
                    let quad = [cell(1, 1), cell(0, 1), cell(0, 0), cell(1, 0)];
                    if quad.contains(&NO_VERTEX) {
                        continue;
                    }
                    // Solid at the low end means the surface faces +axis; otherwise flip.
                    let [a, b, c, e] = if here < 0.0 { quad } else { [quad[0], quad[3], quad[2], quad[1]] };
                    // Split along the shorter diagonal; always using the same one zigzags along creases.
                    let p = |i: u32| Vec3::from(mesh.vertices[i as usize].position);
                    if p(a).distance_squared(p(c)) <= p(b).distance_squared(p(e)) {
                        mesh.indices.extend_from_slice(&[a, b, c, a, c, e]);
                    } else {
                        mesh.indices.extend_from_slice(&[b, c, e, b, e, a]);
                    }
                }
            }
        }
    }
    mesh
}

/// Gradient of the trilinear interpolation of a cell's corners at `t` (each axis in [0, 1]).
fn gradient(c: &[f32; 8], t: Vec3) -> Vec3 {
    let lerp = |a: f32, b: f32, t: f32| a + (b - a) * t;
    // d/dx: difference of x-faces interpolated over y and z.
    let dx = lerp(lerp(c[1] - c[0], c[3] - c[2], t.y), lerp(c[5] - c[4], c[7] - c[6], t.y), t.z);
    let dy = lerp(lerp(c[2] - c[0], c[3] - c[1], t.x), lerp(c[6] - c[4], c[7] - c[5], t.x), t.z);
    let dz = lerp(lerp(c[4] - c[0], c[5] - c[1], t.x), lerp(c[6] - c[2], c[7] - c[3], t.x), t.y);
    Vec3::new(dx, dy, dz)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
