//! Deterministic procedural terrain.
//!
//! A 2D height field (continents, hills, domain-warped ridged mountains) gives the base surface. 3D noise
//! then adds overhangs on mountains and carves spaghetti caves. Every value is a pure function of the
//! seed and world position, so any chunk can be regenerated on any thread at any time.

use fastnoise_lite::{DomainWarpType, FastNoiseLite, FractalType, NoiseType};
use glam::{Vec2, Vec3};

use crate::chunk::{CHUNK, ChunkCoord, ChunkData, SAMPLES};
use crate::edit::DENSITY_CLAMP;
use crate::material::Material;

/// Surfaces below this height are beaches.
const SEA_LEVEL: f32 = 2.0;
const SNOW_LINE: f32 = 140.0;

pub struct WorldGen {
    seed: i32,
    continent: FastNoiseLite,
    hills: FastNoiseLite,
    ridges: FastNoiseLite,
    warp: FastNoiseLite,
    overhang: FastNoiseLite,
    cave_a: FastNoiseLite,
    cave_b: FastNoiseLite,
    ore: FastNoiseLite,
}

/// Per-(x, z) values shared by every sample in a column.
#[derive(Clone, Copy)]
struct Column {
    height: f32,
    /// Length of the height gradient (rise over run).
    slope: f32,
    /// 1 / |∇(y - h)|, which turns `y - h` into an approximate true distance.
    inv_grad: f32,
    /// 0 on plains, 1 in mountains. Gates overhangs.
    mountain: f32,
}

fn noise(seed: i32, kind: NoiseType, freq: f32, fractal: Option<(FractalType, i32)>) -> FastNoiseLite {
    let mut n = FastNoiseLite::with_seed(seed);
    n.set_noise_type(Some(kind));
    n.set_frequency(Some(freq));
    if let Some((ty, octaves)) = fractal {
        n.set_fractal_type(Some(ty));
        n.set_fractal_octaves(Some(octaves));
    }
    n
}

/// Polynomial smooth maximum with blend width `k`.
fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (a - b) / k).clamp(0.0, 1.0);
    b + (a - b) * h + k * h * (1.0 - h)
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl WorldGen {
    pub fn new(seed: i32) -> Self {
        use FractalType::{FBm, Ridged};
        use NoiseType::OpenSimplex2;
        let mut warp =
            noise(seed.wrapping_add(3), OpenSimplex2, 0.002, Some((FractalType::DomainWarpProgressive, 2)));
        warp.set_domain_warp_type(Some(DomainWarpType::OpenSimplex2));
        warp.set_domain_warp_amp(Some(60.0));
        Self {
            seed,
            continent: noise(seed, OpenSimplex2, 0.0006, Some((FBm, 3))),
            hills: noise(seed.wrapping_add(1), OpenSimplex2, 0.005, Some((FBm, 5))),
            ridges: noise(seed.wrapping_add(2), OpenSimplex2, 0.0018, Some((Ridged, 5))),
            warp,
            overhang: noise(seed.wrapping_add(4), OpenSimplex2, 0.014, Some((FBm, 2))),
            cave_a: noise(seed.wrapping_add(5), OpenSimplex2, 0.012, None),
            cave_b: noise(seed.wrapping_add(6), OpenSimplex2, 0.012, None),
            ore: noise(seed.wrapping_add(7), OpenSimplex2, 0.08, None),
        }
    }

    pub fn seed(&self) -> i32 {
        self.seed
    }

    /// Mountain mask in [0, 1] from the continent value.
    fn mountain_mask(continent: f32) -> f32 {
        smoothstep(0.05, 0.55, continent)
    }

    /// Terrain height at (x, z), before any 3D features.
    pub fn height(&self, x: f32, z: f32) -> f32 {
        let continent = self.continent.get_noise_2d(x, z);
        let base = continent * 40.0 + 10.0;
        let hills = self.hills.get_noise_2d(x, z) * 14.0;
        let (wx, wz) = self.warp.domain_warp_2d(x, z);
        let ridge = self.ridges.get_noise_2d(wx, wz) * 0.5 + 0.5;
        base + hills + ridge * ridge * 110.0 * Self::mountain_mask(continent)
    }

    /// `around` holds the heights at x-1, x+1, z-1 and z+1.
    fn column_from(&self, x: f32, z: f32, h: f32, around: [f32; 4]) -> Column {
        let [hx0, hx1, hz0, hz1] = around;
        let g = Vec2::new(hx1 - hx0, hz1 - hz0) * 0.5;
        let slope = g.length();
        let continent = self.continent.get_noise_2d(x, z);
        Column {
            height: h,
            slope,
            inv_grad: 1.0 / (1.0 + slope * slope).sqrt(),
            mountain: Self::mountain_mask(continent),
        }
    }

    fn column(&self, x: f32, z: f32) -> Column {
        let h = self.height(x, z);
        let (hx0, hx1) = (self.height(x - 1.0, z), self.height(x + 1.0, z));
        let (hz0, hz1) = (self.height(x, z - 1.0), self.height(x, z + 1.0));
        self.column_from(x, z, h, [hx0, hx1, hz0, hz1])
    }

    /// Density and material at a point, given its column.
    fn point(&self, p: Vec3, col: &Column) -> (f32, u8) {
        let surface = (p.y - col.height) * col.inv_grad;
        if surface > DENSITY_CLAMP + 8.0 {
            return (DENSITY_CLAMP, Material::Air as u8);
        }
        let mut d = surface;
        if col.mountain > 0.0 && surface.abs() < 12.0 {
            d += self.overhang.get_noise_3d(p.x, p.y, p.z) * 4.5 * col.mountain;
        }
        if p.y < col.height + 6.0 {
            let a = self.cave_a.get_noise_3d(p.x, p.y * 1.6, p.z);
            let b = self.cave_b.get_noise_3d(p.x, p.y * 1.6, p.z);
            // Tunnels widen within a few metres of the surface so they break through as an opening
            // instead of leaving a thin crust that looks like a floating slab.
            let depth = col.height - p.y;
            let widen = 0.06 * (1.0 - depth / 6.0).clamp(0.0, 1.0);
            let cave = (a.abs().max(b.abs()) - 0.075 - widen) * 40.0;
            // Smooth subtraction rounds off cave mouths, which a 1 m grid can't show as sharp creases.
            d = smooth_max(d, -cave, 2.5);
        }
        let d = d.clamp(-DENSITY_CLAMP, DENSITY_CLAMP);
        if d >= 0.0 {
            return (d, Material::Air as u8);
        }
        (d, self.material(p, col) as u8)
    }

    fn material(&self, p: Vec3, col: &Column) -> Material {
        let depth = col.height - p.y;
        let beach = col.height < SEA_LEVEL + 1.5;
        if depth > 5.0 {
            if self.ore.get_noise_3d(p.x, p.y, p.z) > 0.72 {
                return Material::CopperOre;
            }
            return Material::Rock;
        }
        if col.slope > 1.1 {
            return Material::Rock;
        }
        if depth < 1.5 {
            if col.height > SNOW_LINE && col.mountain > 0.5 && col.slope < 0.8 {
                Material::Snow
            } else if beach {
                Material::Sand
            } else {
                Material::Grass
            }
        } else if beach {
            Material::Sand
        } else {
            Material::Soil
        }
    }

    /// Density and material at any world point. Matches [`Self::generate`] exactly at sample positions.
    pub fn sample(&self, p: Vec3) -> (f32, u8) {
        let col = self.column(p.x, p.z);
        self.point(p, &col)
    }

    /// Generate a chunk's padded samples (no edits applied).
    pub fn generate(&self, coord: ChunkCoord) -> ChunkData {
        // Heights for every sample column plus a one-sample border for the gradient.
        const H: usize = SAMPLES + 2;
        let base = ChunkData::sample_pos(coord, 0, 0, 0);
        let mut heights = vec![0.0f32; H * H];
        for hz in 0..H {
            for hx in 0..H {
                let x = base.x + hx as f32 - 1.0;
                let z = base.z + hz as f32 - 1.0;
                heights[hx + H * hz] = self.height(x, z);
            }
        }
        let mut columns = Vec::with_capacity(SAMPLES * SAMPLES);
        let mut max_height = f32::MIN;
        for z in 0..SAMPLES {
            for x in 0..SAMPLES {
                let (hx, hz) = (x + 1, z + 1);
                let h = |dx: usize, dz: usize| heights[(hx + dx - 1) + H * (hz + dz - 1)];
                let col = self.column_from(
                    base.x + x as f32,
                    base.z + z as f32,
                    h(1, 1),
                    [h(0, 1), h(2, 1), h(1, 0), h(1, 2)],
                );
                max_height = max_height.max(col.height);
                columns.push(col);
            }
        }

        let n = SAMPLES * SAMPLES * SAMPLES;
        let mut chunk = ChunkData {
            coord,
            density: vec![DENSITY_CLAMP; n],
            material: vec![Material::Air as u8; n],
            surface: columns.iter().map(|c| c.height).collect(),
            uniform: None,
        };
        // Whole chunk far above the terrain: leave it as air.
        if base.y > max_height + DENSITY_CLAMP + 20.0 {
            chunk.uniform = Some(crate::chunk::Uniform::Air);
            return chunk;
        }
        for z in 0..SAMPLES {
            for y in 0..SAMPLES {
                for x in 0..SAMPLES {
                    let col = &columns[x + SAMPLES * z];
                    let p = base + Vec3::new(x as f32, y as f32, z as f32);
                    let i = ChunkData::index(x, y, z);
                    (chunk.density[i], chunk.material[i]) = self.point(p, col);
                }
            }
        }
        chunk.update_uniform();
        chunk
    }
}

// Keep the chunk size visible here: worldgen feature scales assume 1 m cells.
const _: () = assert!(CHUNK == 32);
