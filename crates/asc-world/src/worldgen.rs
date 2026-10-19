//! Deterministic procedural terrain.
//!
//! A 2D height field (continents, hills, domain-warped ridged mountains) gives the base surface. 3D noise
//! carves spaghetti tunnels and deep caverns underground; tunnels only reach the surface at rare
//! hillside entrances. Every value is a pure function of the seed and world position, so any chunk
//! can be regenerated on any thread at any time.

use fastnoise_lite::{DomainWarpType, FastNoiseLite, FractalType, NoiseType};
use std::sync::{Arc, RwLock};

use ahash::AHashMap;
use glam::{IVec2, Vec2, Vec3};

use crate::chunk::{CHUNK, ChunkCoord, ChunkData, SAMPLES};
use crate::edit::DENSITY_CLAMP;
use crate::material::{Material, ore_byte};
use crate::ores::{OreTable, VEIN_CELL, Vein};

/// Surfaces within a few metres of this height are sandy shores. Deeper basins are lowland
/// meadows rather than vast sand plains (there is no water yet to make them seas).
const SEA_LEVEL: f32 = 2.0;
const SHORE_BELOW: f32 = 4.0;
const SHORE_ABOVE: f32 = 1.0;
/// Temperature falls by this much per metre above 20 m, so mountains grow cold belts (taiga, then
/// tundra, then arctic snow) whatever the climate below them.
const LAPSE: f32 = 0.0034;

/// What grows where, from climate (temperature and moisture), height and slope. Surface
/// materials follow the biome, so sand, snow, moss and dry grass line up with what grows on them.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Biome {
    /// Frozen: snowfields and ice, a few snowy pines.
    Arctic,
    /// Cold and dry: moss, low shrubs, boulders, snow patches.
    Tundra,
    /// Cold and wet: dense pine forest.
    Taiga,
    /// Temperate and dry: open grassland and flowers.
    Meadow,
    /// Temperate: broadleaf woods (oak, birch, maple) with clearings.
    Forest,
    /// Temperate, wet and low-lying: mud, willows, reeds, dead trees.
    Swamp,
    /// Hot and dry: sand dunes, sandstone, cacti.
    Desert,
    /// Hot, seasonally dry: yellow grass and flat-topped acacias.
    Savanna,
    /// Hot and wet: dense broadleaf trees, palms and ferns.
    Jungle,
    /// Beaches and the strip just above the sea (outside deserts and the arctic).
    Shore,
}

impl Biome {
    pub const ALL: [Biome; 10] = [
        Biome::Arctic,
        Biome::Tundra,
        Biome::Taiga,
        Biome::Meadow,
        Biome::Forest,
        Biome::Swamp,
        Biome::Desert,
        Biome::Savanna,
        Biome::Jungle,
        Biome::Shore,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Biome::Arctic => "arctic",
            Biome::Tundra => "tundra",
            Biome::Taiga => "taiga",
            Biome::Meadow => "meadow",
            Biome::Forest => "forest",
            Biome::Swamp => "swamp",
            Biome::Desert => "desert",
            Biome::Savanna => "savanna",
            Biome::Jungle => "jungle",
            Biome::Shore => "shore",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|b| b.name() == name)
    }
}

/// Everything the scatter needs to know about a surface point.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ground {
    pub height: f32,
    /// Rise over run.
    pub slope: f32,
    pub biome: Biome,
    /// 0..1 how wooded: 0 in the open, rising through woodland edges to 1 deep in a wood.
    pub forest: f32,
    /// 0..1 after the height lapse; 0 is frozen, 1 is tropical.
    pub temperature: f32,
    pub moisture: f32,
    /// Whether the surface is under the sea.
    pub underwater: bool,
}

pub struct WorldGen {
    seed: i32,
    continent: FastNoiseLite,
    hills: FastNoiseLite,
    ridges: FastNoiseLite,
    warp: FastNoiseLite,
    cave_a: FastNoiseLite,
    cave_b: FastNoiseLite,
    cavern: FastNoiseLite,
    /// 2D mask choosing the rare places where tunnels may reach the surface.
    entrance: FastNoiseLite,
    /// Low-frequency woodland mask: woods where it's high (with clearings and edges).
    forest: FastNoiseLite,
    /// Climate: very low-frequency temperature and moisture.
    temperature: FastNoiseLite,
    moisture: FastNoiseLite,
    /// Mid-frequency patches (snow in the tundra, moss in the taiga, mud in swamps, ice sheets).
    patches: FastNoiseLite,
    /// Ridged dune field, stretched across the wind.
    dunes: FastNoiseLite,
    ores: &'static OreTable,
    /// Veins per vein cell, worked out once: every chunk and point sample near a cell reads them.
    veins: RwLock<AHashMap<IVec2, Arc<[Vein]>>>,
}

/// Per-(x, z) values shared by every sample in a column.
#[derive(Clone, Copy)]
struct Column {
    height: f32,
    /// Length of the height gradient (rise over run).
    slope: f32,
    /// 1 / |∇(y - h)|, which turns `y - h` into an approximate true distance.
    inv_grad: f32,
    /// 0 on plains, 1 in mountains.
    mountain: f32,
    /// 0..1: how freely caves may open to the surface here (1 only in rare entrance regions).
    entrance: f32,
    biome: Biome,
    /// 0..1 mid-frequency patch value, for snow, moss, mud and ice patches.
    patch: f32,
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
            cave_a: noise(seed.wrapping_add(5), OpenSimplex2, 0.012, None),
            cave_b: noise(seed.wrapping_add(6), OpenSimplex2, 0.012, None),
            cavern: noise(seed.wrapping_add(8), OpenSimplex2, 0.007, Some((FBm, 2))),
            entrance: noise(seed.wrapping_add(9), OpenSimplex2, 0.004, None),
            forest: noise(seed.wrapping_add(10), OpenSimplex2, 0.0028, Some((FBm, 3))),
            temperature: noise(seed.wrapping_add(11), OpenSimplex2, 0.00005, Some((FBm, 2))),
            moisture: noise(seed.wrapping_add(12), OpenSimplex2, 0.00008, Some((FBm, 2))),
            patches: noise(seed.wrapping_add(13), OpenSimplex2, 0.012, Some((FBm, 2))),
            dunes: noise(seed.wrapping_add(14), OpenSimplex2, 0.011, Some((Ridged, 2))),
            ores: crate::ores::builtin(),
            veins: RwLock::new(AHashMap::new()),
        }
    }

    pub fn seed(&self) -> i32 {
        self.seed
    }

    /// The ore table veins are placed from.
    pub fn ores(&self) -> &'static OreTable {
        self.ores
    }

    /// The veins centred in `cell` (cached).
    pub fn veins_in_cell(&self, cell: IVec2) -> Arc<[Vein]> {
        if let Some(v) = self.veins.read().unwrap().get(&cell) {
            return v.clone();
        }
        let v: Arc<[Vein]> = self.ores.veins_in_cell(self, cell).into();
        self.veins.write().unwrap().insert(cell, v.clone());
        v
    }

    /// Every vein that may reach into the box `lo..hi`.
    pub fn veins_touching(&self, lo: Vec3, hi: Vec3) -> Vec<Vein> {
        let reach = self.ores.max_reach();
        let c0 = (Vec2::new(lo.x, lo.z) - reach).div_euclid(Vec2::splat(VEIN_CELL)).as_ivec2();
        let c1 = (Vec2::new(hi.x, hi.z) + reach).div_euclid(Vec2::splat(VEIN_CELL)).as_ivec2();
        let mut out = Vec::new();
        for cz in c0.y..=c1.y {
            for cx in c0.x..=c1.x {
                for v in self.veins_in_cell(IVec2::new(cx, cz)).iter() {
                    let r = Vec3::splat(v.reach());
                    if (v.center + r).cmpge(lo).all() && (v.center - r).cmple(hi).all() {
                        out.push(*v);
                    }
                }
            }
        }
        out
    }

    /// Mountain mask in [0, 1] from the continent value.
    fn mountain_mask(continent: f32) -> f32 {
        smoothstep(0.05, 0.55, continent)
    }

    /// Sea-level temperature and moisture at (x, z), both 0..1.
    fn climate(&self, x: f32, z: f32) -> (f32, f32) {
        // The noise rarely reaches its extremes, so stretch it to use the whole range.
        let t = (0.5 + 0.75 * self.temperature.get_noise_2d(x, z)).clamp(0.0, 1.0);
        let m = (0.5 + 0.9 * self.moisture.get_noise_2d(x, z)).clamp(0.0, 1.0);
        (t, m)
    }

    /// 1 in hot, dry desert, fading to 0 at its edges.
    fn desert_mask(t: f32, m: f32) -> f32 {
        smoothstep(0.6, 0.7, t) * (1.0 - smoothstep(0.24, 0.32, m))
    }

    /// Terrain height at (x, z), before any 3D features.
    pub fn height(&self, x: f32, z: f32) -> f32 {
        let continent = self.continent.get_noise_2d(x, z);
        let base = continent * 40.0 + 10.0;
        let hills = self.hills.get_noise_2d(x, z) * 14.0;
        let (wx, wz) = self.warp.domain_warp_2d(x, z);
        let ridge = self.ridges.get_noise_2d(wx, wz) * 0.5 + 0.5;
        let mountain = Self::mountain_mask(continent);
        let mut h = base + hills + ridge * ridge * 110.0 * mountain;
        // Dunes in the deserts: ridges across a prevailing wind, on low ground only.
        let (t, m) = self.climate(x, z);
        let desert =
            Self::desert_mask(t, m) * (1.0 - mountain) * smoothstep(SEA_LEVEL + 1.0, SEA_LEVEL + 6.0, h);
        if desert > 0.0 {
            let d = self.dunes.get_noise_2d(x * 0.55, z) * 0.5 + 0.5;
            h += desert * d * d * 7.0;
        }
        h
    }

    /// The biome from climate, height and slope (see [`Biome`]). Also returns the woodland value,
    /// the lapsed temperature and the moisture.
    fn biome(&self, x: f32, z: f32, h: f32, mountain: f32) -> (Biome, f32, f32, f32) {
        let (t0, m) = self.climate(x, z);
        let t = (t0 - (h - 20.0).max(0.0) * LAPSE).clamp(0.0, 1.0);
        let wood = self.forest.get_noise_2d(x, z) * 0.5 + 0.5;
        let shore = h < SEA_LEVEL + SHORE_ABOVE + 1.5 && h > SEA_LEVEL - SHORE_BELOW && mountain < 0.5;
        let hot_dry = t >= 0.64 && m < 0.3;
        let (biome, forest) = if t < 0.12 {
            (Biome::Arctic, 0.15 * smoothstep(0.5, 0.75, wood))
        } else if shore && !hot_dry {
            (Biome::Shore, 0.0)
        } else if t < 0.29 {
            if m < 0.42 {
                (Biome::Tundra, 0.2 * smoothstep(0.55, 0.8, wood))
            } else {
                (Biome::Taiga, 0.55 + 0.45 * smoothstep(0.35, 0.65, wood))
            }
        } else if t < 0.64 {
            let woods = smoothstep(0.28, 0.45, m) * smoothstep(0.4, 0.58, wood);
            if m > 0.72 && h < 25.0 {
                (Biome::Swamp, 0.4 * smoothstep(0.4, 0.7, wood))
            } else if woods > 0.15 {
                (Biome::Forest, woods)
            } else {
                (Biome::Meadow, woods)
            }
        } else if m < 0.3 {
            (Biome::Desert, 0.0)
        } else if m < 0.58 {
            (Biome::Savanna, 0.25 * smoothstep(0.5, 0.75, wood))
        } else {
            (Biome::Jungle, 0.6 + 0.4 * smoothstep(0.35, 0.65, wood))
        };
        (biome, forest, t, m)
    }

    /// `around` holds the heights at x-1, x+1, z-1 and z+1.
    fn column_from(&self, x: f32, z: f32, h: f32, around: [f32; 4]) -> Column {
        let [hx0, hx1, hz0, hz1] = around;
        let g = Vec2::new(hx1 - hx0, hz1 - hz0) * 0.5;
        let slope = g.length();
        let continent = self.continent.get_noise_2d(x, z);
        let mountain = Self::mountain_mask(continent);
        let (biome, _, _, _) = self.biome(x, z, h, mountain);
        Column {
            height: h,
            slope,
            inv_grad: 1.0 / (1.0 + slope * slope).sqrt(),
            mountain,
            biome,
            patch: self.patches.get_noise_2d(x, z) * 0.5 + 0.5,
            // Cave mouths belong on hillsides; on flat ground a near-horizontal tunnel only skims the
            // surface and leaves a long slit.
            entrance: smoothstep(0.55, 0.75, self.entrance.get_noise_2d(x, z)) * smoothstep(0.3, 0.7, slope),
        }
    }

    fn column(&self, x: f32, z: f32) -> Column {
        let h = self.height(x, z);
        let (hx0, hx1) = (self.height(x - 1.0, z), self.height(x + 1.0, z));
        let (hz0, hz1) = (self.height(x, z - 1.0), self.height(x, z + 1.0));
        self.column_from(x, z, h, [hx0, hx1, hz0, hz1])
    }

    /// The surface at (x, z): height, slope, biome, climate and how wooded it is.
    pub fn ground(&self, x: f32, z: f32) -> Ground {
        let col = self.column(x, z);
        let (biome, forest, temperature, moisture) = self.biome(x, z, col.height, col.mountain);
        // Woods thin out on steep ground.
        let forest = forest * (1.0 - smoothstep(0.75, 1.1, col.slope));
        let underwater = col.height < SEA_LEVEL - 0.3;
        Ground { height: col.height, slope: col.slope, biome, forest, temperature, moisture, underwater }
    }

    /// The material showing at the surface at (x, z).
    pub fn surface_material(&self, x: f32, z: f32) -> Material {
        let col = self.column(x, z);
        Material::from_u8(self.material(Vec3::new(x, col.height - 0.5, z), &col, &[]))
    }

    /// The surface material at (x, z) given its height and slope, which the far terrain already
    /// has from its own grid (so this skips the five height evaluations of a full column).
    pub fn surface_material_at(&self, x: f32, z: f32, height: f32, slope: f32) -> Material {
        let mountain = Self::mountain_mask(self.continent.get_noise_2d(x, z));
        let (biome, _, _, _) = self.biome(x, z, height, mountain);
        let col = Column {
            height,
            slope,
            inv_grad: 1.0 / (1.0 + slope * slope).sqrt(),
            mountain,
            entrance: 0.0,
            biome,
            patch: self.patches.get_noise_2d(x, z) * 0.5 + 0.5,
        };
        Material::from_u8(self.material(Vec3::new(x, height - 0.5, z), &col, &[]))
    }

    /// Density and material at a point, given its column.
    fn point(&self, p: Vec3, col: &Column, veins: &[Vein]) -> (f32, u8) {
        let surface = (p.y - col.height) * col.inv_grad;
        if surface > DENSITY_CLAMP + 8.0 {
            return (DENSITY_CLAMP, Material::Air as u8);
        }
        let mut d = surface;
        let depth = col.height - p.y;
        if depth > 2.0 || col.entrance > 0.0 {
            // Spaghetti tunnels: where two noise fields are both near zero. They pinch shut over the
            // last ~14 m below the surface, except in entrance regions, so the surface is only broken
            // by the occasional cave mouth rather than trenches everywhere.
            let near_surface = smoothstep(4.0, 14.0, depth);
            let width = 0.085 * near_surface.max(col.entrance * smoothstep(-2.0, 2.0, depth));
            if width > 0.0 {
                let a = self.cave_a.get_noise_3d(p.x, p.y * 1.25, p.z);
                let b = self.cave_b.get_noise_3d(p.x, p.y * 1.25, p.z);
                let tunnel = (a.abs().max(b.abs()) - width) * 30.0;
                // Smooth subtraction rounds off mouths, which a 1 m grid can't show as sharp creases.
                d = smooth_max(d, -tunnel, 2.0);
            }
            // Occasional big caverns, deep down only.
            if depth > 28.0 {
                let open = smoothstep(28.0, 45.0, depth);
                let cavern =
                    (0.5 - self.cavern.get_noise_3d(p.x, p.y * 1.5, p.z)) * 25.0 + (1.0 - open) * 20.0;
                d = smooth_max(d, -cavern, 2.0);
            }
        }
        let d = d.clamp(-DENSITY_CLAMP, DENSITY_CLAMP);
        if d >= 0.0 {
            return (d, Material::Air as u8);
        }
        (d, self.material(p, col, veins))
    }

    /// The material byte of a solid point: ores replace the rock wherever a vein passes through.
    fn material(&self, p: Vec3, col: &Column, veins: &[Vein]) -> u8 {
        let base = self.base_material(p, col);
        if matches!(base, Material::Rock | Material::Sandstone)
            && let Some((ore, _)) = self.ores.ore_in(veins, p)
        {
            return ore_byte(ore);
        }
        base as u8
    }

    fn base_material(&self, p: Vec3, col: &Column) -> Material {
        let depth = col.height - p.y;
        let sandy = matches!(col.biome, Biome::Desert);
        if depth > 5.0 && !(sandy && depth < 12.0) {
            return Material::Rock;
        }
        if col.slope > 1.1 {
            return if matches!(col.biome, Biome::Desert | Biome::Savanna) {
                Material::Sandstone
            } else {
                Material::Rock
            };
        }
        if depth >= 1.5 {
            return match col.biome {
                Biome::Desert => Material::Sandstone,
                Biome::Shore => Material::Sand,
                _ => Material::Soil,
            };
        }
        let patch = col.patch;
        match col.biome {
            Biome::Arctic if col.slope < 0.12 && patch > 0.62 => Material::Ice,
            Biome::Arctic => Material::Snow,
            Biome::Tundra if patch > 0.66 && col.slope < 0.7 => Material::Snow,
            Biome::Tundra => Material::Moss,
            Biome::Taiga if patch > 0.6 => Material::Moss,
            Biome::Swamp if patch > 0.42 => Material::Mud,
            Biome::Desert | Biome::Shore => Material::Sand,
            Biome::Savanna => Material::DryGrass,
            Biome::Meadow if patch < 0.22 => Material::DryGrass,
            _ => Material::Grass,
        }
    }

    /// Density and material at any world point. Matches [`Self::generate`] exactly at sample positions.
    pub fn sample(&self, p: Vec3) -> (f32, u8) {
        let col = self.column(p.x, p.z);
        self.sample_in(p, &col)
    }

    /// [`Self::sample`] down one vertical line: the column is worked out once for every height.
    pub fn line(&self, x: f32, z: f32) -> impl Fn(f32) -> (f32, u8) + '_ {
        let col = self.column(x, z);
        move |y| self.sample_in(Vec3::new(x, y, z), &col)
    }

    fn sample_in(&self, p: Vec3, col: &Column) -> (f32, u8) {
        let (d, m) = self.point(p, col, &[]);
        // Only rock can hold ore: look the veins up just for that.
        if (m == Material::Rock as u8 || m == Material::Sandstone as u8)
            && let Some((ore, _)) = self.ores.ore_in(&self.veins_touching(p, p), p)
        {
            return (d, ore_byte(ore));
        }
        (d, m)
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
        let veins = self.veins_touching(base, base + Vec3::splat((SAMPLES - 1) as f32));
        for z in 0..SAMPLES {
            for y in 0..SAMPLES {
                for x in 0..SAMPLES {
                    let col = &columns[x + SAMPLES * z];
                    let p = base + Vec3::new(x as f32, y as f32, z as f32);
                    let i = ChunkData::index(x, y, z);
                    (chunk.density[i], chunk.material[i]) = self.point(p, col, &veins);
                }
            }
        }
        chunk.update_uniform();
        chunk
    }
}

// Keep the chunk size visible here: worldgen feature scales assume 1 m cells.
const _: () = assert!(CHUNK == 32);
