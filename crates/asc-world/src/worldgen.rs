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
use crate::volcano::{self, FeatureKind, Volcano, VolcanoKind};

/// Surfaces within a few metres of this height are sandy shores. Deeper basins are lowland
/// meadows rather than vast sand plains (there is no water yet to make them seas).
pub const SEA_LEVEL: f32 = 2.0;
const SHORE_BELOW: f32 = 4.0;
const SHORE_ABOVE: f32 = 1.0;
/// Temperature falls by this much per metre above 20 m, so mountains grow cold belts (taiga, then
/// tundra, then arctic snow) whatever the climate below them.
pub(crate) const LAPSE: f32 = 0.0034;

/// Rivers: their surfaces step down in terraces this tall (each step a waterfall).
const RIVER_STEP: f32 = 3.0;
/// A river runs only where it would cut no deeper than this into the land (beyond, it fades: high
/// mountains have springs, not rivers through their peaks).
const RIVER_MAX_CUT: f32 = 45.0;
/// Lakes: one chance per cell this wide, kept well inside it so a lake only touches its own cell.
const LAKE_CELL: f32 = 640.0;
const LAKE_CHANCE: f32 = 0.32;
const LAKE_RADIUS: [f32; 2] = [30.0, 120.0];
/// Glaciers: in cold country, ice fills the land up to this far over its smooth base level (so
/// valleys fill and ridges stand clear; on flat arctic ground it's an ice sheet), at most
/// `ICE_MAX` thick. Fully glaciated below `ICE_COLD` (lapsed temperature at the ice's surface),
/// none above `ICE_WARM`: the snout of a valley glacier is where it warms up downhill.
const ICE_ABOVE: f32 = 16.0;
const ICE_MAX: f32 = 70.0;
const ICE_COLD: f32 = 0.07;
const ICE_WARM: f32 = 0.15;
/// Snow lies on glaciers colder than this (their upper, accumulating reaches); bare ice below.
const ICE_SNOW: f32 = 0.1;
/// Ice this thick or more holds caves; crevasses open in ice thicker than `CREVASSE_ICE`.
const ICE_CAVE_ICE: f32 = 14.0;
const CREVASSE_ICE: f32 = 8.0;

/// A lake basin: a bowl carved below `level`, which sits just under the lowest point of its rim so
/// the water can't spill.
#[derive(Clone, Copy, Debug)]
struct Lake {
    center: Vec2,
    radius: f32,
    depth: f32,
    level: f32,
}

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
    /// 0..1 how bare the ground is (volcanic ash and fresh rock, glacier ice): plants thin out on it.
    pub barren: f32,
    /// 0..1 how volcanic the ground is (1 on a volcano proper): veins favour it differently.
    pub volcanic: f32,
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
    /// The river network: rivers run along this noise's zero lines.
    river: FastNoiseLite,
    /// The lake in each lake cell, if any (worked out once).
    lakes: RwLock<AHashMap<IVec2, Option<Lake>>>,
    /// The volcano cone in each cone cell and the caldera in each caldera cell, if any.
    cones: RwLock<AHashMap<IVec2, Option<Arc<Volcano>>>>,
    calderas: RwLock<AHashMap<IVec2, Option<Arc<Volcano>>>>,
}

/// A column's shape: its height (the top of any ice), natural water and lava, volcanic ground,
/// and glacier ice.
#[derive(Clone, Copy, Debug)]
struct Shape {
    height: f32,
    water: Option<f32>,
    lava: Option<f32>,
    volcanic: volcano::Surface,
    /// Glacier ice over the rock (the top `ice` metres under `height`), and how deep a crevasse
    /// cuts into it here.
    ice: f32,
    crevasse: f32,
    /// Snow on the ice's surface (a glacier's colder reaches).
    snow: bool,
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
    /// Natural water surface over the column, if any (sea, lake or river).
    water: Option<f32>,
    /// Natural lava surface over the column (a crater or a caldera's pool), if any.
    lava: Option<f32>,
    volcanic: volcano::Surface,
    /// Glacier ice at the top of the column, a crevasse's depth, and snow on the ice.
    ice: f32,
    crevasse: f32,
    ice_snow: bool,
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
            river: noise(seed.wrapping_add(15), OpenSimplex2, 0.00065, Some((FBm, 2))),
            lakes: RwLock::new(AHashMap::new()),
            cones: RwLock::new(AHashMap::new()),
            calderas: RwLock::new(AHashMap::new()),
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

    /// The climate at (x, z) at sea level: temperature and moisture, 0..1.
    pub fn climate_at(&self, x: f32, z: f32) -> (f32, f32) {
        self.climate(x, z)
    }

    /// 1 in hot, dry desert, fading to 0 at its edges.
    fn desert_mask(t: f32, m: f32) -> f32 {
        smoothstep(0.6, 0.7, t) * (1.0 - smoothstep(0.24, 0.32, m))
    }

    /// Terrain height at (x, z), before any 3D features (rivers and lakes carved in).
    pub fn height(&self, x: f32, z: f32) -> f32 {
        self.shape(x, z).0
    }

    /// The natural water surface over column (x, z), if any: the sea where the ground is below sea
    /// level, otherwise a lake or river.
    pub fn water_level(&self, x: f32, z: f32) -> Option<f32> {
        self.shape(x, z).1
    }

    /// Height, natural water and lava, and the glacier's surface material (snow or ice; `None`
    /// without ice) together, in one evaluation (for the far terrain).
    pub fn height_water_lava(&self, x: f32, z: f32) -> (f32, Option<f32>, Option<f32>, Option<Material>) {
        let s = self.shape_full(x, z);
        let ice = (s.ice > 0.3).then_some(if s.snow { Material::Snow } else { Material::Ice });
        (s.height, s.water, s.lava, ice)
    }

    /// How thick the glacier ice is at (x, z).
    pub fn ice_thickness(&self, x: f32, z: f32) -> f32 {
        self.shape_full(x, z).ice
    }

    /// Height and natural water surface together (one evaluation instead of two).
    pub fn height_and_water(&self, x: f32, z: f32) -> (f32, Option<f32>) {
        self.shape(x, z)
    }

    /// Height with rivers and lakes carved in, and the water surface over it.
    fn shape(&self, x: f32, z: f32) -> (f32, Option<f32>) {
        let s = self.shape_full(x, z);
        (s.height, s.water)
    }

    /// The land, then any volcano built on it, then rivers and lakes carved in.
    fn shape_full(&self, x: f32, z: f32) -> Shape {
        let raw = self.raw_height(x, z);
        let shaped = self.volcano_at(Vec2::new(x, z)).and_then(|v| v.shape(Vec2::new(x, z), raw, |p| self.hills.get_noise_2d(p.x, p.y)));
        let (raw, volcanic_water, lava, volcanic) = match shaped {
            Some(s) => (s.height, s.water, s.lava, s.surface),
            None => (raw, None, None, volcano::Surface::default()),
        };
        // Rivers and lakes keep off the volcano itself (they'd cut its crater or flood its floor).
        let (h, river) = if volcanic.strength > 0.3 { (raw, None) } else { self.carve_river(x, z, raw) };
        let (h, lake) = if volcanic.strength > 0.0 { (h, None) } else { self.carve_lake(x, z, h) };
        let water = if h < SEA_LEVEL && volcanic_water.is_none() {
            Some(SEA_LEVEL)
        } else {
            [river, lake, volcanic_water].into_iter().flatten().filter(|l| *l > h).reduce(f32::max)
        };
        let (ice, crevasse, snow) = if volcanic.strength > 0.0 || h < SEA_LEVEL { (0.0, 0.0, false) } else { self.glacier(x, z, h) };
        // Rivers and lakes under a glacier are frozen into it.
        let water = water.filter(|_| ice < 0.5);
        Shape { height: h + ice, water, lava: lava.filter(|l| *l > h), volcanic, ice, crevasse, snow }
    }

    /// Glacier ice over land at `h`: its thickness, a crevasse's depth, and whether snow lies on it.
    fn glacier(&self, x: f32, z: f32, h: f32) -> (f32, f32, bool) {
        let base = self.continent.get_noise_2d(x, z) * 40.0 + 10.0 + self.hills.get_noise_2d(x, z) * 7.0;
        let top = base + ICE_ABOVE;
        if top <= h {
            return (0.0, 0.0, false);
        }
        let (t0, _) = self.climate(x, z);
        let t = t0 - (top - 20.0).max(0.0) * LAPSE;
        let cold = smoothstep(ICE_WARM, ICE_COLD, t);
        if cold <= 0.0 {
            return (0.0, 0.0, false);
        }
        let ice = ((top - h) * cold).min(ICE_MAX);
        // Crevasses: narrow cracks along the lines of a ridged noise, across thick ice.
        let mut crevasse = 0.0;
        if ice > CREVASSE_ICE {
            let line = 1.0 - (self.dunes.get_noise_2d(x * 1.8 + 7000.0, z * 0.7) * 0.5 + 0.5);
            let crack = smoothstep(0.07, 0.02, line);
            crevasse = crack * ice * 0.75 * smoothstep(CREVASSE_ICE, CREVASSE_ICE * 2.0, ice);
        }
        (ice, crevasse, t < ICE_SNOW)
    }

    /// The volcano (cone or caldera) whose ground holds (x, z), if any.
    fn volcano_at(&self, p: Vec2) -> Option<Arc<Volcano>> {
        let caldera = self.caldera_in((p / volcano::CALDERA_CELL).floor().as_ivec2());
        if let Some(c) = caldera.as_ref().filter(|c| c.center.distance(p) < c.radius * 1.3) {
            return Some(c.clone());
        }
        self.cone_in((p / volcano::CONE_CELL).floor().as_ivec2()).filter(|c| c.center.distance(p) < c.radius * 1.3)
    }

    fn caldera_in(&self, cell: IVec2) -> Option<Arc<Volcano>> {
        if let Some(v) = self.calderas.read().unwrap().get(&cell) {
            return v.clone();
        }
        let v = volcano::place_caldera(
            self.seed,
            cell,
            |p| self.raw_height(p.x, p.y),
            |p| self.hills.get_noise_2d(p.x, p.y),
        )
        .map(Arc::new);
        self.calderas.write().unwrap().insert(cell, v.clone());
        v
    }

    fn cone_in(&self, cell: IVec2) -> Option<Arc<Volcano>> {
        if let Some(v) = self.cones.read().unwrap().get(&cell) {
            return v.clone();
        }
        let center = (cell.as_vec2() + Vec2::splat(0.5)) * volcano::CONE_CELL;
        let caldera = self.caldera_in((center / volcano::CALDERA_CELL).floor().as_ivec2());
        let v = volcano::place_cone(self.seed, cell, |p| self.raw_height(p.x, p.y), caldera.as_deref()).map(Arc::new);
        self.cones.write().unwrap().insert(cell, v.clone());
        v
    }

    /// Every volcano with its centre within `reach` of `p` (cones and calderas).
    pub fn volcanoes_near(&self, p: Vec2, reach: f32) -> Vec<Arc<Volcano>> {
        let mut out = Vec::new();
        for (cell_size, caldera) in [(volcano::CALDERA_CELL, true), (volcano::CONE_CELL, false)] {
            let lo = ((p - Vec2::splat(reach)) / cell_size).floor().as_ivec2();
            let hi = ((p + Vec2::splat(reach)) / cell_size).floor().as_ivec2();
            for cz in lo.y..=hi.y {
                for cx in lo.x..=hi.x {
                    let cell = IVec2::new(cx, cz);
                    let v = if caldera { self.caldera_in(cell) } else { self.cone_in(cell) };
                    if let Some(v) = v.filter(|v| v.center.distance(p) <= reach) {
                        out.push(v);
                    }
                }
            }
        }
        out
    }

    /// The natural lava surface over (x, z), if any (a crater's lava lake or a caldera's pool).
    pub fn lava_level(&self, x: f32, z: f32) -> Option<f32> {
        self.shape_full(x, z).lava
    }

    /// Whether the natural water at (x, z) is a hot spring.
    pub fn hot_spring(&self, x: f32, z: f32) -> bool {
        let p = Vec2::new(x, z);
        self.volcano_at(p).is_some_and(|v| {
            v.features.iter().any(|f| f.kind == volcano::FeatureKind::HotSpring && f.at.distance(p) < f.radius)
        })
    }

    /// A river's surface at (x, z): sloping smoothly with a smoothed copy of the land, stepping
    /// down in `RIVER_STEP` terraces (waterfalls) only where that land is steep.
    fn river_level(&self, x: f32, z: f32) -> f32 {
        let smooth = |x: f32, z: f32| self.continent.get_noise_2d(x, z) * 40.0 + 10.0;
        let here = smooth(x, z);
        let e = 3.0;
        let slope = Vec2::new(smooth(x + e, z) - smooth(x - e, z), smooth(x, z + e) - smooth(x, z - e)).length()
            / (2.0 * e);
        let gentle = here - 2.0;
        let stepped = ((here - 2.0) / RIVER_STEP).floor() * RIVER_STEP;
        let steep = smoothstep(0.24, 0.38, slope);
        (gentle + (stepped - gentle) * steep).max(SEA_LEVEL)
    }

    /// How the natural water at (x, z) flows: the river's flow where its water is a river's, zero on
    /// the sea, lakes and dry land.
    pub fn water_flow(&self, x: f32, z: f32) -> Vec2 {
        let raw = self.raw_height(x, z);
        let (h, river) = self.carve_river(x, z, raw);
        let (h, _) = self.carve_lake(x, z, h);
        if h >= SEA_LEVEL && river.is_some() && self.shape(x, z).1 == river {
            self.river_flow(x, z)
        } else {
            Vec2::ZERO
        }
    }

    /// Which way a river at (x, z) flows, as a unit vector scaled by how steeply its land falls
    /// (zero off rivers): along the channel, towards the lower ground.
    pub fn river_flow(&self, x: f32, z: f32) -> Vec2 {
        let e = 2.0;
        let n = |x: f32, z: f32| self.river.get_noise_2d(x, z);
        let s = |x: f32, z: f32| self.continent.get_noise_2d(x, z) * 40.0;
        let gn = Vec2::new(n(x + e, z) - n(x - e, z), n(x, z + e) - n(x, z - e));
        let gs = Vec2::new(s(x + e, z) - s(x - e, z), s(x, z + e) - s(x, z - e)) / (2.0 * e);
        // The channel runs across the noise's gradient; pick the way that goes downhill.
        let along = Vec2::new(-gn.y, gn.x).normalize_or_zero();
        let along = if along.dot(gs) > 0.0 { -along } else { along };
        along * (0.3 + gs.length() * 40.0).min(2.0)
    }

    /// Rivers run along the zero lines of a low-frequency noise, wider and deeper on low ground.
    /// The surface steps down a smoothed copy of the land in terraces (each step a waterfall), the
    /// channel is carved below it, the banks are cut back to just above it (steeply in dry country:
    /// canyons) and raised to it where the land is lower (levees), so water never stands above the
    /// ground beside it.
    fn carve_river(&self, x: f32, z: f32, h: f32) -> (f32, Option<f32>) {
        if h < SEA_LEVEL - 1.0 {
            return (h, None);
        }
        let continent = self.continent.get_noise_2d(x, z);
        let smooth = continent * 40.0 + 10.0;
        let lowness = smoothstep(110.0, 10.0, smooth);
        let width = 0.011 + 0.012 * lowness;
        let (t, m) = self.climate(x, z);
        let arid = Self::desert_mask(t, m).max(smoothstep(0.5, 0.25, m) * smoothstep(0.55, 0.7, t));
        let banks = width * (4.0 - 2.4 * arid);
        let a = self.river.get_noise_2d(x, z).abs();
        if a >= banks {
            return (h, None);
        }
        let level = self.river_level(x, z);
        let cut = h - level;
        // Fade out where the river would have to cut too deep (it starts as a spring).
        let strength = 1.0 - smoothstep(RIVER_MAX_CUT * 0.75, RIVER_MAX_CUT, cut);
        if strength <= 0.0 {
            return (h, None);
        }
        let depth = 1.2 + 1.8 * lowness;
        let carved = if a < width {
            // The channel: deepest in the middle, up to the waterline at the edges.
            let edge = smoothstep(width * 0.55, width, a);
            (level - depth) + (depth - 0.2) * edge
        } else {
            // The banks: cut back to just above the water near it, back to the land further out.
            let t = smoothstep(width, banks, a);
            let mut bank = (level + 0.5) + (h - level - 0.5) * t;
            bank = bank.min(h);
            // Levees: never lower than the water right beside the channel, including the next
            // terrace up where the river steps down (or the upper pool spills round its waterfall).
            let levee = 1.0 - smoothstep(width * 1.3, width * 1.9, a);
            let mut wall = level;
            if levee > 0.0 {
                for (dx, dz) in [(2.0, 0.0), (-2.0, 0.0), (0.0, 2.0), (0.0, -2.0)] {
                    wall = wall.max(self.river_level(x + dx, z + dz));
                }
            }
            bank.max(h.min(wall + 0.5) + (wall + 0.5 - h.min(wall + 0.5)) * levee)
        };
        let out = h + (carved - h) * strength;
        let water = (a < width && strength > 0.5).then_some(level);
        (out, water)
    }

    /// The lake of the lake cell holding (x, z), if any (cached).
    fn lake_in(&self, cell: IVec2) -> Option<Lake> {
        if let Some(l) = self.lakes.read().unwrap().get(&cell) {
            return *l;
        }
        let lake = self.place_lake(cell);
        self.lakes.write().unwrap().insert(cell, lake);
        lake
    }

    fn place_lake(&self, cell: IVec2) -> Option<Lake> {
        let mut h = (self.seed as u32 as u64) ^ 0x6c61_6b65;
        let mut next = || {
            h ^= (cell.x as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((cell.y as u32 as u64) << 32);
            h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(29) ^ 0x94D0_49BB_1331_11EB;
            (h >> 40) as f32 / (1u64 << 24) as f32
        };
        if next() >= LAKE_CHANCE {
            return None;
        }
        let margin = LAKE_RADIUS[1] * 1.25;
        let span = LAKE_CELL - 2.0 * margin;
        let center = cell.as_vec2() * LAKE_CELL + Vec2::splat(margin) + Vec2::new(next(), next()) * span;
        let radius = LAKE_RADIUS[0] + (LAKE_RADIUS[1] - LAKE_RADIUS[0]) * next();
        let depth = 3.0 + 9.0 * next();
        let ground = |p: Vec2| {
            let raw = self.raw_height(p.x, p.y);
            self.carve_river(p.x, p.y, raw).0
        };
        let mid = ground(center);
        let mountain = Self::mountain_mask(self.continent.get_noise_2d(center.x, center.y));
        if self.volcano_at(center).is_some() {
            return None;
        }
        if !(SEA_LEVEL + 3.0..150.0).contains(&mid) || mountain > 0.6 {
            return None;
        }
        // The rim: the lowest of 24 points just outside the bowl sets the water level.
        let rim = (0..24)
            .map(|i| {
                let a = i as f32 / 24.0 * std::f32::consts::TAU;
                ground(center + Vec2::new(a.cos(), a.sin()) * radius * 1.15)
            })
            .fold(f32::MAX, f32::min);
        let level = rim - 0.6;
        // A basin worth the name: the middle well below the rim, and not a cliff-side perch.
        if level < SEA_LEVEL + 1.5 || mid > level + 8.0 {
            return None;
        }
        Some(Lake { center, radius, depth, level })
    }

    fn carve_lake(&self, x: f32, z: f32, h: f32) -> (f32, Option<f32>) {
        let p = Vec2::new(x, z);
        let cell = (p / LAKE_CELL).floor().as_ivec2();
        let Some(lake) = self.lake_in(cell) else { return (h, None) };
        let d = p.distance(lake.center) / lake.radius;
        if d >= 1.15 {
            return (h, None);
        }
        if d < 1.0 {
            let bowl = lake.level - lake.depth * (1.0 - d * d);
            let carved = h.min(bowl);
            return (carved, (carved < lake.level).then_some(lake.level));
        }
        // A lip round the bowl: never below the water at its inner edge (the rim samples can miss a
        // dip between them), easing back to the land at 1.15 radii.
        let lip = (lake.level + 0.3) + (h - lake.level - 0.3) * smoothstep(1.0, 1.15, d);
        (lip, None)
    }

    /// The land before rivers and lakes.
    fn raw_height(&self, x: f32, z: f32) -> f32 {
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
    fn column_from(&self, x: f32, z: f32, shape: Shape, around: [f32; 4]) -> Column {
        let (h, water) = (shape.height, shape.water);
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
            water,
            lava: shape.lava,
            volcanic: shape.volcanic,
            ice: shape.ice,
            crevasse: shape.crevasse,
            ice_snow: shape.snow,
        }
    }

    fn column(&self, x: f32, z: f32) -> Column {
        let shape = self.shape_full(x, z);
        let (hx0, hx1) = (self.height(x - 1.0, z), self.height(x + 1.0, z));
        let (hz0, hz1) = (self.height(x, z - 1.0), self.height(x, z + 1.0));
        self.column_from(x, z, shape, [hx0, hx1, hz0, hz1])
    }

    /// The surface at (x, z): height, slope, biome, climate and how wooded it is.
    pub fn ground(&self, x: f32, z: f32) -> Ground {
        let col = self.column(x, z);
        let (biome, forest, temperature, moisture) = self.biome(x, z, col.height, col.mountain);
        // Woods thin out on steep ground.
        let forest = forest * (1.0 - smoothstep(0.75, 1.1, col.slope));
        let underwater = col.water.is_some_and(|l| l > col.height + 0.3) || col.lava.is_some();
        // Nothing grows on glaciers either.
        let barren = if col.ice > 0.5 { 1.0 } else { col.volcanic.barren };
        let volcanic = col.volcanic.strength;
        Ground { height: col.height, slope: col.slope, biome, forest, temperature, moisture, underwater, barren, volcanic }
    }

    /// The material showing at the surface at (x, z).
    pub fn surface_material(&self, x: f32, z: f32) -> Material {
        let col = self.column(x, z);
        Material::from_u8(self.material(Vec3::new(x, col.height - 0.5, z), &col, &[]))
    }

    /// The surface material at (x, z) given its height and slope, which the far terrain already
    /// has from its own grid (so this skips the five height evaluations of a full column).
    pub fn surface_material_at(&self, x: f32, z: f32, height: f32, slope: f32, water: Option<f32>) -> Material {
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
            water,
            lava: None,
            volcanic: self.volcanic_surface(x, z, height),
            ice: 0.0,
            crevasse: 0.0,
            ice_snow: false,
        };
        Material::from_u8(self.material(Vec3::new(x, height - 0.5, z), &col, &[]))
    }

    /// The volcanic ground at (x, z) (for the far terrain's materials), its land at `h`.
    fn volcanic_surface(&self, x: f32, z: f32, h: f32) -> volcano::Surface {
        let p = Vec2::new(x, z);
        self.volcano_at(p)
            .and_then(|v| v.shape(p, h, |q| self.hills.get_noise_2d(q.x, q.y)))
            .map_or_else(volcano::Surface::default, |s| s.surface)
    }

    /// Density and material at a point, given its column.
    fn point(&self, p: Vec3, col: &Column, veins: &[Vein]) -> (f32, u8) {
        let surface = (p.y - col.height) * col.inv_grad;
        if surface > DENSITY_CLAMP + 8.0 {
            return (DENSITY_CLAMP, Material::Air as u8);
        }
        let mut d = surface;
        let depth = col.height - p.y;
        if col.ice > 0.0 && depth < col.ice {
            // Crevasses: a slot down into the ice.
            if depth < col.crevasse {
                d = d.max((col.crevasse - depth).min(1.5));
            }
            // Ice caves: tunnels through thick ice, wider and smoother than the rock's.
            if col.ice > ICE_CAVE_ICE && depth > 3.0 && depth < col.ice - 1.5 {
                let a = self.cave_a.get_noise_3d(p.x * 0.6, p.y * 0.9, p.z * 0.6);
                let b = self.cave_b.get_noise_3d(p.x * 0.6, p.y * 0.9, p.z * 0.6);
                let room = smoothstep(3.0, 6.0, depth) * smoothstep(1.5, 4.0, col.ice - depth);
                let tunnel = (a.abs().max(b.abs()) - 0.11 * room) * 25.0;
                d = smooth_max(d, -tunnel, 2.5);
            }
            let d = d.clamp(-DENSITY_CLAMP, DENSITY_CLAMP);
            if d >= 0.0 {
                return (d, Material::Air as u8);
            }
            let m = if col.ice_snow && depth < 0.6 { Material::Snow } else { Material::Ice };
            return (d, m as u8);
        }
        // Under any ice, the rock's own depth.
        let depth = depth - col.ice;
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
        if col.volcanic.strength > 0.0
            && let Some(m) = self.volcanic_material(p, col, depth)
        {
            return m;
        }
        // River and lake beds (and the sea floor near the shore) are sand.
        if let Some(level) = col.water
            && level > col.height
            && depth < 2.5
            && level - col.height < 6.0
        {
            return Material::Sand;
        }
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

    /// Volcanic ground: basalt under the cones and calderas, obsidian round lava, ash on active
    /// cones and caldera floors, pumice high on dormant cones, sulfur and sinter round vents.
    /// `None` leaves the ordinary material (a dormant cone's lower flanks are grown over).
    fn volcanic_material(&self, p: Vec3, col: &Column, depth: f32) -> Option<Material> {
        let v = &col.volcanic;
        if let Some(lava) = v.lava_edge
            && (p.y - lava).abs() < 3.5
            && depth < 4.0
        {
            return Some(Material::Obsidian);
        }
        let proper = v.strength > 0.5;
        if depth > 5.0 || col.slope > 1.1 {
            return proper.then_some(Material::Basalt);
        }
        if depth >= 1.5 {
            return match v.kind {
                Some(VolcanoKind::Active | VolcanoKind::Super) if proper => Some(Material::Ash),
                _ => proper.then_some(Material::Basalt),
            };
        }
        let patch = col.patch;
        match (v.kind?, v.near) {
            (VolcanoKind::Super, Some(FeatureKind::HotSpring | FeatureKind::Fumarole)) => Some(Material::Sulfur),
            (VolcanoKind::Super, Some(FeatureKind::Geyser)) => Some(Material::Pumice),
            (VolcanoKind::Super, _) if v.d < 0.62 => Some(if patch > 0.7 { Material::Sulfur } else { Material::Ash }),
            (VolcanoKind::Super, _) if v.d < 0.8 => Some(Material::Basalt),
            (VolcanoKind::Active, _) if v.strength > 0.4 => Some(if patch > 0.6 { Material::Basalt } else { Material::Ash }),
            (VolcanoKind::Dormant, _) if v.d < 0.45 => Some(if patch > 0.5 { Material::Pumice } else { Material::Basalt }),
            _ => None,
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
        let mut shapes = Vec::with_capacity(H * H);
        for hz in 0..H {
            for hx in 0..H {
                let x = base.x + hx as f32 - 1.0;
                let z = base.z + hz as f32 - 1.0;
                shapes.push(self.shape_full(x, z));
            }
        }
        let heights: Vec<f32> = shapes.iter().map(|s| s.height).collect();
        let mut columns = Vec::with_capacity(SAMPLES * SAMPLES);
        let mut max_height = f32::MIN;
        for z in 0..SAMPLES {
            for x in 0..SAMPLES {
                let (hx, hz) = (x + 1, z + 1);
                let h = |dx: usize, dz: usize| heights[(hx + dx - 1) + H * (hz + dz - 1)];
                let col = self.column_from(
                    base.x + x as f32,
                    base.z + z as f32,
                    shapes[hx + H * hz],
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
            water: columns.iter().map(|c| c.water.unwrap_or(f32::NAN)).collect(),
            lava: columns.iter().map(|c| c.lava.unwrap_or(f32::NAN)).collect(),
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
