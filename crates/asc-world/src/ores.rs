//! Ore veins: where every ore is, as a pure function of the seed and position.
//!
//! Ores come in veins, as in GregTech. The world is split into `VEIN_CELL`-wide square cells; each
//! cell gets `VEIN_ATTEMPTS` tries, each holding a vein with probability `VEIN_CHANCE`. A vein's
//! type is a weighted pick among all the types, and it then sits at a depth within that type's
//! own range:
//!
//!   weight = abundance_ppm ^ 0.25 * affinity(biome at the surface above)
//!
//! so how common a vein is follows how common its key element is in the real crust (squashed so
//! rare ores stay findable), and deposits favour the places they really form (bauxite in the
//! tropics, salt in deserts, coal under old swamps). A vein is a tilted lens; within it the top
//! third is its primary ore, the bottom third its secondary, the middle its "between" ore, with a
//! sprinkle of its sporadic ore anywhere. Veins near the surface show small outcrops above them.

use std::collections::HashMap;
use std::sync::OnceLock;

use glam::{IVec2, Vec2, Vec3};
use serde::Deserialize;

use crate::worldgen::{Biome, WorldGen};

/// Side of a vein cell in metres.
pub const VEIN_CELL: f32 = 64.0;
/// Tries per cell, and the chance each one holds a vein.
pub const VEIN_ATTEMPTS: u32 = 2;
pub const VEIN_CHANCE: f32 = 0.65;
/// Weight exponent applied to crustal abundance.
pub const ABUNDANCE_POWER: f32 = 0.25;
/// Veins whose top comes within this many metres of the surface show outcrops above them.
pub const HINT_DEPTH: f32 = 25.0;
/// Share of a shallow vein's possible outcrops that actually show: they're rare finds, not a map.
pub const HINT_CHANCE: f32 = 0.25;

/// One ore, as listed in `ores.toml` (fields the world doesn't need are ignored).
#[derive(Clone, Debug, Deserialize)]
pub struct OreDef {
    pub id: String,
    pub name: String,
    pub formula: String,
    pub tier: u8,
    pub vein: String,
    pub style: String,
    /// Linear base colour, metallic and roughness, as in glTF.
    pub color: [f32; 3],
    #[serde(default)]
    pub metallic: f32,
    #[serde(default = "default_roughness")]
    pub roughness: f32,
    #[serde(default)]
    pub byproducts: Vec<String>,
}

fn default_roughness() -> f32 {
    0.7
}

const EMBEDDED: &str = include_str!("../../../assets/data/ores.toml");

/// The shipped ore table (`assets/data/ores.toml`, built in so worldgen can't drift from it).
pub fn builtin() -> &'static OreTable {
    static TABLE: OnceLock<OreTable> = OnceLock::new();
    TABLE.get_or_init(|| {
        let file: OreFile = toml::from_str(EMBEDDED).expect("parsing ores.toml");
        OreTable::new(file).expect("ores.toml")
    })
}

/// One vein type, as listed in `ores.toml`.
#[derive(Clone, Debug, Deserialize)]
pub struct VeinDef {
    pub id: String,
    pub abundance_ppm: f32,
    pub primary: String,
    pub secondary: String,
    pub between: String,
    pub sporadic: String,
    pub depth: [f32; 2],
    pub radius: [f32; 2],
    pub thickness: [f32; 2],
    #[serde(default)]
    pub affinity: HashMap<String, f32>,
}

/// The contents of `ores.toml`.
#[derive(Clone, Debug, Deserialize)]
pub struct OreFile {
    pub ore: Vec<OreDef>,
    pub vein: Vec<VeinDef>,
}

/// Which part of a vein an ore came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Layer {
    Primary,
    Between,
    Secondary,
    Sporadic,
}

#[derive(Clone, Debug)]
pub struct VeinType {
    pub def: VeinDef,
    /// Ore indices: primary, between, secondary, sporadic.
    pub ores: [usize; 4],
    /// Base weight from abundance.
    pub weight: f32,
    /// Multiplier per biome, in `Biome::ALL` order.
    pub affinity: [f32; Biome::ALL.len()],
    /// Affinity on volcanic ground (the `volcanic` key; 1 if not given), blended in by how
    /// volcanic the ground over the vein is.
    pub volcanic: f32,
}

/// Ores and vein types with names resolved to indices.
#[derive(Clone, Debug)]
pub struct OreTable {
    pub ores: Vec<OreDef>,
    pub veins: Vec<VeinType>,
}

impl OreTable {
    pub fn new(file: OreFile) -> Result<Self, String> {
        let index =
            |id: &str| file.ore.iter().position(|o| o.id == id).ok_or_else(|| format!("unknown ore {id}"));
        let mut veins = Vec::new();
        for def in &file.vein {
            let ores =
                [index(&def.primary)?, index(&def.between)?, index(&def.secondary)?, index(&def.sporadic)?];
            let mut affinity = [1.0; Biome::ALL.len()];
            let mut volcanic = 1.0;
            for (name, value) in &def.affinity {
                if name == "volcanic" {
                    volcanic = *value;
                    continue;
                }
                let b =
                    Biome::from_name(name).ok_or_else(|| format!("vein {}: unknown biome {name}", def.id))?;
                affinity[biome_index(b)] = *value;
            }
            if def.depth[0] > def.depth[1]
                || def.radius[0] > def.radius[1]
                || def.thickness[0] > def.thickness[1]
            {
                return Err(format!("vein {}: a range runs backwards", def.id));
            }
            veins.push(VeinType {
                def: def.clone(),
                ores,
                weight: def.abundance_ppm.max(0.0).powf(ABUNDANCE_POWER),
                affinity,
                volcanic,
            });
        }
        Ok(Self { ores: file.ore, veins })
    }

    pub fn ore_index(&self, id: &str) -> Option<usize> {
        self.ores.iter().position(|o| o.id == id)
    }

    /// The furthest any vein reaches from its centre, horizontally or vertically.
    pub fn max_reach(&self) -> f32 {
        self.veins.iter().map(|v| v.def.radius[1] + v.def.thickness[1] * 0.5).fold(0.0, f32::max)
    }

    /// The ore at `p` among `veins`, and its layer.
    pub fn ore_in(&self, veins: &[Vein], p: Vec3) -> Option<(usize, Layer)> {
        for vein in veins {
            if let Some(layer) = vein.layer_at(p) {
                let t = &self.veins[vein.kind];
                let ore = match layer {
                    Layer::Primary => t.ores[0],
                    Layer::Between => t.ores[1],
                    Layer::Secondary => t.ores[2],
                    Layer::Sporadic => t.ores[3],
                };
                return Some((ore, layer));
            }
        }
        None
    }

    /// How ore `index` looks in the terrain, packed for a vertex: sRGB-ish colour (square root
    /// of linear, so dark ores keep precision) and metallic (high nibble) / roughness (low).
    pub fn look(&self, index: usize) -> [u8; 4] {
        let o = &self.ores[index];
        let c = o.color.map(|v| (v.clamp(0.0, 1.0).sqrt() * 255.0).round() as u8);
        let nib = |v: f32| (v.clamp(0.0, 1.0) * 15.0).round() as u8;
        [c[0], c[1], c[2], (nib(o.metallic) << 4) | nib(o.roughness)]
    }

    /// The veins whose centres lie in `cell`.
    pub fn veins_in_cell(&self, worldgen: &WorldGen, cell: IVec2) -> Vec<Vein> {
        let mut out = Vec::new();
        for attempt in 0..VEIN_ATTEMPTS {
            let mut r = Rng::new(worldgen.seed(), cell, attempt);
            if r.next() >= VEIN_CHANCE {
                continue;
            }
            let x = (cell.x as f32 + r.next()) * VEIN_CELL;
            let z = (cell.y as f32 + r.next()) * VEIN_CELL;
            let ground = worldgen.ground(x, z);
            let b = biome_index(ground.biome);
            let weight = |v: &VeinType| v.weight * v.affinity[b] * (1.0 + (v.volcanic - 1.0) * ground.volcanic);
            let total: f32 = self.veins.iter().map(weight).sum();
            if total <= 0.0 {
                continue;
            }
            let mut pick = r.next() * total;
            let mut kind = self.veins.len() - 1;
            for (i, v) in self.veins.iter().enumerate() {
                let w = weight(v);
                if pick < w {
                    kind = i;
                    break;
                }
                pick -= w;
            }
            let def = &self.veins[kind].def;
            let lerp = |range: [f32; 2], t: f32| range[0] + (range[1] - range[0]) * t;
            let depth = lerp(def.depth, r.next());
            // Tilted up to about 25 degrees from flat.
            let tilt = Vec2::new(r.next() - 0.5, r.next() - 0.5) * 0.9;
            out.push(Vein {
                kind,
                center: Vec3::new(x, ground.height - depth, z),
                radius: lerp(def.radius, r.next()),
                thickness: lerp(def.thickness, r.next()),
                normal: Vec3::new(tilt.x, 1.0, tilt.y).normalize(),
                seed: r.next_u32(),
            });
        }
        out
    }

    /// The ore at `p` and which layer of its vein it's in, or `None` for plain rock.
    pub fn ore_at(&self, worldgen: &WorldGen, p: Vec3) -> Option<(usize, Layer)> {
        self.ore_in(&worldgen.veins_touching(p, p), p)
    }

    /// Outcrops on the surface above a shallow vein: (ore, ground position). Empty for veins
    /// deeper than `HINT_DEPTH`.
    pub fn surface_hints(&self, worldgen: &WorldGen, vein: &Vein) -> Vec<(usize, Vec3)> {
        let ground = worldgen.height(vein.center.x, vein.center.z);
        if ground - (vein.center.y + vein.thickness * 0.5) > HINT_DEPTH {
            return Vec::new();
        }
        let t = &self.veins[vein.kind];
        let mut r = Rng::new(vein.seed as i32, IVec2::ZERO, 7);
        let count = 1 + (vein.radius / 8.0) as usize;
        (0..count)
            .filter_map(|_| {
                let a = r.next() * std::f32::consts::TAU;
                let d = vein.radius * 0.8 * r.next().sqrt();
                let (x, z) = (vein.center.x + a.cos() * d, vein.center.z + a.sin() * d);
                let ore = if r.next() < 0.7 { t.ores[0] } else { t.ores[1] };
                (r.next() < HINT_CHANCE).then(|| (ore, Vec3::new(x, worldgen.height(x, z), z)))
            })
            .collect()
    }
}

fn biome_index(b: Biome) -> usize {
    Biome::ALL.iter().position(|x| *x == b).unwrap_or(0)
}

/// A placed vein: a tilted lens of rock.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vein {
    pub kind: usize,
    pub center: Vec3,
    /// Half-width of the lens.
    pub radius: f32,
    /// Full height at the middle; it thins to nothing at the rim.
    pub thickness: f32,
    /// Normal of the lens's plane.
    pub normal: Vec3,
    pub seed: u32,
}

impl Vein {
    /// Distance from the centre beyond which nothing is ore, on any axis.
    pub fn reach(&self) -> f32 {
        self.radius + self.thickness * 0.5
    }

    /// Which layer of the vein `p` is in, if it's ore (about half the lens is, in patches).
    pub fn layer_at(&self, p: Vec3) -> Option<Layer> {
        let d = p - self.center;
        let h = d.dot(self.normal);
        let planar = (d - self.normal * h).length();
        let edge = 1.0 - (planar / self.radius).powi(2);
        if edge <= 0.0 {
            return None;
        }
        let half = self.thickness * 0.5 * edge.sqrt();
        if h.abs() > half {
            return None;
        }
        // Patchy: ore in clumps a couple of metres across, rock between.
        let s = self.seed as f32 * 0.001;
        if value_noise(p * 0.45 + Vec3::splat(s)) < 0.42 {
            return None;
        }
        if hash3(p.floor().as_ivec3(), self.seed) < 0.1 {
            return Some(Layer::Sporadic);
        }
        let t = h / half;
        Some(if t > 1.0 / 3.0 {
            Layer::Primary
        } else if t < -1.0 / 3.0 {
            Layer::Secondary
        } else {
            Layer::Between
        })
    }
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

struct Rng(u64);

impl Rng {
    fn new(seed: i32, cell: IVec2, slot: u32) -> Self {
        let mut h = splitmix(seed as u32 as u64 ^ 0x6F72_6573);
        h = splitmix(h ^ cell.x as u32 as u64);
        h = splitmix(h ^ ((cell.y as u32 as u64) << 1));
        Self(splitmix(h ^ ((slot as u64) << 40)))
    }

    fn next_u32(&mut self) -> u32 {
        self.0 = splitmix(self.0);
        (self.0 >> 32) as u32
    }

    fn next(&mut self) -> f32 {
        (self.next_u32() >> 8) as f32 / (1u32 << 24) as f32
    }
}

fn hash3(p: glam::IVec3, seed: u32) -> f32 {
    let h =
        splitmix(splitmix(splitmix(seed as u64 ^ p.x as u32 as u64) ^ p.y as u32 as u64) ^ p.z as u32 as u64);
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// Smooth 3D value noise in [0, 1].
fn value_noise(p: Vec3) -> f32 {
    let i = p.floor();
    let f = p - i;
    let u = f * f * (Vec3::splat(3.0) - 2.0 * f);
    let c = i.as_ivec3();
    let h = |dx: i32, dy: i32, dz: i32| hash3(c + glam::IVec3::new(dx, dy, dz), 0x5eed);
    let x00 = h(0, 0, 0) + (h(1, 0, 0) - h(0, 0, 0)) * u.x;
    let x10 = h(0, 1, 0) + (h(1, 1, 0) - h(0, 1, 0)) * u.x;
    let x01 = h(0, 0, 1) + (h(1, 0, 1) - h(0, 0, 1)) * u.x;
    let x11 = h(0, 1, 1) + (h(1, 1, 1) - h(0, 1, 1)) * u.x;
    let y0 = x00 + (x10 - x00) * u.y;
    let y1 = x01 + (x11 - x01) * u.y;
    y0 + (y1 - y0) * u.z
}
