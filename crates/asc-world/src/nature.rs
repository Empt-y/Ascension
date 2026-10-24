//! Where nature grows: which trees, shrubs, plants, grass and rocks stand where, from
//! `assets/data/nature.toml`'s densities and model lists.
//!
//! Placement is a pure function of the seed: each terrain column (`COLUMN` metres square) and kind
//! gets a jittered grid of candidates, sized from that kind's highest density, and each candidate
//! is kept with the probability its biome's density gives (trees also thin out towards the edges of
//! woods). The client draws what grows here (`asc-game`'s scatter); the server needs it too, so
//! animals walk round the trunks and rocks the player sees.

use std::collections::BTreeMap;
use std::sync::OnceLock;

use glam::{IVec2, Vec3};
use serde::Deserialize;

use crate::{Biome, WorldGen};

const EMBEDDED: &str = include_str!("../../../assets/data/nature.toml");

/// Side of a placement column (metres).
pub const COLUMN: f32 = 32.0;

/// How a kind stops things passing through it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Collide {
    /// A capsule up the trunk.
    Trunk,
    /// A sphere.
    Rock,
    None,
}

/// One `[[model]]` of `nature.toml`.
#[derive(Clone, Debug, Deserialize)]
pub struct ModelSpec {
    pub id: String,
    pub kind: String,
    pub biomes: Vec<String>,
}

#[derive(Deserialize)]
struct File {
    kinds: BTreeMap<String, RawKind>,
    density: BTreeMap<String, BTreeMap<String, f32>>,
    model: Vec<ModelSpec>,
}

/// The placement half of a `[kinds.*]` entry (the drawing half is the client's business).
#[derive(Deserialize)]
struct RawKind {
    scale: [f32; 2],
    max_slope: f32,
    collide: Collide,
}

/// A kind (tree, grass...) and where it grows.
#[derive(Clone, Debug)]
pub struct Kind {
    pub name: String,
    pub scale: [f32; 2],
    pub max_slope: f32,
    pub collide: Collide,
    /// Side of a candidate cell: one candidate per cell at the kind's highest density (0: grows
    /// nowhere).
    pub cell: f32,
    /// Instances per square metre, by biome (`Biome::ALL` order).
    pub density: [f32; Biome::ALL.len()],
    /// By biome: the models (indices into `Nature::models`) it may be, in file order.
    pub models: [Vec<u16>; Biome::ALL.len()],
}

/// Every kind (in name order) and model.
#[derive(Clone, Debug)]
pub struct Nature {
    pub kinds: Vec<Kind>,
    pub models: Vec<ModelSpec>,
}

/// One thing growing: which model (index into `Nature::models`), where its base is, and how it's
/// turned and scaled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grown {
    pub model: u16,
    pub position: Vec3,
    pub yaw: f32,
    pub scale: f32,
}

impl Nature {
    /// The `nature.toml` built in.
    pub fn builtin() -> &'static Nature {
        static NATURE: OnceLock<Nature> = OnceLock::new();
        NATURE.get_or_init(|| Nature::parse(EMBEDDED).expect("assets/data/nature.toml"))
    }

    pub fn parse(text: &str) -> Result<Self, String> {
        let file: File = toml::from_str(text).map_err(|e| format!("parsing nature.toml: {e}"))?;
        Self::from_parts(
            file.kinds.iter().map(|(name, k)| (name.as_str(), k.scale, k.max_slope, k.collide)),
            &file.density,
            file.model,
        )
    }

    /// From a file's pieces: each kind's name (in name order), scale range, slope limit and
    /// collider; the densities by biome then kind; and the models.
    pub fn from_parts<'a>(
        kinds: impl IntoIterator<Item = (&'a str, [f32; 2], f32, Collide)>,
        density: &BTreeMap<String, BTreeMap<String, f32>>,
        models: Vec<ModelSpec>,
    ) -> Result<Self, String> {
        let kinds: Vec<_> = kinds.into_iter().collect();
        for biome in density.keys() {
            Biome::from_name(biome).ok_or_else(|| format!("density: unknown biome {biome}"))?;
        }
        for m in &models {
            if !kinds.iter().any(|k| k.0 == m.kind) {
                return Err(format!("{}: unknown kind {}", m.id, m.kind));
            }
            for b in &m.biomes {
                Biome::from_name(b).ok_or_else(|| format!("{}: unknown biome {b}", m.id))?;
            }
        }
        let kinds = kinds
            .into_iter()
            .map(|(name, scale, max_slope, collide)| {
                let mut d = [0.0; Biome::ALL.len()];
                let mut list: [Vec<u16>; Biome::ALL.len()] = Default::default();
                for (i, biome) in Biome::ALL.into_iter().enumerate() {
                    d[i] = density.get(biome.name()).and_then(|d| d.get(name)).copied().unwrap_or(0.0);
                    for (m, spec) in models.iter().enumerate() {
                        if spec.kind == name && spec.biomes.iter().any(|b| b == biome.name()) {
                            list[i].push(m as u16);
                        }
                    }
                }
                let max = d.iter().copied().fold(0.0f32, f32::max);
                Kind {
                    name: name.to_owned(),
                    scale,
                    max_slope,
                    collide,
                    cell: if max > 0.0 { 1.0 / max.sqrt() } else { 0.0 },
                    density: d,
                    models: list,
                }
            })
            .collect();
        Ok(Self { kinds, models })
    }

    /// Index of the kind called `name`.
    pub fn kind(&self, name: &str) -> Option<usize> {
        self.kinds.iter().position(|k| k.name == name)
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
    fn next(&mut self) -> f32 {
        self.0 = splitmix(self.0);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// What of `kind` grows in `column`. Pure: the same inputs always give the same instances.
pub fn grow(nature: &Nature, seed: u64, worldgen: &WorldGen, column: IVec2, kind: usize) -> Vec<Grown> {
    let k = &nature.kinds[kind];
    let mut out = Vec::new();
    if k.cell <= 0.0 {
        return out;
    }
    let n = (COLUMN / k.cell).ceil().max(1.0) as i32;
    let cell = COLUMN / n as f32;
    let origin = column.as_vec2() * COLUMN;
    let woodland = |b: Biome| matches!(b, Biome::Forest | Biome::Taiga | Biome::Jungle | Biome::Swamp);
    for j in 0..n {
        for i in 0..n {
            let gx = (column.x * n + i) as i64 as u64;
            let gz = (column.y * n + j) as i64 as u64;
            let mut r = Rng(splitmix(seed ^ splitmix(gx ^ splitmix(gz ^ ((kind as u64) << 56)))));
            let x = origin.x + (i as f32 + r.next()) * cell;
            let z = origin.y + (j as f32 + r.next()) * cell;
            let roll = r.next();
            let pick = r.next();
            let yaw = r.next() * std::f32::consts::TAU;
            let scale = k.scale[0] + (k.scale[1] - k.scale[0]) * r.next();
            let g = worldgen.ground(x, z);
            if g.underwater || g.slope > k.max_slope {
                continue;
            }
            let b = Biome::ALL.iter().position(|x| *x == g.biome).unwrap_or(0);
            let mut density = k.density[b];
            if k.name == "tree" && woodland(g.biome) {
                density *= 0.15 + 0.85 * g.forest;
            } else if k.name == "grass" && woodland(g.biome) {
                // Less grass in the shade of woods.
                density *= 1.0 - 0.6 * g.forest;
            }
            // Bare volcanic ground (ash, fresh rock) grows little.
            density *= 1.0 - g.barren;
            if roll >= density * cell * cell {
                continue;
            }
            let models = &k.models[b];
            if models.is_empty() {
                continue;
            }
            let model = models[((pick * models.len() as f32) as usize).min(models.len() - 1)];
            // The height field is the surface except where caves break through it (cave mouths):
            // check the real density, and leave out anything that would float or be buried.
            let base = Vec3::new(x, g.height, z);
            if worldgen.sample(base + Vec3::Y * 0.25).0 <= 0.0 || worldgen.sample(base - Vec3::Y * 0.25).0 >= 0.0 {
                continue;
            }
            out.push(Grown { model, position: base, yaw, scale });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_builtin_file_parses_and_trees_grow_in_woods() {
        let nature = Nature::builtin();
        let tree = nature.kind("tree").expect("trees");
        assert_eq!(nature.kinds[tree].collide, Collide::Trunk);
        let worldgen = WorldGen::new(1337);
        let grown: usize = (0..40).map(|c| grow(nature, 7, &worldgen, IVec2::new(c * 5 - 100, c * 3 - 60), tree).len()).sum();
        assert!(grown > 20, "only {grown} trees in 40 columns");
    }
}
