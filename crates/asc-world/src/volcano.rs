//! Volcanoes: rare cones, dormant or active, and a few supervolcano calderas per seed.
//!
//! Placed like lakes: one chance per cell of a coarse grid, kept well inside its cell so each one
//! only ever touches its own. A cone is a concave stratovolcano with radial gullies and a summit
//! crater: an active one holds a lava lake, a dormant one sometimes a crater lake. A supervolcano
//! never erupts: it's a broad rim round a sunken caldera floor holding a lake, hot springs, geyser
//! mounds, fumaroles and small lava pools.

use glam::Vec2;

/// Cones: one chance per cell this wide.
pub const CONE_CELL: f32 = 3072.0;
const CONE_CHANCE: f32 = 0.15;
const CONE_RADIUS: [f32; 2] = [450.0, 800.0];
const CONE_HEIGHT: [f32; 2] = [220.0, 420.0];
/// How many cones are active (the rest are dormant).
const ACTIVE_SHARE: f32 = 0.45;
/// Calderas: one chance per cell this wide (about 2–3 per 16 km region).
pub const CALDERA_CELL: f32 = 8192.0;
const CALDERA_CHANCE: f32 = 0.6;
const CALDERA_RADIUS: [f32; 2] = [1800.0, 2400.0];
/// The caldera's rim and floor, as fractions of its radius.
const RIM: f32 = 0.62;
const FLOOR: f32 = 0.55;
const RIM_HEIGHT: f32 = 90.0;
/// The caldera floor sits this far below the land it was built on (at its middle: it tilts by
/// `FLOOR_TILT` across its radius), and its lake fills the low side to `CALDERA_LAKE` below.
const FLOOR_DROP: f32 = 15.0;
const FLOOR_TILT: f32 = 12.0;
const CALDERA_LAKE: f32 = 16.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum VolcanoKind {
    Dormant,
    Active,
    Super,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum FeatureKind {
    /// A warm pool.
    HotSpring,
    /// A mound of sinter round a vent that erupts now and then.
    Geyser,
    /// A small pool of lava welling up through the floor.
    LavaPool,
    /// A vent breathing steam and sulfur: a crust round it.
    Fumarole,
}

/// A feature on a caldera floor.
#[derive(Clone, Copy, Debug)]
pub struct Feature {
    pub kind: FeatureKind,
    pub at: Vec2,
    pub radius: f32,
    /// The floor's height there (pools fill to it, mounds rise from it).
    pub ground: f32,
}

#[derive(Clone, Debug)]
pub struct Volcano {
    pub kind: VolcanoKind,
    pub center: Vec2,
    /// Where the cone (or the caldera's outer slope) meets the land.
    pub radius: f32,
    /// The land it was built on, at its centre.
    pub base: f32,
    /// Summit height above `base` (cones), or the rim's (calderas).
    pub height: f32,
    pub crater_radius: f32,
    pub crater_floor: f32,
    /// A dormant crater's lake.
    pub crater_lake: Option<f32>,
    /// An active crater's lava lake.
    pub lava: Option<f32>,
    pub features: Vec<Feature>,
    /// Which way a caldera's floor falls (unit), toward its lake.
    pub tilt: Vec2,
}

/// What a volcano does to one column.
#[derive(Clone, Copy, Debug, Default)]
pub struct Shaped {
    pub height: f32,
    pub water: Option<f32>,
    pub lava: Option<f32>,
    /// Whether the water is a hot spring's.
    pub hot: bool,
    pub surface: Surface,
}

/// The volcanic ground at a column, for materials and the scatter.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Surface {
    /// 0 outside, 1 on the volcano proper.
    pub strength: f32,
    pub kind: Option<VolcanoKind>,
    /// Distance from the centre as a fraction of the radius.
    pub d: f32,
    /// A feature this column is next to.
    pub near: Option<FeatureKind>,
    /// A lava level close by (crater or pool): obsidian forms round it.
    pub lava_edge: Option<f32>,
    /// 0..1 how bare the ground is (ash, fresh rock): plants thin out.
    pub barren: f32,
}

/// A small deterministic sequence of numbers in 0..1 from the seed, a salt and a cell.
struct Rng(u64);

impl Rng {
    fn new(seed: i32, salt: u64, cell: glam::IVec2) -> Self {
        let mut h = (seed as u32 as u64) ^ salt;
        h ^= (cell.x as u32 as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ ((cell.y as u32 as u64) << 32);
        Self(h)
    }

    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(29) ^ 0x94D0_49BB_1331_11EB;
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }

    fn range(&mut self, r: [f32; 2]) -> f32 {
        r[0] + (r[1] - r[0]) * self.next()
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn smooth_max(a: f32, b: f32, k: f32) -> f32 {
    let h = (0.5 + 0.5 * (a - b) / k).clamp(0.0, 1.0);
    b + (a - b) * h + k * h * (1.0 - h)
}

/// The cone in `cell`, if any. `land` gives the land's height before volcanoes; `caldera` is the
/// caldera (if any) whose cell holds this one, which a cone must keep clear of.
pub fn place_cone(seed: i32, cell: glam::IVec2, land: impl Fn(Vec2) -> f32, caldera: Option<&Volcano>) -> Option<Volcano> {
    let mut r = Rng::new(seed, 0x766f_6c63, cell);
    if r.next() >= CONE_CHANCE {
        return None;
    }
    let radius = r.range(CONE_RADIUS);
    let margin = CONE_RADIUS[1] * 1.3;
    let center = cell.as_vec2() * CONE_CELL + Vec2::splat(margin) + Vec2::new(r.next(), r.next()) * (CONE_CELL - 2.0 * margin);
    if caldera.is_some_and(|c| c.center.distance(center) < c.radius + radius * 1.3) {
        return None;
    }
    let base = land(center);
    // Not out at sea (a volcanic island is fine) or on top of high mountains.
    if !(crate::SEA_LEVEL - 25.0..120.0).contains(&base) {
        return None;
    }
    let height = r.range(CONE_HEIGHT);
    let active = r.next() < ACTIVE_SHARE;
    let crater_radius = radius * (0.08 + 0.04 * r.next());
    let rim = base + height * (1.0 - crater_radius / radius).powf(1.6);
    let crater_depth = 35.0 + 40.0 * r.next();
    let crater_floor = rim - crater_depth;
    let (kind, lava, crater_lake) = if active {
        (VolcanoKind::Active, Some(crater_floor + 6.0 + 6.0 * r.next()), None)
    } else {
        let lake = (r.next() < 0.5).then(|| crater_floor + crater_depth * (0.25 + 0.2 * r.next()));
        (VolcanoKind::Dormant, None, lake)
    };
    Some(Volcano {
        kind,
        center,
        radius,
        base,
        height,
        crater_radius,
        crater_floor,
        crater_lake,
        lava,
        features: Vec::new(),
        tilt: Vec2::ZERO,
    })
}

/// The caldera in `cell`, if any; `floor_noise` gives the floor's hummocks (about -1..1).
pub fn place_caldera(seed: i32, cell: glam::IVec2, land: impl Fn(Vec2) -> f32, floor_noise: impl Fn(Vec2) -> f32) -> Option<Volcano> {
    let mut r = Rng::new(seed, 0x6361_6c64, cell);
    if r.next() >= CALDERA_CHANCE {
        return None;
    }
    let radius = r.range(CALDERA_RADIUS);
    let margin = CALDERA_RADIUS[1] * 1.1;
    let center = cell.as_vec2() * CALDERA_CELL + Vec2::splat(margin) + Vec2::new(r.next(), r.next()) * (CALDERA_CELL - 2.0 * margin);
    let base = land(center);
    if !(crate::SEA_LEVEL + 2.0..110.0).contains(&base) {
        return None;
    }
    let mut v = Volcano {
        kind: VolcanoKind::Super,
        center,
        radius,
        base,
        height: RIM_HEIGHT,
        crater_radius: radius * RIM,
        crater_floor: base - FLOOR_DROP,
        crater_lake: Some(base - CALDERA_LAKE),
        lava: None,
        features: Vec::new(),
        tilt: Vec2::from_angle(r.next() * std::f32::consts::TAU),
    };
    // Features on the dry floor, clear of each other and of the lake.
    let wanted = [(FeatureKind::HotSpring, 12), (FeatureKind::Geyser, 6), (FeatureKind::LavaPool, 3), (FeatureKind::Fumarole, 8)];
    for (kind, count) in wanted {
        let mut placed = 0;
        for _ in 0..count * 20 {
            if placed == count {
                break;
            }
            let a = r.next() * std::f32::consts::TAU;
            let at = center + Vec2::new(a.cos(), a.sin()) * radius * FLOOR * 0.85 * r.next().sqrt();
            let radius = match kind {
                FeatureKind::HotSpring => 4.0 + 6.0 * r.next(),
                FeatureKind::Geyser => 4.0 + 2.0 * r.next(),
                FeatureKind::LavaPool => 7.0 + 8.0 * r.next(),
                FeatureKind::Fumarole => 8.0 + 8.0 * r.next(),
            };
            let ground = floor_height(&v, at, &floor_noise);
            let lake = v.crater_lake.unwrap_or(f32::MIN);
            if ground < lake + 1.5 || v.features.iter().any(|f| f.at.distance(at) < f.radius + radius + 25.0) {
                continue;
            }
            v.features.push(Feature { kind, at, radius, ground });
            placed += 1;
        }
    }
    Some(v)
}

/// The caldera floor (before features) at `p`: hummocks on a tilt.
fn floor_height(v: &Volcano, p: Vec2, floor_noise: impl Fn(Vec2) -> f32) -> f32 {
    let tilt = (p - v.center).dot(v.tilt) / v.radius * FLOOR_TILT;
    v.crater_floor - tilt + floor_noise(p) * 4.0
}

impl Volcano {
    /// Its effect on the column at `p`, whose land is at `h` (before rivers and lakes). `noise` is
    /// a smooth noise (about -1..1) for gullies and the caldera floor's hummocks.
    pub fn shape(&self, p: Vec2, h: f32, noise: impl Fn(Vec2) -> f32) -> Option<Shaped> {
        let dist = p.distance(self.center);
        let d = dist / self.radius;
        if d >= 1.3 {
            return None;
        }
        Some(match self.kind {
            VolcanoKind::Super => self.caldera(p, d, h, noise),
            _ => self.cone(p, dist, d, h, noise),
        })
    }

    fn cone(&self, p: Vec2, dist: f32, d: f32, h: f32, noise: impl Fn(Vec2) -> f32) -> Shaped {
        let rel = p - self.center;
        let angle = rel.y.atan2(rel.x);
        // Gullies run down the flanks: noise round the cone, stretched along its slope.
        let along = Vec2::new(angle.cos(), angle.sin()) * self.radius * 0.25 + Vec2::splat(dist * 0.06);
        let gullies = noise(along) * self.height * 0.07 * (d * (1.0 - d) * 4.0).clamp(0.0, 1.0);
        let profile = (1.0 - d).max(0.0).powf(1.6);
        let mut cone = self.base + self.height * profile + gullies;
        let rc = self.crater_radius;
        let rim = self.base + self.height * (1.0 - rc / self.radius).powf(1.6);
        let in_crater = dist < rc;
        if in_crater {
            let t = dist / rc;
            cone = cone.min(self.crater_floor + (rim - self.crater_floor) * t.powf(2.5));
        }
        let height = if in_crater { cone } else { smooth_max(h, cone, 15.0) };
        let water = self.crater_lake.filter(|l| in_crater && *l > height);
        let lava = self.lava.filter(|l| in_crater && *l > height);
        let strength = 1.0 - smoothstep(0.75, 1.1, d);
        let lava_edge = self.lava.filter(|_| dist < rc * 1.25);
        let barren = match self.kind {
            VolcanoKind::Active => strength,
            _ => 1.0 - smoothstep(0.35, 0.55, d),
        };
        Shaped {
            height,
            water,
            lava,
            hot: false,
            surface: Surface { strength, kind: Some(self.kind), d, near: None, lava_edge, barren },
        }
    }

    fn caldera(&self, p: Vec2, d: f32, h: f32, noise: impl Fn(Vec2) -> f32) -> Shaped {
        let rim_top = self.base + self.height;
        let strength = 1.0 - smoothstep(0.85, 1.15, d);
        let mut surface = Surface { strength, kind: Some(self.kind), d, ..Default::default() };
        if d >= RIM {
            // The outer slope, rising gently to the rim.
            let rise = self.height * smoothstep(1.0, RIM, d).powf(1.3);
            surface.barren = 0.3 * (1.0 - smoothstep(RIM, 0.8, d));
            return Shaped { height: smooth_max(h, self.base + rise, 20.0), surface, ..Default::default() };
        }
        // The floor, and the wall down to it from the rim.
        let floor = floor_height(self, p, &noise);
        let mut height = floor + (rim_top - floor) * smoothstep(FLOOR, RIM, d);
        let mut water = self.crater_lake.filter(|l| *l > height && d < FLOOR);
        let mut lava = None;
        let mut hot = false;
        surface.barren = 0.75;
        for f in &self.features {
            let fd = p.distance(f.at) / f.radius;
            if fd >= 1.8 {
                continue;
            }
            surface.near = Some(f.kind);
            match f.kind {
                FeatureKind::HotSpring if fd < 1.0 => {
                    height = height.min(f.ground - 1.8 * (1.0 - fd * fd));
                    water = Some(f.ground - 0.2);
                    hot = true;
                }
                FeatureKind::LavaPool if fd < 1.0 => {
                    height = height.min(f.ground - 3.0 * (1.0 - fd * fd));
                    lava = Some(f.ground - 0.6);
                }
                FeatureKind::LavaPool => surface.lava_edge = Some(f.ground - 0.6),
                FeatureKind::Geyser if fd < 1.0 => {
                    // A sinter mound with a vent in its top.
                    let vent = 0.8 / f.radius;
                    height = if fd < vent { f.ground - 2.0 } else { f.ground + 2.2 * (1.0 - fd * fd) };
                }
                _ => {}
            }
        }
        let water = water.filter(|l| *l > height);
        let lava = lava.filter(|l| *l > height);
        Shaped { height, water, lava, hot: hot && water.is_some(), surface }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{WorldGen, fluid};
    use glam::{IVec3, Vec3};

    fn volcanoes(seed: i32) -> (WorldGen, Vec<std::sync::Arc<Volcano>>) {
        let wg = WorldGen::new(seed);
        let list = wg.volcanoes_near(Vec2::ZERO, 16000.0);
        (wg, list)
    }

    #[test]
    fn volcanoes_are_rare_and_mixed() {
        let mut kinds = std::collections::HashMap::new();
        let mut total = 0;
        for seed in [1337, 7, 1256132528] {
            let (_, list) = volcanoes(seed);
            total += list.len();
            for v in &list {
                *kinds.entry(v.kind).or_insert(0) += 1;
            }
        }
        // Three seeds' 32 km squares: on the order of a few cones each, and some calderas.
        let per_seed = total as f32 / 3.0;
        assert!((3.0..25.0).contains(&per_seed), "{per_seed} volcanoes per seed");
        for k in [VolcanoKind::Active, VolcanoKind::Dormant, VolcanoKind::Super] {
            assert!(kinds.get(&k).copied().unwrap_or(0) > 0, "no {k:?} volcanoes: {kinds:?}");
        }
    }

    #[test]
    fn cones_rise_to_a_crater_and_active_ones_hold_lava() {
        let (wg, list) = volcanoes(1256132528);
        let active = list.iter().find(|v| v.kind == VolcanoKind::Active).expect("an active cone");
        let c = active.center;
        let rim = wg.height(c.x + active.crater_radius, c.y);
        let floor = wg.height(c.x, c.y);
        let foot = wg.height(c.x + active.radius * 1.4, c.y);
        assert!(rim > foot + 150.0, "rim {rim} over the foot {foot}");
        assert!(floor < rim - 25.0, "crater floor {floor} under the rim {rim}");
        let lava = wg.lava_level(c.x, c.y).expect("a lava lake");
        assert!(lava > floor);
        // The lava is fluid: pristine cells under its surface, and the chunk's mesh shows it.
        let cell = IVec3::new(c.x as i32, (floor + 1.0).ceil() as i32, c.y as i32);
        assert_eq!(fluid::pristine(&wg, cell).0, fluid::FluidKind::Lava);
        let coord = crate::chunk_of(Vec3::new(c.x, lava, c.y));
        let natural = wg.generate(coord);
        let mesh = fluid::natural_mesh(&wg, &natural, &natural);
        assert!(mesh.vertices.iter().any(|v| v.weights[1] > 128), "the crater's mesh has lava");
    }

    #[test]
    fn calderas_have_springs_geysers_and_lava_pools() {
        let (wg, list) = volcanoes(1256132528);
        let caldera = list.iter().find(|v| v.kind == VolcanoKind::Super).expect("a caldera");
        for kind in [FeatureKind::HotSpring, FeatureKind::Geyser, FeatureKind::LavaPool, FeatureKind::Fumarole] {
            assert!(caldera.features.iter().any(|f| f.kind == kind), "no {kind:?}");
        }
        for f in &caldera.features {
            let p = f.at;
            match f.kind {
                FeatureKind::HotSpring => {
                    assert!(wg.water_level(p.x, p.y).is_some() && wg.hot_spring(p.x, p.y), "a dry spring at {p}")
                }
                FeatureKind::LavaPool => assert!(wg.lava_level(p.x, p.y).is_some(), "no lava in the pool at {p}"),
                FeatureKind::Geyser => assert!(wg.height(p.x + f.radius * 0.5, p.y) > f.ground + 0.5, "no mound at {p}"),
                FeatureKind::Fumarole => {}
            }
        }
        // The rim stands over the floor.
        let c = caldera.center;
        let rim = wg.height(c.x + caldera.radius * RIM, c.y);
        assert!(rim > caldera.crater_floor + 50.0, "rim {rim}, floor {}", caldera.crater_floor);
    }
}
