//! Wild creatures: herds of animals on land and schools of fish in the water, near the player.
//!
//! Where they live comes from `assets/data/creatures.toml`. The world is split into `CELL`-metre
//! cells; each cell may hold one herd on land and one in its water, decided by a hash of the seed
//! and the cell and the chances its biome gives each kind. So the same herds are always in the
//! same places: they're spawned when the player comes within `SPAWN_REACH` and forgotten beyond
//! `KEEP_REACH`, and aren't saved.
//!
//! Until there's AI they keep to themselves: land animals stand where they are, switching between
//! their idle clips every few seconds, and swimmers circle their spot (its radius, speed and phase
//! are part of the creature, so its position at any tick is known to both sides without being
//! sent).

use std::collections::BTreeMap;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use asc_world::{Biome, World};
use glam::{IVec2, Vec2, Vec3};
use serde::{Deserialize, Serialize};

const EMBEDDED: &str = include_str!("../../../assets/data/creatures.toml");

/// Cells this many metres across hold at most one herd on land and one in the water.
pub const CELL: f32 = 96.0;
/// Herds are spawned with their cell's middle within this of the player...
const SPAWN_REACH: f32 = 180.0;
/// ...and forgotten beyond this.
const KEEP_REACH: f32 = 240.0;
/// How often the cells round the player are looked at again, and acts changed (ticks).
const SPAWN_EVERY: u64 = 20;
const ACTS_EVERY: u64 = 5;
/// A land animal keeps an idle clip this long (seconds) before choosing again.
const ACT_SECONDS: [f32; 2] = [6.0, 15.0];
const TICKS_PER_SECOND: f32 = 20.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Habitat {
    Land,
    /// Rivers and lakes.
    Fresh,
    Sea,
}

/// A kind of creature, as `creatures.toml` describes it.
#[derive(Clone, Debug)]
pub struct Kind {
    pub id: String,
    /// The models it may be drawn with, the leader's first if it has one (see `Creature::model`).
    pub models: Vec<String>,
    pub has_leader: bool,
    pub habitat: Habitat,
    pub herd: [u32; 2],
    pub spread: f32,
    pub scale: f32,
    /// Land: the idle clips. Swimmers: their one swimming clip.
    pub clips: Vec<String>,
    pub speed: f32,
    pub depth: f32,
    pub min_height: f32,
    pub max_slope: f32,
    /// Chance per cell, by biome (`Biome::ALL` order); water kinds have the same everywhere.
    pub chance: [f32; Biome::ALL.len()],
}

#[derive(Deserialize)]
struct File {
    creature: Vec<Raw>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    id: String,
    models: Vec<String>,
    leader: Option<String>,
    habitat: Habitat,
    herd: [u32; 2],
    spread: f32,
    #[serde(default = "one")]
    scale: f32,
    #[serde(default)]
    idle: Vec<String>,
    swim: Option<String>,
    #[serde(default)]
    speed: f32,
    #[serde(default)]
    depth: f32,
    #[serde(default)]
    chance: f32,
    #[serde(default)]
    biomes: BTreeMap<String, f32>,
    #[serde(default = "neg")]
    min_height: f32,
    #[serde(default = "slope")]
    max_slope: f32,
}

fn one() -> f32 {
    1.0
}
fn neg() -> f32 {
    f32::MIN
}
fn slope() -> f32 {
    0.6
}

/// Every kind, from `creatures.toml`.
pub fn kinds() -> &'static [Kind] {
    static KINDS: OnceLock<Vec<Kind>> = OnceLock::new();
    KINDS.get_or_init(|| parse(EMBEDDED).expect("assets/data/creatures.toml"))
}

pub fn parse(text: &str) -> Result<Vec<Kind>> {
    let file: File = toml::from_str(text).context("parsing creatures.toml")?;
    let mut out = Vec::new();
    for r in file.creature {
        let mut chance = [0.0; Biome::ALL.len()];
        if r.habitat == Habitat::Land {
            for (name, c) in &r.biomes {
                let Some(i) = Biome::ALL.iter().position(|b| b.name() == name) else {
                    bail!("{}: no biome called {name}", r.id);
                };
                chance[i] = *c;
            }
        } else {
            chance = [r.chance; Biome::ALL.len()];
        }
        let clips = match (&r.swim, r.habitat) {
            (Some(s), Habitat::Fresh | Habitat::Sea) => vec![s.clone()],
            (None, Habitat::Land) if !r.idle.is_empty() => r.idle.clone(),
            _ => bail!("{}: land animals need `idle` clips, swimmers a `swim` one", r.id),
        };
        if r.herd[0] == 0 || r.herd[1] < r.herd[0] || r.models.is_empty() {
            bail!("{}: herd or models", r.id);
        }
        let mut models = r.models.clone();
        if let Some(l) = &r.leader {
            models.insert(0, l.clone());
        }
        out.push(Kind {
            id: r.id,
            models,
            has_leader: r.leader.is_some(),
            habitat: r.habitat,
            herd: r.herd,
            spread: r.spread,
            scale: r.scale,
            clips,
            speed: r.speed,
            depth: r.depth,
            min_height: r.min_height,
            max_slope: r.max_slope,
            chance,
        });
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CreatureId(pub u64);

/// A swimmer's circle round its spot.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Circle {
    pub radius: f32,
    /// Metres per second (negative: the other way round).
    pub speed: f32,
    pub phase: f32,
}

/// A creature as both sides know it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Creature {
    pub id: CreatureId,
    /// Index into `kinds()`.
    pub kind: u16,
    /// Index into its kind's `models`.
    pub model: u8,
    /// Where it stands (a swimmer: the middle of its circle).
    pub at: Vec3,
    pub yaw: f32,
    /// Index into its kind's `clips`, and the tick it started.
    pub act: u8,
    pub since: u64,
    pub circle: Option<Circle>,
}

impl Creature {
    pub fn kind(&self) -> &'static Kind {
        &kinds()[self.kind as usize]
    }

    /// Where it is at `ticks` and which way it faces (a swimmer moves round its circle).
    pub fn pose(&self, ticks: u64) -> (Vec3, f32) {
        self.pose_at(ticks as f32 / TICKS_PER_SECOND)
    }

    /// `pose` at `t` seconds of game time (for drawing between ticks).
    pub fn pose_at(&self, t: f32) -> (Vec3, f32) {
        match self.circle {
            None => (self.at, self.yaw),
            Some(c) => {
                let a = c.phase + c.speed * t / c.radius.max(0.5);
                let p = self.at + Vec3::new(a.cos(), 0.0, a.sin()) * c.radius;
                // Facing along the circle.
                let dir = Vec2::new(-a.sin(), a.cos()) * c.speed.signum();
                (p, yaw_facing(dir))
            }
        }
    }
}

/// The yaw that turns a model (which faces +Z, as glTF's do) to face `dir` (x, z): models are
/// turned as parts are (see `asc_render::PartInstance::yaw`).
pub fn yaw_facing(dir: Vec2) -> f32 {
    dir.x.atan2(dir.y)
}

fn hash(seed: i32, a: i64, b: i64, salt: u64) -> u64 {
    let mut h = (seed as u32 as u64) ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for v in [a, b] {
        h ^= v as u64;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(29) ^ 0x94D0_49BB_1331_11EB;
    }
    h ^ (h >> 31)
}

fn unit(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// The herds in one cell (deterministic): its land herd and its water one, either possibly empty.
pub fn herds_in(world: &World, seed: i32, cell: IVec2) -> Vec<Creature> {
    let wg = &world.worldgen;
    let centre = (cell.as_vec2() + 0.5) * CELL;
    let g = wg.ground(centre.x, centre.y);
    let b = Biome::ALL.iter().position(|x| *x == g.biome).unwrap_or(0);
    let mut out = Vec::new();
    for habitat in [Habitat::Land, Habitat::Fresh, Habitat::Sea] {
        let h = hash(seed, cell.x as i64, cell.y as i64, habitat as u64 + 1);
        // One roll per habitat, against the kinds' chances in turn.
        let mut roll = unit(h);
        let Some((k, kind)) = kinds().iter().enumerate().filter(|(_, k)| k.habitat == habitat).find(|(_, k)| {
            roll -= k.chance[b];
            roll < 0.0
        }) else {
            continue;
        };
        let jitter = Vec2::new(unit(h.rotate_left(13)) - 0.5, unit(h.rotate_left(27)) - 0.5) * CELL * 0.6;
        let Some(middle) = find_spot(world, kind, centre + jitter, h) else { continue };
        let count = kind.herd[0] + (unit(h.rotate_left(41)) * (kind.herd[1] - kind.herd[0] + 1) as f32) as u32;
        let count = count.min(kind.herd[1]);
        for m in 0..count {
            let mh = hash(seed, cell.x as i64 * 64 + m as i64, cell.y as i64, 0x6d65_6d62 ^ k as u64);
            let a = unit(mh) * std::f32::consts::TAU;
            let r = kind.spread * unit(mh.rotate_left(17)).sqrt();
            let spot = middle + Vec2::from_angle(a) * r;
            let Some(at) = place(world, kind, spot, mh) else { continue };
            let model = if kind.has_leader && m == 0 {
                0
            } else {
                let first = usize::from(kind.has_leader);
                let n = kind.models.len() - first;
                (first + ((unit(mh.rotate_left(5)) * n as f32) as usize).min(n - 1)) as u8
            };
            let circle = (kind.habitat != Habitat::Land).then(|| Circle {
                radius: 1.5 + kind.spread * 0.5 * unit(mh.rotate_left(23)),
                speed: kind.speed * (0.7 + 0.6 * unit(mh.rotate_left(31))) * if mh & 1 == 0 { 1.0 } else { -1.0 },
                phase: unit(mh.rotate_left(37)) * std::f32::consts::TAU,
            });
            out.push(Creature {
                id: CreatureId(hash(seed, cell.x as i64, cell.y as i64, 0x6964 ^ ((k as u64) << 8) ^ m as u64)),
                kind: k as u16,
                model,
                at,
                yaw: unit(mh.rotate_left(47)) * std::f32::consts::TAU,
                act: (unit(mh.rotate_left(53)) * kind.clips.len() as f32) as u8 % kind.clips.len() as u8,
                since: 0,
                circle,
            });
        }
    }
    out
}

/// A spot in or round `near` that suits `kind` (dry ground for land animals, water deep enough for
/// swimmers): `near` itself, or one of a few tries round it.
fn find_spot(world: &World, kind: &Kind, near: Vec2, h: u64) -> Option<Vec2> {
    (0..6).map(|i| {
        let a = unit(h.rotate_left(7 * i + 3)) * std::f32::consts::TAU;
        near + if i == 0 { Vec2::ZERO } else { Vec2::from_angle(a) * CELL * 0.3 }
    })
    .find(|p| place(world, kind, *p, h).is_some())
}

/// Where a member of `kind` goes at (x, z): on the ground, or under the water's surface; `None`
/// if it doesn't suit.
fn place(world: &World, kind: &Kind, p: Vec2, h: u64) -> Option<Vec3> {
    let wg = &world.worldgen;
    let ground = wg.ground(p.x, p.y);
    let water = asc_world::fluid::water_level(wg, p.x, p.y).filter(|l| *l > ground.height + 0.2);
    match kind.habitat {
        Habitat::Land => {
            if water.is_some() || ground.slope > kind.max_slope || ground.height < kind.min_height {
                return None;
            }
            Some(crate::rules::find_ground(world, p.x, p.y))
        }
        Habitat::Fresh | Habitat::Sea => {
            let level = water?;
            let sea = (level - asc_world::SEA_LEVEL).abs() < 0.5;
            if sea != (kind.habitat == Habitat::Sea) || level - ground.height < kind.depth {
                return None;
            }
            // Somewhere between a metre under the surface and a metre off the bottom.
            let depth = 1.0 + (level - ground.height - 2.0).max(0.0) * unit(h.rotate_left(19));
            Some(Vec3::new(p.x, level - depth, p.y))
        }
    }
}

/// The server's creatures: the herds of the cells near the player.
#[derive(Default)]
pub struct Creatures {
    seed: i32,
    /// By cell: its creatures.
    cells: BTreeMap<[i32; 2], Vec<Creature>>,
    /// When each creature next changes what it's doing.
    next_act: BTreeMap<CreatureId, u64>,
}

/// What changed this tick, for the clients.
#[derive(Debug, Default)]
pub struct Changes {
    pub spawned: Vec<Creature>,
    pub gone: Vec<CreatureId>,
    pub acts: Vec<(CreatureId, u8, u64)>,
}

impl Creatures {
    pub fn new(seed: i32) -> Self {
        Self { seed, ..Default::default() }
    }

    pub fn all(&self) -> impl Iterator<Item = &Creature> {
        self.cells.values().flatten()
    }

    /// Bring the herds near `player` in, forget the far ones, and move idle animals on to their
    /// next clip when it's time.
    pub fn tick(&mut self, world: &World, player: Vec3, ticks: u64) -> Changes {
        let mut changes = Changes::default();
        if ticks.is_multiple_of(SPAWN_EVERY) {
            let here = Vec2::new(player.x, player.z);
            let reach = (SPAWN_REACH / CELL).ceil() as i32 + 1;
            let home = (here / CELL).floor().as_ivec2();
            for dz in -reach..=reach {
                for dx in -reach..=reach {
                    let cell = home + IVec2::new(dx, dz);
                    let centre = (cell.as_vec2() + 0.5) * CELL;
                    if centre.distance(here) > SPAWN_REACH || self.cells.contains_key(&cell.to_array()) {
                        continue;
                    }
                    let mut herd = herds_in(world, self.seed, cell);
                    for c in &mut herd {
                        c.since = ticks;
                        self.next_act.insert(c.id, ticks + self.act_length(c.id, ticks));
                    }
                    changes.spawned.extend(herd.iter().copied());
                    self.cells.insert(cell.to_array(), herd);
                }
            }
            let gone: Vec<[i32; 2]> = self
                .cells
                .keys()
                .filter(|c| ((IVec2::from_array(**c).as_vec2() + 0.5) * CELL).distance(here) > KEEP_REACH)
                .copied()
                .collect();
            for cell in gone {
                for c in self.cells.remove(&cell).unwrap_or_default() {
                    self.next_act.remove(&c.id);
                    changes.gone.push(c.id);
                }
            }
        }
        if ticks.is_multiple_of(ACTS_EVERY) {
            for c in self.cells.values_mut().flatten() {
                let kind = c.kind();
                if kind.clips.len() < 2 || self.next_act.get(&c.id).is_some_and(|t| *t > ticks) {
                    continue;
                }
                let h = hash(self.seed, c.id.0 as i64, ticks as i64, 0x0061_6374);
                // Something other than what it's doing now.
                let n = kind.clips.len() as u8;
                c.act = (c.act + 1 + (unit(h) * (n - 1) as f32) as u8 % (n - 1)) % n;
                c.since = ticks;
                changes.acts.push((c.id, c.act, ticks));
                let length = Self::act_length_of(self.seed, c.id, ticks);
                self.next_act.insert(c.id, ticks + length);
            }
        }
        changes
    }

    fn act_length(&self, id: CreatureId, ticks: u64) -> u64 {
        Self::act_length_of(self.seed, id, ticks)
    }

    fn act_length_of(seed: i32, id: CreatureId, ticks: u64) -> u64 {
        let u = unit(hash(seed, id.0 as i64, ticks as i64, 0x006c_656e));
        ((ACT_SECONDS[0] + (ACT_SECONDS[1] - ACT_SECONDS[0]) * u) * TICKS_PER_SECOND) as u64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_data_file_parses_and_names_real_biomes() {
        let k = kinds();
        assert!(k.len() >= 8);
        let cattle = k.iter().find(|k| k.id == "cattle").unwrap();
        assert!(cattle.has_leader && cattle.models[0] == "bull");
        assert!(k.iter().filter(|k| k.habitat == Habitat::Land).all(|k| k.chance.iter().any(|c| *c > 0.0)));
        assert!(parse("[[creature]]\nid='x'\nmodels=['x']\nhabitat='land'\nherd=[1,1]\nspread=1.0\nidle=['Idle']\nbiomes={moon=0.1}").is_err());
    }

    #[test]
    fn herds_are_the_same_every_time_and_only_where_they_live() {
        let world = World::new(1337);
        let (mut land, mut water) = (0, 0);
        for z in -12..12 {
            for x in -12..12 {
                let cell = IVec2::new(x, z);
                let herd = herds_in(&world, 1337, cell);
                assert_eq!(herd, herds_in(&world, 1337, cell), "not deterministic");
                for c in &herd {
                    let kind = c.kind();
                    let p = Vec2::new(c.at.x, c.at.z);
                    let g = world.worldgen.ground(p.x, p.y);
                    let level = asc_world::fluid::water_level(&world.worldgen, p.x, p.y);
                    match kind.habitat {
                        Habitat::Land => {
                            land += 1;
                            // On the ground, not in water.
                            assert!(level.is_none_or(|l| l <= g.height + 0.2), "{} in the water at {p}", kind.id);
                            assert!((c.at.y - world.worldgen.height(p.x, p.y)).abs() < 3.0, "{} floating at {}", kind.id, c.at);
                            let centre = (cell.as_vec2() + 0.5) * CELL;
                            let biome = world.worldgen.ground(centre.x, centre.y).biome;
                            let b = Biome::ALL.iter().position(|x| *x == biome).unwrap();
                            assert!(kind.chance[b] > 0.0, "{} in {biome:?}", kind.id);
                        }
                        _ => {
                            water += 1;
                            let l = level.expect("a swimmer out of the water");
                            assert!(c.at.y < l && c.at.y > g.height, "{} at {} (water {l}, bed {})", kind.id, c.at.y, g.height);
                            assert!(l - g.height >= kind.depth);
                        }
                    }
                }
            }
        }
        assert!(land > 20, "only {land} land animals in 576 cells");
        assert!(water > 0, "no swimmers in 576 cells");
    }

    #[test]
    fn herds_come_near_and_go_far_and_animals_change_what_they_do() {
        let world = World::new(1337);
        let mut creatures = Creatures::new(1337);
        let mut seen = std::collections::BTreeSet::new();
        let mut acts = 0;
        let mut player = Vec3::new(0.0, 0.0, 0.0);
        for t in 0..20 * 60 {
            // Walk east at 10 m/s.
            player.x = t as f32 * 0.5;
            let ch = creatures.tick(&world, player, t);
            for c in &ch.spawned {
                assert!(seen.insert(c.id), "spawned twice");
                assert!(Vec2::new(c.at.x - player.x, c.at.z - player.z).length() < SPAWN_REACH + CELL);
            }
            for id in &ch.gone {
                assert!(seen.contains(id));
            }
            acts += ch.acts.len();
            for c in creatures.all() {
                assert!(Vec2::new(c.at.x - player.x, c.at.z - player.z).length() < KEEP_REACH + CELL);
            }
        }
        assert!(!seen.is_empty());
        assert!(acts > 0, "nobody changed what they were doing in a minute");
    }

    #[test]
    fn a_swimmer_circles_its_spot() {
        let c = Creature {
            id: CreatureId(1),
            kind: kinds().iter().position(|k| k.habitat == Habitat::Fresh).unwrap() as u16,
            model: 0,
            at: Vec3::new(10.0, -2.0, 5.0),
            yaw: 0.0,
            act: 0,
            since: 0,
            circle: Some(Circle { radius: 3.0, speed: 1.5, phase: 0.0 }),
        };
        for t in [0, 37, 400] {
            let (p, _) = c.pose(t);
            assert!(((p - c.at).length() - 3.0).abs() < 1e-3 && p.y == c.at.y);
        }
        assert_ne!(c.pose(0).0, c.pose(20).0);
    }
}
