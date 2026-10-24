//! Wild creatures: herds of animals on land and schools of fish in the water, near the player.
//!
//! Where they live comes from `assets/data/creatures.toml`. The world is split into `CELL`-metre
//! cells; each cell may hold one herd on land and one in its water, decided by a hash of the seed
//! and the cell and the chances its biome gives each kind. So the same herds are always in the
//! same places: they're spawned when the player comes within `SPAWN_REACH` and forgotten beyond
//! `KEEP_REACH`. They aren't saved, but what's been killed is (`losses`): a herd comes back short
//! of the animals it lost, and grows back one a day.
//!
//! Each animal thinks for itself (`brain`): it sees and hears the player, predators, prey and
//! danger, gets hungry and thirsty and sleepy, remembers being chased, and picks what to do. Herds
//! and packs act together (`group`); animals walk the navigation grid (`nav`, `path`) and move
//! (`motion`); and they fight, are hurt and die (`combat`). The clients get where each animal is
//! a few times a second and what it's doing when that changes.

mod brain;
mod combat;
mod group;
mod motion;
#[cfg(test)]
mod tests;

use std::collections::BTreeMap;
use std::sync::OnceLock;

use anyhow::{Context, Result, bail};
use asc_world::{Biome, World};
use glam::{IVec2, Vec2, Vec3};
use serde::{Deserialize, Serialize};

use crate::nav::{Ground, NavGrid, Profile};
use crate::path::PathFinder;
use crate::sim::DAY_TICKS;
use crate::vitals::Cause;

pub use brain::Behaviour;

const EMBEDDED: &str = include_str!("../../../../assets/data/creatures.toml");

/// Cells this many metres across hold at most one herd on land and one in the water.
pub const CELL: f32 = 96.0;
/// Herds are spawned with their cell's middle within this of the player...
const SPAWN_REACH: f32 = 180.0;
/// ...and forgotten beyond this.
const KEEP_REACH: f32 = 240.0;
/// How often the cells round the player are looked at again (ticks).
const SPAWN_EVERY: u64 = 20;
/// Where animals are is sent this often (ticks), for those that moved.
const SEND_EVERY: u64 = 4;
/// Navigation tiles built per tick, at most.
const TILES_PER_TICK: usize = 2;
/// Path search cells per tick, shared by every search.
const PATH_BUDGET: usize = 6000;
/// A carcass lies this long (ticks) before it's gone.
const CARCASS_TICKS: u64 = 20 * 300;
pub const TICKS_PER_SECOND: f32 = 20.0;
/// What the player's bare fists do.
pub const FIST_DAMAGE: i32 = 60;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Habitat {
    Land,
    /// Rivers and lakes.
    Fresh,
    Sea,
}

/// When an animal is up and about.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Activity {
    Day,
    /// Dawn and dusk.
    Dusk,
    Night,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Diet {
    Graze,
    Hunt,
    /// Swimmers: they feed as they go.
    Forage,
}

/// An attack: the clip it's played with, what it does, how near the target must be, how long the
/// clip takes to land the blow (seconds) and how long before the next.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attack {
    pub clip: String,
    pub damage: i32,
    pub reach: f32,
    pub windup: f32,
    pub cooldown: f32,
}

/// What a carcass gives when butchered: an item (by id) and how many.
#[derive(Clone, Debug, PartialEq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Drop {
    pub item: String,
    pub count: [u32; 2],
}

/// A kind of creature, as `creatures.toml` describes it.
#[derive(Clone, Debug)]
pub struct Kind {
    pub id: String,
    pub name: String,
    /// "was killed by a wolf".
    pub killed: String,
    /// The models it may be drawn with, the leader's first if it has one (see `Creature::model`).
    pub models: Vec<String>,
    pub has_leader: bool,
    pub habitat: Habitat,
    pub herd: [u32; 2],
    pub spread: f32,
    pub scale: f32,
    /// Land: the idle clips. Swimmers: their one swimming clip.
    pub clips: Vec<String>,
    pub sleep_clip: String,
    pub eat_clip: String,
    pub health: i32,
    /// Life size, metres: how wide (for getting through gaps and keeping apart) and tall.
    pub radius: f32,
    pub height: f32,
    /// Speeds (m/s): walking (swimmers: cruising) and running.
    pub walk: f32,
    pub run: f32,
    /// The speeds the walk and run clips' feet keep up with, at its scale.
    pub stride: [f32; 2],
    /// How fast it turns (radians a second) and how long it can run flat out (seconds).
    pub turn: f32,
    pub stamina: f32,
    pub max_slope: f32,
    pub wade: f32,
    /// Swimmers: the water must be at least this deep (m).
    pub depth: f32,
    pub min_height: f32,
    pub sight: f32,
    pub hearing: f32,
    /// It runs from the player coming nearer than this (when it isn't wary).
    pub flee: f32,
    /// 0..1: how quickly it learns to fear the player.
    pub wariness: f32,
    /// 0..1: the chance it turns on what hurt it, rather than running.
    pub aggression: f32,
    /// The leader charges a wary herd's threat coming near (a bull with his cows).
    pub guard: bool,
    /// Comes to look at the player (calm, not too near).
    pub curious: bool,
    pub activity: Activity,
    pub diet: Diet,
    /// Predators: the kinds it hunts (by index into `kinds()`), and those only when very hungry.
    pub prey: Vec<u16>,
    pub big_prey: Vec<u16>,
    pub attack: Option<Attack>,
    pub drops: Vec<Drop>,
    /// Chance per cell, by biome (`Biome::ALL` order); water kinds have the same everywhere.
    pub chance: [f32; Biome::ALL.len()],
}

impl Kind {
    pub fn swims(&self) -> bool {
        self.habitat != Habitat::Land
    }

    /// How it gets about, for the navigation grid.
    pub fn profile(&self) -> Profile {
        if self.swims() {
            Profile { radius: self.radius.min(1.0), max_slope: f32::MAX, step: f32::MAX, water: self.depth, swims: true }
        } else {
            Profile { radius: self.radius, max_slope: self.max_slope, step: 0.45, water: self.wade, swims: false }
        }
    }

    pub fn predator(&self) -> bool {
        self.diet == Diet::Hunt
    }
}

#[derive(Deserialize)]
struct File {
    creature: Vec<Raw>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Raw {
    id: String,
    name: String,
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
    #[serde(default = "sleep_clip")]
    sleep: String,
    #[serde(default = "eat_clip")]
    eat: String,
    health: i32,
    radius: f32,
    height: f32,
    walk: f32,
    run: f32,
    stride: [f32; 2],
    #[serde(default = "turn")]
    turn: f32,
    #[serde(default = "stamina")]
    stamina: f32,
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
    #[serde(default = "wade")]
    wade: f32,
    #[serde(default = "sight")]
    sight: f32,
    #[serde(default = "one")]
    hearing: f32,
    #[serde(default = "flee")]
    flee: f32,
    #[serde(default = "half")]
    wariness: f32,
    #[serde(default)]
    aggression: f32,
    #[serde(default)]
    guard: bool,
    #[serde(default)]
    curious: bool,
    #[serde(default = "day")]
    activity: Activity,
    #[serde(default = "graze")]
    diet: Diet,
    #[serde(default)]
    prey: Vec<String>,
    #[serde(default)]
    big_prey: Vec<String>,
    attack: Option<Attack>,
    #[serde(default)]
    drops: Vec<Drop>,
}

fn one() -> f32 {
    1.0
}
fn half() -> f32 {
    0.5
}
fn neg() -> f32 {
    f32::MIN
}
fn slope() -> f32 {
    0.8
}
fn wade() -> f32 {
    0.6
}
fn sight() -> f32 {
    50.0
}
fn flee() -> f32 {
    25.0
}
fn turn() -> f32 {
    3.0
}
fn stamina() -> f32 {
    20.0
}
fn day() -> Activity {
    Activity::Day
}
fn graze() -> Diet {
    Diet::Graze
}
fn sleep_clip() -> String {
    "Idle_Headlow".into()
}
fn eat_clip() -> String {
    "Eating".into()
}

/// Every kind, from `creatures.toml`.
pub fn kinds() -> &'static [Kind] {
    static KINDS: OnceLock<Vec<Kind>> = OnceLock::new();
    KINDS.get_or_init(|| parse(EMBEDDED).expect("assets/data/creatures.toml"))
}

/// The kind called `id`.
pub fn kind_index(id: &str) -> Option<u16> {
    kinds().iter().position(|k| k.id == id).map(|i| i as u16)
}

pub fn parse(text: &str) -> Result<Vec<Kind>> {
    let file: File = toml::from_str(text).context("parsing creatures.toml")?;
    let ids: Vec<String> = file.creature.iter().map(|r| r.id.clone()).collect();
    let index = |list: &[String], of: &str| -> Result<Vec<u16>> {
        list.iter()
            .map(|p| ids.iter().position(|i| i == p).map(|i| i as u16).with_context(|| format!("{of}: no creature called {p}")))
            .collect()
    };
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
        if r.health <= 0 || r.walk <= 0.0 || r.run < r.walk || r.stride.iter().any(|s| *s <= 0.0) {
            bail!("{}: health, speeds and strides must be positive (and run at least walk)", r.id);
        }
        let mut models = r.models.clone();
        if let Some(l) = &r.leader {
            models.insert(0, l.clone());
        }
        let article = if r.name.starts_with(['a', 'e', 'i', 'o', 'u', 'A', 'E', 'I', 'O', 'U']) { "an" } else { "a" };
        out.push(Kind {
            killed: format!("was killed by {article} {}", r.name.to_lowercase()),
            prey: index(&r.prey, &r.id)?,
            big_prey: index(&r.big_prey, &r.id)?,
            id: r.id,
            name: r.name,
            models,
            has_leader: r.leader.is_some(),
            habitat: r.habitat,
            herd: r.herd,
            spread: r.spread,
            scale: r.scale,
            clips,
            sleep_clip: r.sleep,
            eat_clip: r.eat,
            health: r.health,
            radius: r.radius,
            height: r.height,
            walk: r.walk,
            run: r.run,
            stride: r.stride,
            turn: r.turn,
            stamina: r.stamina,
            max_slope: r.max_slope,
            wade: r.wade,
            depth: r.depth,
            min_height: r.min_height,
            sight: r.sight,
            hearing: r.hearing,
            flee: r.flee,
            wariness: r.wariness,
            aggression: r.aggression,
            guard: r.guard,
            curious: r.curious,
            activity: r.activity,
            diet: r.diet,
            attack: r.attack,
            drops: r.drops,
            chance,
        });
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct CreatureId(pub u64);

/// A herd (or pack, or school): the cell it lives in and its habitat, hashed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct HerdId(pub u64);

/// What an animal is doing, as far as drawing it goes (how it moves is from its speed).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Act {
    /// One of its kind's idle clips (by index); walking or running when it moves.
    Idle(u8),
    Eat,
    Sleep,
    /// Head up, watching something.
    Alert,
    Attack,
    /// Just hurt.
    Hit,
    /// Dead (a carcass).
    Die,
    /// Leaping up a ledge at a run.
    Jump,
}

/// A creature as both sides know it.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Creature {
    pub id: CreatureId,
    /// Index into `kinds()`.
    pub kind: u16,
    /// Index into its kind's `models`.
    pub model: u8,
    /// Where it stands (its feet; a swimmer's middle).
    pub at: Vec3,
    pub yaw: f32,
    /// Metres per second over the ground.
    pub speed: f32,
    pub act: Act,
    /// The tick `act` started.
    pub since: u64,
    pub health: i32,
}

impl Creature {
    pub fn kind(&self) -> &'static Kind {
        &kinds()[self.kind as usize]
    }

    pub fn dead(&self) -> bool {
        self.act == Act::Die
    }

    /// The middle of its body (for reach and aim).
    pub fn body(&self) -> Vec3 {
        let k = self.kind();
        if k.swims() { self.at } else { self.at + Vec3::Y * k.height * 0.5 }
    }

    /// The way it faces, (x, z).
    pub fn forward(&self) -> Vec2 {
        Vec2::new(self.yaw.sin(), self.yaw.cos())
    }
}

/// Where a creature has got to, for the clients (sent for those that moved, a few times a second).
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Motion {
    pub id: CreatureId,
    pub at: Vec3,
    pub yaw: f32,
    pub speed: f32,
    pub tick: u64,
}

/// The yaw that turns a model (which faces +Z, as glTF's do) to face `dir` (x, z): models are
/// turned as parts are (see `asc_render::PartInstance::yaw`).
pub fn yaw_facing(dir: Vec2) -> f32 {
    dir.x.atan2(dir.y)
}

pub(crate) fn hash(seed: i32, a: i64, b: i64, salt: u64) -> u64 {
    let mut h = (seed as u32 as u64) ^ salt.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for v in [a, b] {
        h ^= v as u64;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(29) ^ 0x94D0_49BB_1331_11EB;
    }
    h ^ (h >> 31)
}

pub(crate) fn unit(h: u64) -> f32 {
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// The herds in one cell (deterministic): its land herd and its water one, either possibly
/// missing, each with its members.
pub fn herds_in(world: &World, seed: i32, cell: IVec2) -> Vec<(HerdId, u16, Vec<Creature>)> {
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
        let mut members = Vec::new();
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
            let leader = kind.has_leader && m == 0;
            members.push(Creature {
                id: CreatureId(hash(seed, cell.x as i64, cell.y as i64, 0x6964 ^ ((k as u64) << 8) ^ m as u64)),
                kind: k as u16,
                model,
                at,
                yaw: unit(mh.rotate_left(47)) * std::f32::consts::TAU,
                speed: 0.0,
                act: Act::Idle((unit(mh.rotate_left(53)) * kind.clips.len() as f32) as u8 % kind.clips.len() as u8),
                since: 0,
                health: if leader { kind.health * 13 / 10 } else { kind.health },
            });
        }
        if !members.is_empty() {
            let id = HerdId(hash(seed, cell.x as i64, cell.y as i64, 0x6865_7264 ^ habitat as u64));
            out.push((id, k as u16, members));
        }
    }
    out
}

/// A spot in or round `near` that suits `kind` (dry ground for land animals, water deep enough for
/// swimmers): `near` itself, or one of a few tries round it.
fn find_spot(world: &World, kind: &Kind, near: Vec2, h: u64) -> Option<Vec2> {
    (0..6)
        .map(|i| {
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

/// What the animals know of the player this tick.
#[derive(Clone, Copy, Debug)]
pub struct PlayerView {
    pub feet: Vec3,
    /// Metres per second.
    pub velocity: Vec3,
    /// 0..1 how much noise they're making (sprinting, digging, blasting).
    pub noise: f32,
    pub alive: bool,
    /// Swimming (fish and dolphins notice; land animals can't reach them).
    pub in_water: bool,
    /// 0..1 of full health.
    pub health: f32,
}

impl PlayerView {
    pub fn xz(&self) -> Vec2 {
        Vec2::new(self.feet.x, self.feet.z)
    }
}

/// Danger the animals run from: burning ground, pyroclastic flows and blasts coming, volcanoes
/// stirring.
#[derive(Clone, Debug, Default)]
pub struct Hazards {
    /// Middles of burning fire cells near the player.
    pub fires: Vec<Vec2>,
    pub flows: Vec<crate::volcanism::Flow>,
    pub blasts: Vec<crate::volcanism::Blast>,
    /// Volcanoes rumbling or erupting: summit and crater radius.
    pub volcanoes: Vec<(Vec3, f32)>,
}

/// A flow's cloud reaches this far either side of its way and is still deadly this far behind its
/// front (m); its way ahead is to be got out of.
const FLOW_SIDE: f32 = 40.0;
const FLOW_BEHIND: f32 = 350.0;

impl Hazards {
    /// How dangerous (x, z) is to stand on now or soon: 0 safe, more for worse.
    pub fn danger(&self, p: Vec2) -> f32 {
        let mut d = 0.0f32;
        for f in &self.fires {
            let r = f.distance(p);
            if r < crate::fire::CELL * 2.0 {
                d = d.max(6.0 * (1.0 - r / (crate::fire::CELL * 2.0)) + 1.0);
            }
        }
        for flow in &self.flows {
            let (along, side, half) = flow_frame(flow, p);
            if (0.0..=flow.length).contains(&along) && side < half + FLOW_SIDE && along > flow.front - FLOW_BEHIND {
                d = d.max(if along < flow.front { 20.0 } else { 8.0 });
            }
        }
        for b in &self.blasts {
            let scar = crate::volcanism::Scar::Blast { center: Vec2::new(b.center.x, b.center.z), dir: b.dir };
            if scar.contains(p) && p.distance(Vec2::new(b.center.x, b.center.z)) > b.radius - 100.0 {
                d = d.max(10.0);
            }
        }
        d
    }

    /// Whether (x, z) is inside something that kills (a flow's cloud, a blast's front).
    pub fn deadly(&self, p: Vec2) -> bool {
        self.flows.iter().any(|flow| {
            let (along, side, half) = flow_frame(flow, p);
            along >= 0.0 && along < flow.front && along > flow.front - FLOW_BEHIND && side < half
        }) || self.blasts.iter().any(|b| {
            let c = Vec2::new(b.center.x, b.center.z);
            let r = p.distance(c);
            crate::volcanism::Scar::Blast { center: c, dir: b.dir }.contains(p) && r < b.radius && r > b.radius - 150.0
        })
    }

    /// Whether (x, z) is on burning ground.
    pub fn burning(&self, p: Vec2) -> bool {
        let cell = crate::fire::cell_of(p.x, p.y);
        self.fires.iter().any(|f| crate::fire::cell_of(f.x, f.y) == cell)
    }
}

/// How far down a flow's way `p` is, how far off its middle line, and its half width there.
fn flow_frame(flow: &crate::volcanism::Flow, p: Vec2) -> (f32, f32, f32) {
    let rel = p - Vec2::new(flow.origin.x, flow.origin.z);
    let along = rel.dot(flow.dir);
    let half = flow.width * 0.5 * (0.25 + 0.75 * (along / flow.length.max(1.0)).clamp(0.0, 1.0));
    (along, rel.perp_dot(flow.dir).abs(), half)
}

/// What changed this tick, for the clients (and the player's health).
#[derive(Debug, Default)]
pub struct Changes {
    pub spawned: Vec<Creature>,
    pub gone: Vec<CreatureId>,
    pub moves: Vec<Motion>,
    pub acts: Vec<(CreatureId, Act, u64)>,
    /// Creatures hurt: their health now.
    pub hurt: Vec<(CreatureId, i32)>,
    /// Damage to the player.
    pub player: Vec<(i32, Cause)>,
}

/// Why a strike didn't land.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Miss {
    NoSuchCreature,
    OutOfReach,
    TooSoon,
    AlreadyDead,
}

/// The server's creatures: the herds of the cells near the player.
pub struct Creatures {
    seed: i32,
    /// By cell: its herds.
    cells: BTreeMap<[i32; 2], Vec<HerdId>>,
    herds: BTreeMap<HerdId, group::Herd>,
    agents: BTreeMap<CreatureId, brain::Agent>,
    pub nav: NavGrid,
    paths: PathFinder<CreatureId>,
    /// Animals killed from each herd: how many, and when the last was.
    losses: BTreeMap<HerdId, (u32, u64)>,
    /// Changes made between ticks (strikes), sent with the next tick's.
    pending: Changes,
    /// When the player last struck.
    last_strike: u64,
    /// Herds live where they live (off only in tests that place their own).
    pub(crate) natural: bool,
}

impl Creatures {
    pub fn new(seed: i32) -> Self {
        Self {
            seed,
            cells: BTreeMap::new(),
            herds: BTreeMap::new(),
            agents: BTreeMap::new(),
            nav: NavGrid::new(seed),
            paths: PathFinder::default(),
            losses: BTreeMap::new(),
            pending: Changes::default(),
            last_strike: 0,
            natural: true,
        }
    }

    pub fn all(&self) -> impl Iterator<Item = &Creature> {
        self.agents.values().map(|a| &a.c)
    }

    pub fn get(&self, id: CreatureId) -> Option<&Creature> {
        self.agents.get(&id).map(|a| &a.c)
    }

    /// What each is doing (for tests and logging).
    pub fn behaviour(&self, id: CreatureId) -> Option<Behaviour> {
        self.agents.get(&id).map(|a| a.behaviour)
    }

    /// Navigation and pathfinding since last asked: tiles built and kept, searches finished, cells
    /// they expanded, and searches under way.
    pub fn take_stats(&mut self) -> [usize; 5] {
        let (built, tiles) = self.nav.take_built();
        let (finished, expanded, searching) = self.paths.take_stats();
        [built, tiles, finished, expanded, searching]
    }

    /// Herds' losses, for saving: herd, how many, when the last was killed.
    pub fn losses(&self) -> Vec<(u64, u32, u64)> {
        self.losses.iter().map(|(h, (n, t))| (h.0, *n, *t)).collect()
    }

    pub fn restore_losses(&mut self, saved: &[(u64, u32, u64)]) {
        self.losses = saved.iter().map(|(h, n, t)| (HerdId(*h), (*n, *t))).collect();
    }

    /// How many a herd is short now: its losses, less one for every day since the last.
    fn short(&self, herd: HerdId, ticks: u64) -> u32 {
        self.losses.get(&herd).map_or(0, |(n, t)| n.saturating_sub((ticks.saturating_sub(*t) / DAY_TICKS) as u32))
    }

    /// The ground changed round `p` (dug, filled, built on, water or lava moved, a node mined).
    pub fn ground_changed(&mut self, p: Vec3, radius: f32) {
        self.nav.invalidate(Vec2::new(p.x, p.z), radius);
    }

    /// One tick: bring the herds near the player in and forget the far ones, build the ground
    /// they'll walk on, and let every animal think, find its way, move and fight.
    pub fn tick(
        &mut self,
        ground: &Ground,
        player: &PlayerView,
        hazards: &Hazards,
        ticks: u64,
        time_of_day: f32,
    ) -> Changes {
        let mut changes = std::mem::take(&mut self.pending);
        if self.natural && ticks.is_multiple_of(SPAWN_EVERY) {
            self.come_and_go(ground.world, player.xz(), ticks, &mut changes);
        }
        self.nav.build(ground, player.xz(), TILES_PER_TICK);
        let world = brain::World { player, hazards, ticks, time_of_day, seed: self.seed };
        let seen = brain::Seen::of(&self.agents);
        if ticks.is_multiple_of(group::EVERY) {
            for herd in self.herds.values_mut() {
                group::update(herd, &self.agents, &seen, &self.nav, &world);
            }
        }
        // Think: the near ones often, the far ones now and then.
        let mut events = Vec::new();
        for agent in self.agents.values_mut() {
            if agent.next_think <= ticks {
                let herd = self.herds.get(&agent.herd);
                brain::think(agent, herd, &seen, &mut self.nav, &world, &mut events, &mut changes);
            }
        }
        for event in events {
            match event {
                brain::Event::Path(ask) => {
                    let Some(a) = self.agents.get_mut(&ask.id) else { continue };
                    let profile = a.c.kind().profile();
                    match self.paths.request(&self.nav, ask.id, a.xz(), ask.to, profile, ask.prefs) {
                        Some(path) => a.follow(path),
                        None if !self.paths.searching(ask.id) => a.no_way(),
                        None => {}
                    }
                }
                brain::Event::Alarm(herd, from) => {
                    if let Some(h) = self.herds.get_mut(&herd)
                        && h.alarm.is_none_or(|(t, _)| ticks > t + 20)
                    {
                        h.alarm = Some((ticks, from));
                    }
                }
                brain::Event::Ate(carcass) => {
                    if let Some(c) = self.agents.get_mut(&carcass) {
                        c.eaten = true;
                    }
                }
            }
        }
        let danger = |c: IVec2| hazards.danger(crate::nav::centre(c));
        let (found, missing) = self.paths.step(&self.nav, &danger, PATH_BUDGET);
        for (id, path) in found {
            if let Some(a) = self.agents.get_mut(&id) {
                a.follow(path);
            }
        }
        for c in missing {
            self.nav.want(crate::nav::centre(c), 1.0);
        }
        // Move, then fight.
        for agent in self.agents.values_mut() {
            motion::step(agent, &self.nav, player, &seen, ticks, &mut changes);
        }
        let killed = combat::resolve(&mut self.agents, &world, &mut changes);
        for (herd, by_player) in killed {
            self.lost(herd, by_player, ticks);
        }
        self.hazards_hurt(hazards, ticks, &mut changes);
        // Carcasses rot away.
        let rotten: Vec<CreatureId> = self
            .agents
            .values()
            .filter(|a| a.c.dead() && ticks.saturating_sub(a.c.since) > CARCASS_TICKS)
            .map(|a| a.c.id)
            .collect();
        for id in rotten {
            self.remove(id, &mut changes);
        }
        if ticks.is_multiple_of(SEND_EVERY) {
            for a in self.agents.values_mut() {
                if let Some(m) = a.motion_to_send(ticks) {
                    changes.moves.push(m);
                }
            }
        }
        changes
    }

    /// The herds of cells near the player come; those of far cells go.
    fn come_and_go(&mut self, world: &World, here: Vec2, ticks: u64, changes: &mut Changes) {
        let reach = (SPAWN_REACH / CELL).ceil() as i32 + 1;
        let home = (here / CELL).floor().as_ivec2();
        for dz in -reach..=reach {
            for dx in -reach..=reach {
                let cell = home + IVec2::new(dx, dz);
                let centre = (cell.as_vec2() + 0.5) * CELL;
                if centre.distance(here) > SPAWN_REACH || self.cells.contains_key(&cell.to_array()) {
                    continue;
                }
                let mut ids = Vec::new();
                for (id, kind, mut members) in herds_in(world, self.seed, cell) {
                    // Those killed (and not yet grown back) stay missing: the last of the herd.
                    let short = self.short(id, ticks) as usize;
                    members.truncate(members.len().saturating_sub(short));
                    if members.is_empty() {
                        continue;
                    }
                    for c in &mut members {
                        c.since = ticks;
                    }
                    let herd = group::Herd::new(id, kind, &members);
                    self.nav.want(herd.home, herd.range + 16.0);
                    for (i, c) in members.iter().enumerate() {
                        let agent = brain::Agent::new(*c, id, i, self.seed, ticks);
                        changes.spawned.push(agent.c);
                        self.agents.insert(c.id, agent);
                    }
                    self.herds.insert(id, herd);
                    ids.push(id);
                }
                self.cells.insert(cell.to_array(), ids);
            }
        }
        let far: Vec<[i32; 2]> = self
            .cells
            .keys()
            .filter(|c| ((IVec2::from_array(**c).as_vec2() + 0.5) * CELL).distance(here) > KEEP_REACH)
            .copied()
            .collect();
        for cell in far {
            for herd in self.cells.remove(&cell).unwrap_or_default() {
                if let Some(h) = self.herds.remove(&herd) {
                    for id in h.members {
                        self.remove(id, changes);
                    }
                }
            }
        }
    }

    fn remove(&mut self, id: CreatureId, changes: &mut Changes) {
        if let Some(a) = self.agents.remove(&id) {
            self.paths.cancel(id);
            if let Some(h) = self.herds.get_mut(&a.herd) {
                h.members.retain(|m| *m != id);
            }
            changes.gone.push(id);
        }
    }

    /// A herd lost an animal: remember it (it comes back a day later), and the herd remembers who.
    fn lost(&mut self, herd: HerdId, by_player: bool, ticks: u64) {
        let short = self.short(herd, ticks);
        self.losses.insert(herd, (short + 1, ticks));
        if by_player && let Some(h) = self.herds.get_mut(&herd) {
            h.wary = 1.0;
        }
    }

    /// Lava, fire and the volcano's flows and blasts hurt and kill animals too.
    fn hazards_hurt(&mut self, hazards: &Hazards, ticks: u64, changes: &mut Changes) {
        let mut killed = Vec::new();
        for a in self.agents.values_mut() {
            if a.c.dead() {
                continue;
            }
            let p = Vec2::new(a.c.at.x, a.c.at.z);
            let lava = self.nav.cell(crate::nav::cell_at(p)).is_some_and(|c| c.has(crate::nav::LAVA));
            let damage = if hazards.deadly(p) {
                a.c.health
            } else if lava && ticks.is_multiple_of(5) {
                150
            } else if hazards.burning(p) && ticks.is_multiple_of(10) {
                40
            } else {
                0
            };
            if damage > 0 && combat::hurt(a, damage, None, ticks, changes) {
                killed.push(a.herd);
            }
        }
        for herd in killed {
            self.lost(herd, false, ticks);
        }
    }

    /// The player strikes creature `id` from `eye` with `damage`. Checked: it must be alive, within
    /// reach and the last strike long enough ago.
    pub fn strike(&mut self, id: CreatureId, eye: Vec3, damage: i32, ticks: u64) -> Result<bool, Miss> {
        let Some(a) = self.agents.get_mut(&id) else { return Err(Miss::NoSuchCreature) };
        if a.c.dead() {
            return Err(Miss::AlreadyDead);
        }
        let k = a.c.kind();
        if a.c.body().distance(eye) > combat::PLAYER_REACH + k.radius + k.height * 0.5 {
            return Err(Miss::OutOfReach);
        }
        if ticks < self.last_strike + combat::STRIKE_TICKS {
            return Err(Miss::TooSoon);
        }
        self.last_strike = ticks;
        let died = combat::hurt(a, damage, Some(brain::Who::Player), ticks, &mut self.pending);
        let herd = a.herd;
        if let Some(h) = self.herds.get_mut(&herd) {
            h.wary = h.wary.max(0.7);
            h.alarm = Some((ticks, Vec2::new(eye.x, eye.z)));
        }
        if died {
            self.lost(herd, true, ticks);
        }
        Ok(died)
    }

    /// Butcher the carcass `id` standing at `feet`: what it gives (item id, count), and it's gone.
    /// `None` if there's no carcass there.
    pub fn butcher(&mut self, id: CreatureId, feet: Vec3, ticks: u64) -> Option<Vec<(&'static str, u32)>> {
        let a = self.agents.get(&id)?;
        if !a.c.dead() || a.c.at.distance(feet) > combat::BUTCHER_REACH + a.c.kind().radius {
            return None;
        }
        let k = a.c.kind();
        // Something has fed on it: there's less left.
        let eaten = a.eaten;
        let out = k
            .drops
            .iter()
            .enumerate()
            .filter_map(|(i, d)| {
                let u = unit(hash(self.seed, id.0 as i64, ticks as i64, 0x6472 ^ i as u64));
                let n = d.count[0] + (u * (d.count[1] - d.count[0] + 1) as f32) as u32;
                let n = n.min(d.count[1]);
                let n = if eaten { n / 2 } else { n };
                (n > 0).then_some((d.item.as_str(), n))
            })
            .collect();
        let mut changes = std::mem::take(&mut self.pending);
        self.remove(id, &mut changes);
        self.pending = changes;
        Some(out)
    }
}
