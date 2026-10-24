use std::collections::BTreeSet;

use asc_build::{Lattice, Structures};
use asc_world::fluid::Fluids;

use super::brain::Behaviour;
use super::*;

#[test]
fn the_data_file_parses_and_names_real_biomes() {
    let k = kinds();
    assert!(k.len() >= 8);
    let cattle = k.iter().find(|k| k.id == "cattle").unwrap();
    assert!(cattle.has_leader && cattle.models[0] == "bull");
    assert!(k.iter().filter(|k| k.habitat == Habitat::Land).all(|k| k.chance.iter().any(|c| *c > 0.0)));
    let wolf = &k[kind_index("wolf").unwrap() as usize];
    assert!(wolf.predator() && wolf.prey.contains(&kind_index("stag").unwrap()));
    assert_eq!(wolf.killed, "was killed by a wolf");
    assert!(parse("[[creature]]\nid='x'\nname='X'\nmodels=['x']\nhabitat='land'\nherd=[1,1]\nspread=1.0\nidle=['Idle']\nbiomes={moon=0.1}\nhealth=1\nradius=1.0\nheight=1.0\nwalk=1.0\nrun=2.0\nstride=[1.0,1.0]").is_err());
    // Every drop is a real item.
    let content = crate::content::Content::load().unwrap();
    for kind in k {
        for d in &kind.drops {
            assert!(content.item_id(&d.item).is_some(), "{} drops {}, which isn't an item", kind.id, d.item);
        }
    }
}

#[test]
fn herds_are_the_same_every_time_and_only_where_they_live() {
    let world = World::new(1337);
    let (mut land, mut water) = (0, 0);
    for z in -12..12 {
        for x in -12..12 {
            let cell = IVec2::new(x, z);
            let herds = herds_in(&world, 1337, cell);
            assert_eq!(herds.iter().map(|h| h.2.clone()).collect::<Vec<_>>(), herds_in(&world, 1337, cell).into_iter().map(|h| h.2).collect::<Vec<_>>(), "not deterministic");
            for (_, _, herd) in &herds {
                for c in herd {
                    let kind = c.kind();
                    let p = Vec2::new(c.at.x, c.at.z);
                    let g = world.worldgen.ground(p.x, p.y);
                    let level = asc_world::fluid::water_level(&world.worldgen, p.x, p.y);
                    match kind.habitat {
                        Habitat::Land => {
                            land += 1;
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
    }
    assert!(land > 20, "only {land} land animals in 576 cells");
    assert!(water > 0, "no swimmers in 576 cells");
}

/// A world to try the animals in.
struct Field {
    world: World,
    structures: Structures,
    fluids: Fluids,
    mined: BTreeSet<[i32; 3]>,
    creatures: Creatures,
    player: PlayerView,
    hazards: Hazards,
    ticks: u64,
    time_of_day: f32,
    /// Damage the player has taken.
    hurt: i32,
}

impl Field {
    fn new() -> Self {
        let spec = asc_build::BuildingSpec::parse(include_str!("../../../../assets/data/building.toml")).unwrap();
        let mut creatures = Creatures::new(1337);
        creatures.natural = false;
        Self {
            world: World::new(1337),
            structures: Structures::new(Lattice::new(&spec)),
            fluids: Fluids::default(),
            mined: BTreeSet::new(),
            creatures,
            player: PlayerView {
                feet: Vec3::new(1e5, 0.0, 1e5),
                velocity: Vec3::ZERO,
                noise: 0.0,
                alive: true,
                in_water: false,
                health: 1.0,
                torch: false,
            },
            hazards: Hazards::default(),
            ticks: 100,
            time_of_day: 0.45,
            hurt: 0,
        }
    }

    /// Dry, gentle ground (where players start).
    fn open_ground(&self) -> Vec2 {
        let p = crate::rules::spawn_point(&self.world);
        Vec2::new(p.x, p.z)
    }

    /// A herd of `n` of `kind` round `at`.
    fn add(&mut self, kind: &str, at: Vec2, n: usize) -> Vec<CreatureId> {
        let k = kind_index(kind).unwrap();
        let kd = &kinds()[k as usize];
        let herd = HerdId(hash(7, at.x as i64, at.y as i64, k as u64));
        let members: Vec<Creature> = (0..n)
            .map(|i| {
                let p = at + Vec2::from_angle(i as f32 * 2.4) * (1.5 + i as f32 * 1.2);
                Creature {
                    id: CreatureId(hash(9, herd.0 as i64, i as i64, 1)),
                    kind: k,
                    model: 0,
                    at: crate::rules::find_ground(&self.world, p.x, p.y),
                    yaw: 0.0,
                    speed: 0.0,
                    act: Act::Idle(0),
                    since: self.ticks,
                    health: kd.health,
                }
            })
            .collect();
        let h = group::Herd::new(herd, k, &members);
        // The ground is only kept round the player: have them about, out of sight and earshot.
        if self.player.xz().distance(h.home) > 250.0 {
            self.player.feet = Vec3::new(h.home.x + 200.0, 0.0, h.home.y);
        }
        self.creatures.nav.want(h.home, h.range + 40.0);
        let ground = Ground { world: &self.world, structures: &self.structures, fluids: &self.fluids, mined: &self.mined };
        self.creatures.nav.build(&ground, h.home, usize::MAX);
        self.creatures.herds.insert(herd, h);
        for (i, c) in members.iter().enumerate() {
            self.creatures.agents.insert(c.id, brain::Agent::new(*c, herd, i, 1337, self.ticks));
        }
        members.iter().map(|c| c.id).collect()
    }

    fn agent(&mut self, id: CreatureId) -> &mut brain::Agent {
        self.creatures.agents.get_mut(&id).unwrap()
    }

    fn at(&self, id: CreatureId) -> Vec2 {
        let c = self.creatures.get(id).unwrap();
        Vec2::new(c.at.x, c.at.z)
    }

    /// Run `seconds`, the player moving at their velocity.
    fn run(&mut self, seconds: f32) {
        for _ in 0..(seconds * TICKS_PER_SECOND) as u32 {
            self.ticks += 1;
            self.player.feet += self.player.velocity / TICKS_PER_SECOND;
            let ground = Ground { world: &self.world, structures: &self.structures, fluids: &self.fluids, mined: &self.mined };
            let ch = self.creatures.tick(&ground, &self.player, &self.hazards, self.ticks, self.time_of_day);
            self.hurt += ch.player.iter().map(|(d, _)| d).sum::<i32>();
        }
    }

    fn behaviour(&self, id: CreatureId) -> Behaviour {
        self.creatures.behaviour(id).unwrap()
    }
}

#[test]
fn a_herd_runs_from_the_player_together() {
    let mut f = Field::new();
    let spot = f.open_ground();
    let herd = f.add("horse", spot, 5);
    let start: Vec<Vec2> = herd.iter().map(|id| f.at(*id)).collect();
    // Settle, then walk up to them from 70 m off at a jog.
    f.player.feet = Vec3::new(spot.x - 70.0, 0.0, spot.y);
    f.run(3.0);
    f.player.velocity = Vec3::new(5.0, 0.0, 0.0);
    for _ in 0..9 {
        f.run(1.0);
        if std::env::var("ASC_TRACE").is_ok() {
            for id in &herd {
                let a = &f.creatures.agents[id];
                println!("  {:?} at {} fear {:.2} speed {:.1} goal {:?} path {} asked {:?} stuck {} act {:?}", a.behaviour, a.xz(), a.fear, a.c.speed, a.goal.map(|g| g.to), a.path.len(), a.asked.map(|x| x.0), a.stuck, a.c.act);
            }
        }
    }
    let fled = herd.iter().filter(|id| matches!(f.behaviour(**id), Behaviour::Flee)).count();
    assert!(fled >= 4, "{fled} of 5 fled");
    f.player.velocity = Vec3::ZERO;
    f.run(6.0);
    let away = herd.iter().zip(&start).filter(|(id, s)| f.at(**id).x > s.x + 15.0).count();
    assert!(away >= 4, "{away} of 5 got away east: {:?}", herd.iter().map(|id| f.at(*id)).collect::<Vec<_>>());
    // And they remember.
    assert!(herd.iter().all(|id| f.creatures.agents[id].wary > 0.05), "nobody grew wary");
}

#[test]
fn a_herd_sleeps_at_night() {
    let mut f = Field::new();
    let spot = f.open_ground();
    let herd = f.add("horse", spot, 3);
    f.time_of_day = 0.02;
    f.run(20.0);
    for id in &herd {
        assert_eq!(f.behaviour(*id), Behaviour::Sleep);
    }
    assert!(herd.iter().any(|id| f.creatures.get(*id).unwrap().act == Act::Sleep));
}

#[test]
fn a_hungry_herd_grazes_and_gets_fuller() {
    let mut f = Field::new();
    let spot = f.open_ground();
    let herd = f.add("horse", spot, 2);
    for id in &herd {
        f.agent(*id).hunger = 0.9;
        f.agent(*id).thirst = 0.0;
    }
    f.run(40.0);
    for id in &herd {
        assert!(f.creatures.agents[id].hunger < 0.85, "hunger still {}", f.creatures.agents[id].hunger);
    }
}

/// Dry ground with water a short walk off: (dry spot, water).
fn near_water(f: &Field) -> (Vec2, Vec2) {
    let wg = &f.world.worldgen;
    for r in 1..400 {
        for a in 0..24 {
            let p = Vec2::from_angle(a as f32 / 24.0 * std::f32::consts::TAU) * r as f32 * 12.0;
            let g = wg.ground(p.x, p.y);
            let Some(l) = wg.water_level(p.x, p.y) else { continue };
            if l - g.height < 0.3 || l - g.height > 2.0 {
                continue;
            }
            // Gentle dry land within 25 m.
            for b in 0..12 {
                let q = p + Vec2::from_angle(b as f32 * 0.52) * 25.0;
                let gq = wg.ground(q.x, q.y);
                if wg.water_level(q.x, q.y).is_none() && gq.slope < 0.3 && (gq.height - g.height).abs() < 4.0 {
                    return (q, p);
                }
            }
        }
    }
    panic!("no water near dry land");
}

#[test]
fn a_thirsty_stag_walks_to_water_and_drinks() {
    let mut f = Field::new();
    let (dry, water) = near_water(&f);
    let id = f.add("stag", dry, 1)[0];
    f.agent(id).thirst = 1.0;
    f.agent(id).hunger = 0.0;
    f.run(60.0);
    let a = &f.creatures.agents[&id];
    assert!(a.thirst < 0.9, "still thirsty ({}) at {} (water {water})", a.thirst, f.at(id));
}

#[test]
fn a_wolf_pack_hunts_down_a_stag_and_feeds() {
    let mut f = Field::new();
    let spot = f.open_ground();
    let stag = f.add("stag", spot, 1)[0];
    // The pack 40 m off, somewhere it can get to the stag from (not over water or cliffs).
    let wolf = kinds()[kind_index("wolf").unwrap() as usize].profile();
    let from = f.at(stag);
    let den = f
        .creatures
        .nav
        .reachable(from, &wolf, 45.0, 20_000)
        .into_iter()
        .map(crate::nav::centre)
        .find(|p| p.distance(from) > 38.0)
        .expect("ground the pack can come over");
    let pack = f.add("wolf", den, 4);
    f.time_of_day = 0.9;
    for w in &pack {
        f.agent(*w).hunger = 0.95;
    }
    f.agent(stag).hunger = 0.0;
    f.agent(stag).thirst = 0.0;
    let mut dead = false;
    for _ in 0..120 {
        f.run(1.0);
        if std::env::var("ASC_TRACE").is_ok() {
            let s = &f.creatures.agents[&stag];
            println!("stag {:?} at {} hp {} st {:.2} fear {:.2}", s.behaviour, s.xz(), s.c.health, s.stamina, s.fear);
            for w in &pack {
                let a = &f.creatures.agents[w];
                println!("  wolf {:?} at {} d {:.1} speed {:.1} st {:.2} path {} att {:?} act {:?} target {:?}", a.behaviour, a.xz(), a.xz().distance(s.xz()), a.c.speed, a.stamina, a.path.len(), a.attack, a.c.act, f.creatures.herds[&a.herd].target.map(|t| t.since));
            }
        }
        if f.creatures.get(stag).unwrap().dead() {
            dead = true;
            break;
        }
    }
    if !dead && std::env::var("ASC_TRACE").is_ok() {
        let c0 = crate::nav::cell_at(f.at(stag));
        for dz in -5..=5 {
            let row: Vec<String> = (-7..=7)
                .map(|dx| {
                    let c = c0 + IVec2::new(dx, dz);
                    f.creatures.nav.cell(c).map_or("  ----  ".into(), |x| format!("{:5.1}{}{}{}", x.height, if wolf.fits(x) { 'o' } else { 'x' }, x.water.min(9), x.clear.min(9)))
                })
                .collect();
            println!("{}", row.join(" "));
        }
    }
    assert!(dead, "the stag got away (health {})", f.creatures.get(stag).unwrap().health);
    f.run(8.0);
    let feeding = pack.iter().filter(|w| f.behaviour(**w) == Behaviour::Feed).count();
    assert!(feeding >= 2, "{feeding} wolves feeding");
    f.run(20.0);
    assert!(f.creatures.agents[&stag].eaten);
    let fed = pack.iter().filter(|w| f.creatures.agents[*w].hunger < 0.5).count();
    assert!(fed >= 2, "{fed} wolves fed");
}

#[test]
fn hitting_a_cow_brings_the_bull() {
    let mut f = Field::new();
    let spot = f.open_ground();
    let herd = f.add("cattle", spot, 4);
    let (bull, cow) = (herd[0], herd[2]);
    f.run(2.0);
    let c = f.creatures.get(cow).unwrap().body();
    f.player.feet = c - Vec3::new(1.5, 0.9, 0.0);
    let eye = f.player.feet + Vec3::Y * 1.6;
    assert_eq!(f.creatures.strike(cow, eye + Vec3::X * 50.0, 60, f.ticks), Err(Miss::OutOfReach));
    assert_eq!(f.creatures.strike(cow, eye, 60, f.ticks), Ok(false));
    assert_eq!(f.creatures.strike(cow, eye, 60, f.ticks + 1), Err(Miss::TooSoon));
    f.run(10.0);
    assert_eq!(f.behaviour(bull), Behaviour::Fight, "the bull is {:?}", f.behaviour(bull));
    assert!(f.hurt > 0, "the bull never landed a blow");
}

#[test]
fn a_kill_can_be_butchered_once_and_the_herd_comes_back_short() {
    let mut f = Field::new();
    let spot = f.open_ground();
    let id = f.add("donkey", spot, 1)[0];
    let feet = f.creatures.get(id).unwrap().at;
    let eye = feet + Vec3::Y * 1.0;
    let mut died = false;
    for i in 0..20 {
        if f.creatures.strike(id, eye, 60, f.ticks + i * 10) == Ok(true) {
            died = true;
            break;
        }
    }
    assert!(died);
    let herd = f.creatures.agents[&id].herd;
    assert_eq!(f.creatures.short(herd, f.ticks), 1);
    let drops = f.creatures.butcher(id, feet, f.ticks).expect("a carcass to butcher");
    assert!(drops.iter().any(|(item, n)| *item == "raw_meat" && *n >= 4));
    assert!(f.creatures.butcher(id, feet, f.ticks).is_none(), "butchered twice");
    // A day later it has grown back.
    assert_eq!(f.creatures.short(herd, f.ticks + DAY_TICKS + 400), 0);
}

#[test]
fn animals_run_from_fire() {
    let mut f = Field::new();
    let spot = f.open_ground();
    let id = f.add("stag", spot, 1)[0];
    f.run(1.0);
    let start = f.at(id);
    f.hazards.fires = (-2..=2).flat_map(|z| (-2..=2).map(move |x| start + Vec2::new(x as f32, z as f32) * crate::fire::CELL)).collect();
    f.run(1.0);
    assert_eq!(f.behaviour(id), Behaviour::Escape);
    f.run(6.0);
    assert!(f.at(id).distance(start) > 12.0, "only {} m from the fire", f.at(id).distance(start));
}

#[test]
fn wariness_fades_over_minutes() {
    let mut f = Field::new();
    let spot = f.open_ground();
    let id = f.add("stag", spot, 1)[0];
    f.agent(id).wary = 0.8;
    f.run(2.0);
    assert!(f.creatures.agents[&id].wary > 0.6);
    f.run(600.0);
    assert!(f.creatures.agents[&id].wary < 0.25, "still {}", f.creatures.agents[&id].wary);
}

#[test]
fn the_same_seed_gives_the_same_chase() {
    let run = || {
        let mut f = Field::new();
        let spot = f.open_ground();
        let herd = f.add("horse", spot, 4);
        f.player.feet = Vec3::new(spot.x - 40.0, 0.0, spot.y);
        f.player.velocity = Vec3::new(6.0, 0.0, 0.0);
        f.run(8.0);
        herd.iter().map(|id| f.creatures.get(*id).unwrap().at).collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}

#[test]
fn herds_come_near_and_go_far() {
    let mut f = Field::new();
    f.creatures.natural = true;
    let mut seen = std::collections::BTreeSet::new();
    f.player.feet = Vec3::ZERO;
    for t in 0..20 * 60 {
        // Walk east at 10 m/s.
        f.player.feet.x = t as f32 * 0.5;
        f.ticks += 1;
        let ground = Ground { world: &f.world, structures: &f.structures, fluids: &f.fluids, mined: &f.mined };
        let ch = f.creatures.tick(&ground, &f.player, &f.hazards, f.ticks, f.time_of_day);
        for c in &ch.spawned {
            assert!(seen.insert(c.id), "spawned twice");
        }
        for id in &ch.gone {
            assert!(seen.contains(id));
        }
    }
    assert!(!seen.is_empty());
    for c in f.creatures.all() {
        assert!(Vec2::new(c.at.x - f.player.feet.x, c.at.z).length() < KEEP_REACH + CELL * 2.0);
    }
}

#[test]
fn a_hundred_animals_think_quickly() {
    let mut f = Field::new();
    let spot = f.open_ground();
    for i in 0..10 {
        let at = spot + Vec2::from_angle(i as f32 * 0.63) * 60.0;
        f.add(if i % 3 == 0 { "wolf" } else { "horse" }, at, 10);
    }
    f.player.feet = Vec3::new(spot.x, 0.0, spot.y);
    f.run(5.0);
    let start = std::time::Instant::now();
    f.run(10.0);
    let per_tick = start.elapsed().as_secs_f32() * 1000.0 / 200.0;
    println!("{} animals: {per_tick:.3} ms a tick", f.creatures.agents.len());
    if !cfg!(debug_assertions) {
        assert!(per_tick < 2.5, "{per_tick:.2} ms a tick");
    }
}
