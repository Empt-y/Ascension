//! Ten minutes of wildlife without the game: the herds round a spot live their lives (the player
//! standing off a little way, then walking through them), and what they got up to is counted.
//! `SEED=.. MINUTES=.. cargo run --release -p asc-sim --example creatures`.
use std::collections::BTreeMap;
use std::time::Instant;

use asc_build::{BuildingSpec, Lattice, Structures};
use asc_sim::creatures::{Act, Creatures, Hazards, PlayerView, kinds};
use asc_sim::nav::Ground;
use asc_world::World;
use asc_world::fluid::Fluids;
use glam::Vec3;

fn main() {
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1337);
    let minutes: f32 = std::env::var("MINUTES").ok().and_then(|s| s.parse().ok()).unwrap_or(10.0);
    let world = World::new(seed);
    let structures = Structures::new(Lattice::new(&BuildingSpec::parse(include_str!("../../../assets/data/building.toml")).unwrap()));
    let fluids = Fluids::default();
    let mined = Default::default();
    let ground = Ground { world: &world, structures: &structures, fluids: &fluids, mined: &mined };
    let start = asc_sim::rules::spawn_point(&world);
    let mut creatures = Creatures::new(seed);
    let hazards = Hazards::default();
    let mut player = PlayerView { feet: start, velocity: Vec3::ZERO, noise: 0.0, alive: true, in_water: false, health: 1.0 };
    let ticks_total = (minutes * 60.0 * 20.0) as u64;
    let mut time_of_day = 0.3f32;
    let mut counts: BTreeMap<String, BTreeMap<String, u64>> = BTreeMap::new();
    let (mut worst, mut total) = (0.0f64, 0.0f64);
    let (mut hurt, mut moves, mut acts) = (0, 0, 0);
    let mut stats = [0usize; 5];
    for t in 1..=ticks_total {
        // Standing off for the first third; then walking through at a jog, east and back.
        let phase = t as f32 / ticks_total as f32;
        player.velocity = if (0.33..0.66).contains(&phase) {
            Vec3::new(if ((t / 600) % 2) == 0 { 4.0 } else { -4.0 }, 0.0, 0.0)
        } else {
            Vec3::ZERO
        };
        player.feet += player.velocity / 20.0;
        player.feet.y = world.worldgen.height(player.feet.x, player.feet.z);
        time_of_day = (time_of_day + 1.0 / 24_000.0).rem_euclid(1.0);
        let t0 = Instant::now();
        let ch = creatures.tick(&ground, &player, &hazards, t, time_of_day);
        let ms = t0.elapsed().as_secs_f64() * 1e3;
        worst = worst.max(ms);
        total += ms;
        hurt += ch.player.iter().map(|(d, _)| *d).sum::<i32>();
        moves += ch.moves.len();
        acts += ch.acts.len();
        let s = creatures.take_stats();
        for i in [0, 2, 3] {
            stats[i] += s[i];
        }
        stats[1] = s[1];
        stats[4] = stats[4].max(s[4]);
        if t % 100 == 0 {
            let ids: Vec<_> = creatures.all().map(|c| (c.id, c.kind, c.act)).collect();
            for (id, kind, act) in ids {
                let b = if act == Act::Die { "Dead".to_string() } else { format!("{:?}", creatures.behaviour(id).unwrap()) };
                *counts.entry(kinds()[kind as usize].id.clone()).or_default().entry(b).or_default() += 1;
            }
        }
    }
    println!("{minutes} minutes round {start} (seed {seed}): {} creatures at the end", creatures.all().count());
    println!("tick: {:.3} ms on average, {:.2} ms at worst", total / ticks_total as f64, worst);
    println!("sent: {moves} moves, {acts} acts; the player took {hurt} damage");
    println!(
        "nav: {} tiles built ({} kept); paths: {} searches, {} cells, at most {} under way at once",
        stats[0], stats[1], stats[2], stats[3], stats[4]
    );
    println!("what they were doing (samples every 5 s):");
    for (kind, b) in counts {
        let all: u64 = b.values().sum();
        let list: Vec<String> = b.iter().map(|(k, n)| format!("{k} {:.0}%", *n as f32 / all as f32 * 100.0)).collect();
        println!("  {kind:>8}: {}", list.join(", "));
    }
}
