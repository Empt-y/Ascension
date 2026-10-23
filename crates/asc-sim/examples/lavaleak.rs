//! Finds where a volcano's own lava (its sources) leaks into open space that isn't marked as
//! lava: the lake's source cells with an open, non-source, non-lava neighbour, after a short forced
//! eruption. Stops after `TICKS` (default 300). `cargo run --release -p asc-sim --example lavaleak`.
use asc_sim::fluids::FluidSim;
use asc_sim::volcanism::{RUMBLE_TICKS, Volcanism};
use asc_world::fluid::{CELLS, FluidKind, cell_index};
use asc_world::volcano::VolcanoKind;
use asc_world::World;
use glam::{IVec3, Vec2, Vec3};

fn main() {
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1256132528);
    let ticks: u64 = std::env::var("TICKS").ok().and_then(|s| s.parse().ok()).unwrap_or(300);
    let world = World::new(seed);
    let wg = &world.worldgen;
    let v = wg.volcanoes_near(Vec2::ZERO, 16_000.0).into_iter().find(|v| v.kind == VolcanoKind::Active).expect("an active cone");
    let player = Vec3::new(v.center.x + v.radius * 0.4, 200.0, v.center.y);
    let mut volcanism = Volcanism::new(seed);
    let mut fluids = FluidSim::new();
    volcanism.force(&world, player, 0, asc_sim::volcanism::Kind::Natural);
    for t in RUMBLE_TICKS..RUMBLE_TICKS + ticks {
        volcanism.tick(&world, &mut fluids, player, t);
        fluids.tick(&world);
    }
    // Source lava next to open cells that hold no lava and aren't sources.
    let mut leaks = Vec::new();
    for (coord, ch) in &fluids.fluids.chunks {
        for i in 0..CELLS {
            if ch.kind[i] != FluidKind::Lava as u8 || ch.source[i] == 0 {
                continue;
            }
            let p = IVec3::new((i % 32) as i32, ((i / 32) % 32) as i32, (i / 1024) as i32);
            for d in [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z, IVec3::NEG_Y] {
                let q = p + d;
                if q.cmplt(IVec3::ZERO).any() || q.cmpge(IVec3::splat(32)).any() {
                    continue;
                }
                let j = cell_index(q.x as usize, q.y as usize, q.z as usize);
                if !ch.solid[j] && ch.source[j] == 0 {
                    leaks.push((*coord * 32 + q, d, ch.amount[j]));
                }
            }
        }
    }
    let spilled: u64 = fluids.fluids.chunks.values().flat_map(|ch| (0..CELLS).filter(move |&i| ch.source[i] == 0).map(move |i| ch.amount[i] as u64)).sum();
    println!("after {ticks} ticks: {} chunks, {} cells spilled; {} open non-source cells beside the lake's sources", fluids.fluids.chunks.len(), spilled / 1000, leaks.len());
    let lava = v.lava.unwrap_or(0.0);
    for (at, d, amount) in leaks.iter().take(15) {
        let r = Vec2::new(at.x as f32 - v.center.x, at.z as f32 - v.center.y).length();
        println!("  {at} ({d} of a source): {r:.0} m from the middle, {:.1} m vs the lake's level, holding {amount}; ground here {:.1}", at.y as f32 - lava, wg.height(at.x as f32, at.z as f32));
    }
}
