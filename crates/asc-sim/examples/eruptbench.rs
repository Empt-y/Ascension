//! Times the server-side work of an eruption: a forced eruption near an active cone, ticking the
//! volcano and the fluids. `SEED=.. cargo run --release -p asc-sim --example eruptbench`.
use std::time::Instant;

use asc_sim::fluids::FluidSim;
use asc_sim::volcanism::{RUMBLE_TICKS, Volcanism};
use asc_world::volcano::VolcanoKind;
use asc_world::World;
use glam::{Vec2, Vec3};

fn main() {
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1256132528);
    let world = World::new(seed);
    let wg = &world.worldgen;
    let v = wg.volcanoes_near(Vec2::ZERO, 16_000.0).into_iter().find(|v| v.kind == VolcanoKind::Active).expect("an active cone");
    let (x, z) = (v.center.x + v.radius * 0.4, v.center.y);
    let player = Vec3::new(x, wg.height(x, z) + 1.0, z);
    let mut volcanism = Volcanism::new(seed);
    let mut fluids = FluidSim::new();
    volcanism.force(&world, player, 0);
    let (mut v_ms, mut f_ms, mut worst) = (0.0f64, 0.0f64, 0.0f64);
    for ticks in 0..(RUMBLE_TICKS + 1200) {
        let t0 = Instant::now();
        volcanism.tick(&world, &mut fluids, player, ticks);
        let t1 = Instant::now();
        fluids.tick(&world);
        let t2 = Instant::now();
        v_ms += (t1 - t0).as_secs_f64() * 1e3;
        f_ms += (t2 - t1).as_secs_f64() * 1e3;
        worst = worst.max((t2 - t0).as_secs_f64() * 1e3);
        if (ticks + 1) % 300 == 0 {
            println!(
                "tick {}: volcano {:.2} ms, fluids {:.2} ms per tick (worst {worst:.1}); {} chunks, {} awake",
                ticks + 1,
                v_ms / 300.0,
                f_ms / 300.0,
                fluids.fluids.chunks.len(),
                fluids.awake_count()
            );
            (v_ms, f_ms, worst) = (0.0, 0.0, 0.0);
        }
    }
}
