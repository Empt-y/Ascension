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
    volcanism.force(&world, player, 0, asc_sim::volcanism::Kind::Natural);
    let (mut v_ms, mut f_ms, mut worst) = (0.0f64, 0.0f64, 0.0f64);
    for ticks in 0..(RUMBLE_TICKS + asc_sim::volcanism::ERUPT_TICKS + 600) {
        let t0 = Instant::now();
        volcanism.tick(&world, &mut fluids, player, ticks);
        let t1 = Instant::now();
        fluids.tick(&world);
        let t2 = Instant::now();
        v_ms += (t1 - t0).as_secs_f64() * 1e3;
        f_ms += (t2 - t1).as_secs_f64() * 1e3;
        worst = worst.max((t2 - t0).as_secs_f64() * 1e3);
        // A runaway (lava flooding out without end) would take hours: stop and say so.
        if (t2 - t0).as_secs_f64() > 0.25 {
            println!("tick {ticks} took {:.0} ms with {} chunks: stopping (runaway)", (t2 - t0).as_secs_f64() * 1e3, fluids.fluids.chunks.len());
            return;
        }
        if (ticks + 1) % 300 == 0 {
            println!(
                "tick {}: volcano {:.2} ms, fluids {:.2} ms per tick (worst {worst:.1}); {} chunks, {} awake",
                ticks + 1,
                v_ms / 300.0,
                f_ms / 300.0,
                fluids.fluids.chunks.len(),
                fluids.awake_count()
            );
            // How far the spilled lava has got from the crater's middle, and how much there is.
            let (mut far, mut cells) = (0.0f32, 0u64);
            for (coord, ch) in &fluids.fluids.chunks {
                for i in 0..asc_world::fluid::CELLS {
                    if ch.kind[i] == 2 && ch.source[i] == 0 && ch.amount[i] > 100 {
                        let p = *coord * 32 + glam::IVec3::new((i % 32) as i32, ((i / 32) % 32) as i32, (i / 1024) as i32);
                        far = far.max(Vec2::new(p.x as f32 - v.center.x, p.z as f32 - v.center.y).length());
                        cells += 1;
                    }
                }
            }
            println!("    spilled lava: {cells} cells, reaching {far:.0} m from the middle (crater radius {:.0})", v.crater_radius);
            (v_ms, f_ms, worst) = (0.0, 0.0, 0.0);
        }
    }
}
