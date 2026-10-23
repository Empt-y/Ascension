//! Compares the river flow direction with the way the channel actually runs at river points near
//! the origin. `cargo run --release -p asc-world --example flowcheck`.
use asc_world::WorldGen;

fn main() {
    let wg = WorldGen::new(1337);
    let mut shown = 0;
    for i in 0..20000 {
        let (x, z) = ((i % 100) as f32 * 6.0 - 300.0, (i / 100) as f32 * 6.0 - 300.0);
        let Some(l) = wg.water_level(x, z) else { continue };
        if l <= asc_world::SEA_LEVEL {
            continue;
        }
        // The channel's direction: of 36 directions, the one with the most water 3..12 m out.
        let mut best = (0, 0.0f32);
        for k in 0..36 {
            let a = k as f32 / 36.0 * std::f32::consts::TAU;
            let wet = (1..=4).filter(|r| wg.water_level(x + a.cos() * *r as f32 * 3.0, z + a.sin() * *r as f32 * 3.0).is_some()).count();
            if wet > best.0 {
                best = (wet, a);
            }
        }
        let f = wg.river_flow(x, z);
        let fa = f.y.atan2(f.x);
        let diff = ((fa - best.1).rem_euclid(std::f32::consts::PI)).min(std::f32::consts::PI - (fa - best.1).rem_euclid(std::f32::consts::PI));
        println!("({x:.0},{z:.0}) channel {:.0} deg, flow {:.0} deg (|{:.2}|), off by {:.0} deg", best.1.to_degrees(), fa.to_degrees(), f.length(), diff.to_degrees());
        shown += 1;
        if shown >= 12 {
            break;
        }
    }
}
