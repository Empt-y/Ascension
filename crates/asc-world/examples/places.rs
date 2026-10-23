//! Finds places worth a look near the origin: a river, a lake, a waterfall and a cave above the lava
//! sea. `cargo run --release -p asc-world --example places`.

use asc_world::{SEA_LEVEL, WorldGen};
use glam::Vec3;

fn main() {
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1337);
    let wg = WorldGen::new(seed);
    let mut river = None;
    let mut lake = None;
    let mut fall = None;
    for r in (40..3000).step_by(16) {
        for i in 0..64 {
            let a = i as f32 / 64.0 * std::f32::consts::TAU;
            let (x, z) = (a.cos() * r as f32, a.sin() * r as f32);
            let Some(l) = wg.water_level(x, z) else { continue };
            if l == SEA_LEVEL {
                continue;
            }
            // A waterfall: water here, water 3 m lower within 6 m.
            if fall.is_none() {
                for d in [(6.0, 0.0), (-6.0, 0.0), (0.0, 6.0), (0.0, -6.0)] {
                    if let Some(l2) = wg.water_level(x + d.0, z + d.1)
                        && l - l2 > 2.5
                        && l2 != SEA_LEVEL
                    {
                        fall = Some((x, z, l));
                    }
                }
            }
            // A lake: water over a wide area at one level.
            let wide = [(30.0, 0.0), (-30.0, 0.0), (0.0, 30.0), (0.0, -30.0)]
                .iter()
                .all(|d| wg.water_level(x + d.0, z + d.1) == Some(l));
            if wide && lake.is_none() {
                lake = Some((x, z, l));
            } else if !wide && river.is_none() {
                river = Some((x, z, l));
            }
        }
        if river.is_some() && lake.is_some() && fall.is_some() {
            break;
        }
    }
    println!("river {river:?}\nlake {lake:?}\nwaterfall {fall:?}");
    // A cave just above the lava sea: open air at -150 with open air below down to the lava.
    'search: for i in 0..20000 {
        let (x, z) = ((i % 141) as f32 * 9.0 - 600.0, (i / 141) as f32 * 9.0 - 600.0);
        for y in [-150.0f32, -152.0, -154.0] {
            if wg.sample(Vec3::new(x, y, z)).0 > 1.0
                && (y as i32 - 1..-158).rev().all(|_| true)
                && wg.sample(Vec3::new(x, -158.0, z)).0 > 0.0
                && wg.sample(Vec3::new(x, -140.0, z)).0 > 0.0
            {
                println!("lava cave ({x}, {y}, {z})");
                break 'search;
            }
        }
    }
}
