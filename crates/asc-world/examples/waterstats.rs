//! Natural water over a patch of world: how much is sea, lake and river, and whether any of it
//! stands above the ground beside it. `cargo run --release -p asc-world --example waterstats`.

use asc_world::{SEA_LEVEL, WorldGen};

fn main() {
    let wg = WorldGen::new(1337);
    let (step, half) = (8.0f32, 4000.0f32);
    let (mut sea, mut fresh, mut land, mut spills, mut checked) = (0u32, 0u32, 0u32, 0u32, 0u32);
    let mut levels = std::collections::BTreeMap::new();
    let mut z = -half;
    while z < half {
        let mut x = -half;
        while x < half {
            match wg.height_and_water(x, z) {
                (_, Some(l)) if l == SEA_LEVEL => sea += 1,
                (_, Some(l)) => {
                    fresh += 1;
                    *levels.entry((l / 10.0) as i32 * 10).or_insert(0u32) += 1;
                    // At the edge of the water, the ground next to it must be at or above the water.
                    for (dx, dz) in [(1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0)] {
                        let (h, w) = wg.height_and_water(x + dx, z + dz);
                        if w.is_none() {
                            checked += 1;
                            if h < l - 0.3 {
                                spills += 1;
                                if spills <= 6 {
                                    println!("  spill at ({:.0}, {:.0}): water {l:.1}, ground {:.1} beside it at {:.1}", x, z, wg.height(x, z), h);
                                }
                            }
                        }
                    }
                }
                _ => land += 1,
            }
            x += step;
        }
        z += step;
    }
    let total = (sea + fresh + land) as f32;
    println!(
        "{:.0} km2: sea {:.1}%, lakes and rivers {:.2}%, land {:.1}%",
        (2.0 * half / 1000.0).powi(2),
        100.0 * sea as f32 / total,
        100.0 * fresh as f32 / total,
        100.0 * land as f32 / total
    );
    println!("fresh water by surface height (m): {levels:?}");
    println!("water edges checked {checked}, standing above the ground beside them {spills} ({:.2}%)", 100.0 * spills as f32 / checked.max(1) as f32);
}
