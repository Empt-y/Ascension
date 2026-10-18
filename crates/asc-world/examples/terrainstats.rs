//! Terrain shape statistics over a 2 km square: `cargo run --release -p asc-world --example terrainstats`.
//!
//! - surface openings: columns where the ground 1.5 m below the height field is air (a cave or
//!   ravine breaking the surface)
//! - floating: columns with solid ground above an air gap above the height field (overhang lips)
//! - underground caves: share of samples 10-60 m below the surface that are air

use asc_world::World;
use glam::Vec3;

fn main() {
    let world = World::new(1337);
    let (mut columns, mut openings, mut floating) = (0u32, 0u32, 0u32);
    let (mut deep, mut deep_air) = (0u32, 0u32);
    let step = 8.0;
    let n = (2000.0 / step) as i32;
    for iz in 0..n {
        for ix in 0..n {
            let (x, z) = (ix as f32 * step - 1000.0, iz as f32 * step - 1000.0);
            let h = world.worldgen.height(x, z);
            columns += 1;
            if world.density_at(Vec3::new(x, h - 1.5, z)) >= 0.0 {
                openings += 1;
            }
            // Solid somewhere 3-12 m above the height field, with air between it and the surface.
            let solid_above = (3..12).any(|dy| world.density_at(Vec3::new(x, h + dy as f32, z)) < 0.0);
            let gap = world.density_at(Vec3::new(x, h + 1.5, z)) >= 0.0;
            if solid_above && gap {
                floating += 1;
            }
            if ix % 4 == 0 && iz % 4 == 0 {
                for d in (10..60).step_by(5) {
                    deep += 1;
                    if world.density_at(Vec3::new(x, h - d as f32, z)) >= 0.0 {
                        deep_air += 1;
                    }
                }
            }
        }
    }
    let pct = |a: u32, b: u32| 100.0 * a as f32 / b.max(1) as f32;
    println!(
        "surface openings {:.2}% | floating lips {:.2}% | underground air {:.1}% ({columns} columns)",
        pct(openings, columns),
        pct(floating, columns),
        pct(deep_air, deep)
    );
}
