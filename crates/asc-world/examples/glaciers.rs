//! Where the glaciers are near the origin: how much ground they cover, how thick they get, and
//! samples of their crevasses and ice caves. `SEED=.. REACH=.. cargo run --release -p asc-world
//! --example glaciers`.
use asc_world::{Material, WorldGen};
use glam::Vec3;

fn main() {
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1337);
    let reach: f32 = std::env::var("REACH").ok().and_then(|s| s.parse().ok()).unwrap_or(8000.0);
    let wg = WorldGen::new(seed);
    let step = 64.0;
    let n = (reach / step) as i32;
    let (mut samples, mut iced, mut thick, mut max) = (0, 0, 0, (0.0f32, 0.0f32, 0.0f32));
    for j in -n..=n {
        for i in -n..=n {
            let (x, z) = (i as f32 * step, j as f32 * step);
            samples += 1;
            let t = wg.ice_thickness(x, z);
            if t > 0.5 {
                iced += 1;
            }
            if t > 20.0 {
                thick += 1;
            }
            if t > max.0 {
                max = (t, x, z);
            }
        }
    }
    println!(
        "seed {seed}: ice on {:.1}% of {samples} samples ({:.1}% over 20 m); thickest {:.0} m at ({:.0}, {:.0})",
        iced as f32 / samples as f32 * 100.0,
        thick as f32 / samples as f32 * 100.0,
        max.0,
        max.1,
        max.2
    );
    if max.0 <= 0.0 {
        return;
    }
    // Round the thickest point: crevasses and caves in the ice.
    let (cx, cz) = (max.1, max.2);
    let (mut columns, mut cracked, mut open, mut ice_cells) = (0, 0, 0, 0);
    for dz in -40..40 {
        for dx in -40..40 {
            let (x, z) = (cx + dx as f32, cz + dz as f32);
            let ice = wg.ice_thickness(x, z);
            if ice < 14.0 {
                continue;
            }
            columns += 1;
            let top = wg.height(x, z);
            let line = wg.line(x, z);
            let near_top = line(top - 1.5).0;
            if near_top > 0.0 {
                cracked += 1;
            }
            for k in 3..(ice as i32 - 2) {
                let (d, m) = line(top - k as f32);
                if d > 0.0 {
                    open += 1;
                } else if m == Material::Ice as u8 {
                    ice_cells += 1;
                }
            }
        }
    }
    println!(
        "  80 m round it: {columns} columns of thick ice, {cracked} cracked open at 1.5 m down (crevasses), {:.1}% of the ice inside them open (caves)",
        open as f32 / (open + ice_cells).max(1) as f32 * 100.0
    );
    println!("  surface: {:?}", wg.surface_material(cx, cz));
    let _ = Vec3::ZERO;
}
