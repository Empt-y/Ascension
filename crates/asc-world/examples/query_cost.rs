//! How long the terrain queries take (for budgeting the navigation grid).
use std::time::Instant;

use asc_world::{World, nature::Nature};
use glam::Vec3;

fn main() {
    let world = World::new(1337);
    let wg = &world.worldgen;
    let n = 20_000;
    let pts: Vec<(f32, f32)> = (0..n).map(|i| ((i % 141) as f32 * 1.37 + 300.0, (i / 141) as f32 * 1.11 - 200.0)).collect();
    let time = |name: &str, f: &dyn Fn(f32, f32) -> f32| {
        let t = Instant::now();
        let mut acc = 0.0;
        for &(x, z) in &pts {
            acc += f(x, z);
        }
        println!("{name:>14}: {:6.2} us  ({acc:.0})", t.elapsed().as_secs_f64() * 1e6 / n as f64);
    };
    time("height", &|x, z| wg.height(x, z));
    time("ground", &|x, z| wg.ground(x, z).height);
    time("water_level", &|x, z| wg.water_level(x, z).unwrap_or(0.0));
    time("sample", &|x, z| wg.sample(Vec3::new(x, wg.height(x, z), z)).0);
    time("density_at", &|x, z| world.density_at(Vec3::new(x, 10.0, z)));
    let _ = Nature::builtin();
}
