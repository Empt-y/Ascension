//! Times point density queries (what physics and raycasts use): `cargo run --release --example densitybench`.

use std::time::Instant;

use asc_world::World;
use glam::Vec3;

fn main() {
    let world = World::new(1337);
    let mut sum = 0.0;
    for (label, depth) in [("surface", 0.0f32), ("10 m down", 10.0), ("40 m down", 40.0)] {
        let t = Instant::now();
        let n = 200_000;
        for i in 0..n {
            let (x, z) = ((i % 400) as f32 * 0.37, (i / 400) as f32 * 0.41);
            let y = world.worldgen.height(x, z) - depth;
            sum += world.density_at(Vec3::new(x, y, z));
        }
        let per = t.elapsed().as_secs_f64() / n as f64 * 1e6;
        println!("{label}: {per:.2} us per density_at");
    }
    println!("({sum})");
}
