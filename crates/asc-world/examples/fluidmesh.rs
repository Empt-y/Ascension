//! The natural water mesh of the chunk at a point: `SEED=.. X=.. Y=.. Z=.. cargo run --release -p
//! asc-world --example fluidmesh`. Prints vertex counts and the depth / flow bytes seen.
use asc_world::{WorldGen, chunk_of, fluid};
use glam::Vec3;

fn main() {
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1337);
    let wg = WorldGen::new(seed);
    let p = Vec3::new(env("X", 0.0), env("Y", 0.0), env("Z", 0.0));
    println!("height {:?}", wg.height_and_water(p.x, p.z));
    for dy in -1..=1 {
        let coord = chunk_of(p) + glam::IVec3::Y * dy;
        let natural = wg.generate(coord);
        let mesh = fluid::natural_mesh(&wg, &natural, &natural);
        let water: Vec<_> = mesh.vertices.iter().filter(|v| v.weights[0] > 128).collect();
        let mut depths: Vec<u8> = water.iter().map(|v| v.ore[3]).collect();
        depths.sort();
        let ys: Vec<f32> = water.iter().map(|v| v.position[1] + (coord.y * 32) as f32).collect();
        let (lo, hi) = ys.iter().fold((f32::MAX, f32::MIN), |(a, b), y| (a.min(*y), b.max(*y)));
        println!(
            "chunk {coord}: {} water verts, y {lo:.1}..{hi:.1}, depth byte p10 {:?} p50 {:?} p90 {:?}",
            water.len(),
            depths.get(depths.len() / 10),
            depths.get(depths.len() / 2),
            depths.get(depths.len() * 9 / 10)
        );
    }
}
