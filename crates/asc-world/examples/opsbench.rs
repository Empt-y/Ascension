//! Costs that size gameplay budgets: `cargo run --release -p asc-world --example opsbench`.

use std::time::Instant;

use asc_world::{EditKind, EditOp, Material, World, chunk_of};
use glam::Vec3;

fn main() {
    let mut world = World::new(1337);
    let h = world.worldgen.height(10.0, 10.0);
    let center = Vec3::new(10.0, h - 3.0, 10.0);

    // A heavily mined chunk: 1000 small bites inside one 32 m cell.
    for i in 0..1000 {
        let t = i as f32;
        let p = center + Vec3::new((t * 0.37) % 24.0 - 12.0, -(t * 0.011), (t * 0.53) % 24.0 - 12.0);
        world.edits.push(EditOp { center: p, radius: 0.9, kind: EditKind::Dig, material: Material::Air });
    }
    let coord = chunk_of(center);
    let ops = world.edits.ops_for(coord).count();
    let t = Instant::now();
    let runs = 20;
    for _ in 0..runs {
        std::hint::black_box(world.build_chunk(coord));
    }
    println!("chunk with {ops} ops builds in {:.2} ms", t.elapsed().as_secs_f64() * 1e3 / runs as f64);

    let n = 20_000;
    let t = Instant::now();
    let mut acc = 0.0;
    for i in 0..n {
        let p = center + Vec3::new(i as f32 * 0.013, (i % 7) as f32 * 0.5, 0.0);
        acc += world.density_at(p);
    }
    std::hint::black_box(acc);
    println!(
        "density_at: {:.2} us per call (in a chunk with {ops} ops)",
        t.elapsed().as_secs_f64() * 1e6 / n as f64
    );

    let t = Instant::now();
    let c = world.census(center, 0.9, 0.25);
    println!("census r=0.9: {:.2} ms, {:.2} m3 solid", t.elapsed().as_secs_f64() * 1e3, c.total_solid());
}
