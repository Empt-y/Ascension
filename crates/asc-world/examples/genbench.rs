//! Times chunk generation and meshing over a patch of terrain: `cargo run --release --example genbench`.

use std::time::Instant;

use asc_world::{World, chunk_of, mesh_chunk};
use glam::{IVec3, Vec3};

fn main() {
    let world = World::new(1337);
    let (mut gen_t, mut mesh_t, mut n, mut tris, mut empty) = (0.0, 0.0, 0, 0, 0);
    for cz in -4..4 {
        for cx in -4..4 {
            let (x, z) = (cx as f32 * 32.0 + 16.0, cz as f32 * 32.0 + 16.0);
            let surface = chunk_of(Vec3::new(x, world.worldgen.height(x, z), z));
            for dy in -2..=1 {
                let coord = surface + IVec3::new(0, dy, 0);
                let t0 = Instant::now();
                let chunk = world.build_chunk(coord);
                let t1 = Instant::now();
                let mesh = mesh_chunk(&chunk);
                let t2 = Instant::now();
                gen_t += (t1 - t0).as_secs_f64();
                mesh_t += (t2 - t1).as_secs_f64();
                n += 1;
                tris += mesh.indices.len() / 3;
                empty += usize::from(mesh.is_empty());
            }
        }
    }
    println!(
        "{n} chunks ({empty} empty): gen {:.2} ms, mesh {:.2} ms per chunk, {} tris per chunk",
        gen_t * 1e3 / n as f64,
        mesh_t * 1e3 / n as f64,
        tris / n
    );
}
