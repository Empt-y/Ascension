//! Do neighbouring chunks' natural water meshes overlap (the same triangle drawn twice)?
//! `SEED=.. X=.. Y=.. Z=.. cargo run --release -p asc-world --example fluidseam`.
use asc_world::{CHUNK, WorldGen, chunk_of, fluid};
use glam::{IVec3, Vec3};

fn main() {
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1337);
    let wg = WorldGen::new(seed);
    let p = Vec3::new(env("X", 0.0), env("Y", 0.0), env("Z", 0.0));
    let c = chunk_of(p);
    let tris = |coord: IVec3| -> Vec<[i64; 9]> {
        let natural = wg.generate(coord);
        let mesh = fluid::natural_mesh(&wg, &natural, &natural);
        let o = (coord * CHUNK).as_vec3();
        let mut out: Vec<[i64; 9]> = mesh
            .indices
            .chunks(3)
            .map(|t| {
                let mut v: Vec<[i64; 3]> = t
                    .iter()
                    .map(|&i| {
                        let q = o + Vec3::from(mesh.vertices[i as usize].position);
                        [(q.x * 64.0).round() as i64, (q.y * 64.0).round() as i64, (q.z * 64.0).round() as i64]
                    })
                    .collect();
                v.sort();
                [v[0][0], v[0][1], v[0][2], v[1][0], v[1][1], v[1][2], v[2][0], v[2][1], v[2][2]]
            })
            .collect();
        out.sort();
        out
    };
    let a = tris(c);
    println!("chunk {c}: {} triangles", a.len());
    for d in [IVec3::X, IVec3::NEG_X, IVec3::Z, IVec3::NEG_Z, IVec3::Y, IVec3::NEG_Y] {
        let b = tris(c + d);
        let shared = a.iter().filter(|t| b.binary_search(t).is_ok()).count();
        println!("  with {}: {} triangles, {shared} drawn by both", c + d, b.len());
    }
}
