//! Cross-section of the natural water mesh across a chunk at world z = Z: every water triangle
//! edge crossing that plane, as (x, y), sorted by x; with the ground and water level per column.
//! `SEED=.. X=.. Y=.. Z=.. cargo run --release -p asc-world --example fluidslice`.
use asc_world::{CHUNK, WorldGen, chunk_of, fluid};
use glam::Vec3;

fn main() {
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1337);
    let wg = WorldGen::new(seed);
    let p = Vec3::new(env("X", 0.0), env("Y", 0.0), env("Z", 0.0));
    let coord = chunk_of(p);
    let natural = wg.generate(coord);
    let mesh = fluid::natural_mesh(&wg, &natural, &natural);
    let o = (coord * CHUNK).as_vec3();
    let mut pts = Vec::new();
    for t in mesh.indices.chunks(3) {
        let v: Vec<Vec3> = t.iter().map(|&i| o + Vec3::from(mesh.vertices[i as usize].position)).collect();
        for (a, b) in [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])] {
            if (a.z - p.z) * (b.z - p.z) < 0.0 {
                let s = (p.z - a.z) / (b.z - a.z);
                let q = a.lerp(b, s);
                pts.push((q.x, q.y));
            }
        }
    }
    pts.sort_by(|a, b| a.0.total_cmp(&b.0));
    let x0 = env("X0", o.x - 1.0);
    let x1 = env("X1", o.x + 33.0);
    for x in (x0 as i32)..=(x1 as i32) {
        let (h, w) = wg.height_and_water(x as f32, p.z);
        let ys: Vec<String> = pts
            .iter()
            .filter(|(px, _)| (*px - x as f32).abs() < 0.5)
            .map(|(_, y)| format!("{y:.2}"))
            .collect();
        println!("x {x:4}: ground {h:6.2} water {:>6}  mesh y: {}", w.map_or("-".into(), |w| format!("{w:.2}")), ys.join(" "));
    }
}
