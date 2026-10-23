//! How far the built terrain's surface under natural water strays from the column height worldgen
//! reports (the water mesh's assumptions). `cargo run --release -p asc-world --example waterbed`.
use asc_world::{CHUNK, ChunkData, SAMPLES, WorldGen, chunk_of};
use glam::Vec3;

fn main() {
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1256132528);
    let wg = WorldGen::new(seed);
    let mut diffs = Vec::new();
    for i in 0..400 {
        let (x, z) = ((i % 20) as f32 * 97.0 - 1000.0, (i / 20) as f32 * 89.0 - 900.0);
        let Some(level) = wg.water_level(x, z) else { continue };
        let coord = chunk_of(Vec3::new(x, level - 2.0, z));
        let ch = wg.generate(coord);
        let base = ChunkData::sample_pos(coord, 0, 0, 0);
        for cz in (1..SAMPLES - 1).step_by(3) {
            for cx in (1..SAMPLES - 1).step_by(3) {
                let col = cx + SAMPLES * cz;
                if ch.water[col].is_nan() {
                    continue;
                }
                let ground = ch.surface[col];
                // Highest crossing from open (above) to solid (below).
                for y in (0..SAMPLES - 1).rev() {
                    let (a, b) = (ch.density[ChunkData::index(cx, y, cz)], ch.density[ChunkData::index(cx, y + 1, cz)]);
                    if a < 0.0 && b >= 0.0 {
                        let real = base.y + y as f32 + a / (a - b);
                        diffs.push(real - ground);
                        break;
                    }
                }
            }
        }
    }
    let _ = CHUNK;
    diffs.sort_by(f32::total_cmp);
    let n = diffs.len();
    let below = |t: f32| diffs.iter().filter(|d| **d < t).count() as f32 / n as f32 * 100.0;
    println!("{n} wet columns: real - column height: min {:.2} p5 {:.2} median {:.2} p95 {:.2} max {:.2}", diffs[0], diffs[n / 20], diffs[n / 2], diffs[n * 19 / 20], diffs[n - 1]);
    println!("more than 1.5 m below: {:.1}%, more than 3 m below: {:.1}%", below(-1.5), below(-3.0));
}
