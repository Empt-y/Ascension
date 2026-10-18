//! Paint a top-down biome map with hill shading, to check climate and biome layout.
//!
//!     cargo run --release -p asc-world --example biomemap -- [seed] [km across] [out.ppm]
//!
//! Writes a binary PPM (convert with any image tool); the world origin is at the centre.

use asc_world::{Biome, WorldGen};

fn color(b: Biome) -> [f32; 3] {
    match b {
        Biome::Arctic => [0.93, 0.95, 1.0],
        Biome::Tundra => [0.62, 0.64, 0.52],
        Biome::Taiga => [0.16, 0.36, 0.28],
        Biome::Meadow => [0.55, 0.75, 0.32],
        Biome::Forest => [0.2, 0.5, 0.15],
        Biome::Swamp => [0.32, 0.36, 0.2],
        Biome::Desert => [0.9, 0.78, 0.5],
        Biome::Savanna => [0.78, 0.68, 0.32],
        Biome::Jungle => [0.05, 0.42, 0.12],
        Biome::Shore => [0.95, 0.88, 0.62],
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seed: i32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(1337);
    let km: f32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40.0);
    let out = args.get(2).cloned().unwrap_or_else(|| "biomes.ppm".into());
    let size = 800usize;
    let worldgen = WorldGen::new(seed);
    let step = km * 1000.0 / size as f32;
    let mut pixels = Vec::with_capacity(size * size * 3);
    for j in 0..size {
        for i in 0..size {
            let x = (i as f32 - size as f32 / 2.0) * step;
            let z = (j as f32 - size as f32 / 2.0) * step;
            let g = worldgen.ground(x, z);
            let mut c = if g.underwater { [0.12, 0.3, 0.55] } else { color(g.biome) };
            // Light from the north-west.
            let east = worldgen.height(x + step, z) - g.height;
            let south = worldgen.height(x, z + step) - g.height;
            let shade = (1.0 - (east + south) / step * 1.5).clamp(0.55, 1.25);
            for v in &mut c {
                *v = (*v * shade).clamp(0.0, 1.0);
            }
            pixels.extend(c.iter().map(|v| (v * 255.0) as u8));
        }
    }
    let mut file = format!("P6\n{size} {size}\n255\n").into_bytes();
    file.extend(pixels);
    std::fs::write(&out, file).expect("writing map");
    println!("wrote {out}: {km} km across, seed {seed}");
}
