//! Paint a top-down biome map with hill shading, to check climate and biome layout.
//!
//!     cargo run --release -p asc-world --example biomemap -- [seed] [km across] [out.ppm]
//!     cargo run --release -p asc-world --example biomemap -- [seed] --find
//!
//! Writes a binary PPM (convert with any image tool); the world origin is at the centre. With
//! `--find` it instead prints, for each biome, the nearest dry spot well inside it.

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
    if args.get(1).is_some_and(|a| a == "--find") {
        find(&WorldGen::new(seed));
        return;
    }
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

/// For each biome, the nearest point (searching outwards in rings) whose neighbours 60 m away are
/// the same biome too, so it's a fair sample rather than an edge.
fn find(worldgen: &WorldGen) {
    for biome in Biome::ALL {
        let inside = |x: f32, z: f32| {
            let g = worldgen.ground(x, z);
            g.biome == biome && !g.underwater && g.slope < 0.5
        };
        let found = (1..400).find_map(|ring| {
            let r = ring as f32 * 50.0;
            (0..ring * 8).find_map(|k| {
                let a = k as f32 / (ring * 8) as f32 * std::f32::consts::TAU;
                let (x, z) = (r * a.cos(), r * a.sin());
                let around = [(60.0, 0.0), (-60.0, 0.0), (0.0, 60.0), (0.0, -60.0)];
                (inside(x, z) && around.iter().all(|(dx, dz)| inside(x + dx, z + dz))).then_some((x, z))
            })
        });
        match found {
            Some((x, z)) => println!("{:8} {x:9.0} {:6.1} {z:9.0}", biome.name(), worldgen.height(x, z)),
            None => println!("{:8} not found within 20 km", biome.name()),
        }
    }
}
