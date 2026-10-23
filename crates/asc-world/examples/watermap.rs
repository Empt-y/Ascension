//! ASCII map of natural water depth around a point: `SEED=.. X=.. Z=.. cargo run --release -p
//! asc-world --example watermap`. '.' dry, digits = depth in 0.25 m steps (9 = 2.25 m+),
//! '~' sea.
use asc_world::{SEA_LEVEL, WorldGen};

fn main() {
    let env = |k: &str, d: f32| std::env::var(k).ok().and_then(|s| s.parse().ok()).unwrap_or(d);
    let wg = WorldGen::new(std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1337));
    let (cx, cz, step) = (env("X", 0.0), env("Z", 0.0), env("STEP", 2.0));
    for j in -30..30 {
        let line: String = (-50..50)
            .map(|i| {
                let (x, z) = (cx + i as f32 * step, cz + j as f32 * step);
                match wg.height_and_water(x, z) {
                    (_, None) => '.',
                    (_, Some(l)) if l == SEA_LEVEL => '~',
                    (h, Some(l)) => char::from_digit(((l - h) / 0.25).min(9.0) as u32, 10).unwrap(),
                }
            })
            .collect();
        println!("{line}");
    }
}
