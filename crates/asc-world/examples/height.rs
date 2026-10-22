//! Prints the terrain height at x, z: `cargo run -p asc-world --example height -- 230 -180 [seed]`.
fn main() {
    let a: Vec<f32> = std::env::args().skip(1).map(|v| v.parse().expect("numbers")).collect();
    let seed = a.get(2).map_or(1337, |s| *s as i32);
    let g = asc_world::WorldGen::new(seed).ground(a[0], a[1]);
    println!("{} {:?}", g.height, g.biome);
}
