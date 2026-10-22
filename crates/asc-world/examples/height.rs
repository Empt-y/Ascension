//! Prints the terrain height and natural water at x, z: `cargo run -p asc-world --example height -- 230 -180 [seed]`.
fn main() {
    let a: Vec<f32> = std::env::args().skip(1).map(|v| v.parse().expect("numbers")).collect();
    let seed = a.get(2).map_or(1337, |s| *s as i32);
    let wg = asc_world::WorldGen::new(seed);
    let g = wg.ground(a[0], a[1]);
    println!("height {} water {:?} biome {:?}", g.height, wg.water_level(a[0], a[1]), g.biome);
}
