//! Compares a save's materialised fluid chunks with their pristine water: chunks that are much
//! drier (or wetter) than nature would have them. `cargo run --release -p asc-sim --example
//! fluidsave -- <save dir>`.
use asc_world::fluid::{self, FluidKind};
use asc_world::World;

fn main() {
    let dir = std::env::args().nth(1).expect("a save directory");
    let save = asc_sim::save::load(std::path::Path::new(&dir)).expect("loading").expect("a save");
    let mut world = World::new(save.meta.seed);
    for op in &save.ops {
        world.edits.push(*op);
    }
    println!("{} fluid chunks saved", save.fluids.len());
    for (coord, kind, amount) in &save.fluids {
        let pristine = fluid::pristine_chunk(&world.worldgen, &world.build_chunk(*coord));
        let water = |k: &[u8], a: &[u16]| -> u64 {
            (0..fluid::CELLS).filter(|&i| k[i] == FluidKind::Water as u8).map(|i| a[i] as u64).sum()
        };
        let (saved, natural) = (water(kind, amount), water(&pristine.kind, &pristine.amount));
        let marker = if natural > 0 && saved * 2 < natural { "  <-- much drier than nature" } else { "" };
        println!("chunk {coord}: water {} cells saved, {} natural{marker}", saved / 1000, natural / 1000);
    }
}
