//! How common each ore vein is, how deep, and where: a table for tuning `assets/data/ores.toml`.
//!
//!     cargo run --release -p asc-world --example orestats -- [seed] [km across]

use std::collections::HashMap;

use asc_world::ores::{OreFile, OreTable, VEIN_CELL};
use asc_world::{Biome, WorldGen};
use glam::IVec2;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let seed: i32 = args.first().and_then(|s| s.parse().ok()).unwrap_or(1337);
    let km: f32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(40.0);
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/data/ores.toml"))
        .expect("reading ores.toml");
    let file: OreFile = toml::from_str(&text).expect("parsing ores.toml");
    let table = OreTable::new(file).expect("resolving ores.toml");
    let worldgen = WorldGen::new(seed);

    let cells = (km * 1000.0 / VEIN_CELL) as i32;
    let half = cells / 2;
    let mut count = vec![0usize; table.veins.len()];
    let mut depth_sum = vec![0.0f32; table.veins.len()];
    let mut by_biome: Vec<HashMap<Biome, usize>> = vec![HashMap::new(); table.veins.len()];
    let mut biome_veins: HashMap<Biome, usize> = HashMap::new();
    let mut shallow = 0;
    let mut total = 0;
    for j in -half..half {
        for i in -half..half {
            for v in table.veins_in_cell(&worldgen, IVec2::new(i, j)) {
                let g = worldgen.ground(v.center.x, v.center.z);
                count[v.kind] += 1;
                depth_sum[v.kind] += g.height - v.center.y;
                *by_biome[v.kind].entry(g.biome).or_default() += 1;
                *biome_veins.entry(g.biome).or_default() += 1;
                total += 1;
                if !table.surface_hints(&worldgen, &v).is_empty() {
                    shallow += 1;
                }
            }
        }
    }
    let area = km * km;
    println!(
        "seed {seed}, {km} km x {km} km: {total} veins ({:.1} per km2), {shallow} show at the surface\n",
        total as f32 / area
    );
    println!(
        "{:<11} {:>12} {:>7} {:>7} {:>9} {:>8}  favoured in (vs its average)",
        "vein", "crust ppm", "weight", "share", "per km2", "depth"
    );
    let mut order: Vec<usize> = (0..table.veins.len()).collect();
    order.sort_by(|a, b| count[*b].cmp(&count[*a]));
    for k in order {
        let v = &table.veins[k];
        let n = count[k].max(1) as f32;
        let share = count[k] as f32 / total.max(1) as f32;
        // The biome where this vein makes up the largest share of veins, relative to its overall share.
        let best = by_biome[k]
            .iter()
            .filter(|(b, c)| **c >= 5 && biome_veins[*b] >= 50)
            .map(|(b, c)| (*b, *c as f32 / biome_veins[b] as f32 / share.max(1e-9)))
            .max_by(|a, b| a.1.total_cmp(&b.1));
        println!(
            "{:<11} {:>12} {:>7.2} {:>6.2}% {:>9.3} {:>6.0} m  {}",
            v.def.id,
            v.def.abundance_ppm,
            v.weight,
            share * 100.0,
            count[k] as f32 / area,
            depth_sum[k] / n,
            best.map_or("-".into(), |(b, r)| format!("{} x{r:.1}", b.name())),
        );
    }
}
