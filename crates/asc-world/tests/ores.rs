use std::collections::HashMap;

use asc_world::ores::{Layer, OreFile, OreTable, Vein};
use asc_world::{Biome, WorldGen};
use glam::{IVec2, Vec3};

fn table() -> OreTable {
    let text =
        std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/../../assets/data/ores.toml")).unwrap();
    let file: OreFile = toml::from_str(&text).unwrap();
    OreTable::new(file).unwrap()
}

fn sample_veins(table: &OreTable, worldgen: &WorldGen, cells: i32) -> Vec<Vein> {
    let mut out = Vec::new();
    for j in 0..cells {
        for i in 0..cells {
            // Spread out, so many climates are covered.
            out.extend(table.veins_in_cell(worldgen, IVec2::new(i * 7 - cells * 3, j * 7 - cells * 3)));
        }
    }
    out
}

#[test]
fn the_ore_file_resolves() {
    let t = table();
    assert!(t.ores.len() >= 40 && t.veins.len() >= 20);
    for v in &t.veins {
        assert!(v.weight > 0.0, "{}", v.def.id);
    }
}

#[test]
fn veins_are_deterministic_and_within_their_depths() {
    let t = table();
    let worldgen = WorldGen::new(1337);
    for c in [IVec2::new(0, 0), IVec2::new(-17, 40), IVec2::new(300, -2)] {
        let a = t.veins_in_cell(&worldgen, c);
        assert_eq!(a, t.veins_in_cell(&worldgen, c));
        for v in a {
            let depth = worldgen.ground(v.center.x, v.center.z).height - v.center.y;
            let def = &t.veins[v.kind].def;
            assert!(depth >= def.depth[0] - 1e-3 && depth <= def.depth[1] + 1e-3, "{} at {depth}", def.id);
        }
    }
}

#[test]
fn common_elements_make_common_veins() {
    let t = table();
    let worldgen = WorldGen::new(1337);
    let veins = sample_veins(&t, &worldgen, 140);
    let mut count: HashMap<&str, usize> = HashMap::new();
    for v in &veins {
        *count.entry(t.veins[v.kind].def.id.as_str()).or_default() += 1;
    }
    let n = |id: &str| *count.get(id).unwrap_or(&0);
    let mut sorted: Vec<_> = count.iter().collect();
    sorted.sort_by(|a, b| b.1.cmp(a.1));
    println!("{} veins: {sorted:?}", veins.len());
    let iron = n("limonite") + n("magnetite");
    assert!(iron > 10 * n("platinum").max(1), "iron {iron} vs platinum {}", n("platinum"));
    assert!(n("copper") > n("tin"), "copper {} tin {}", n("copper"), n("tin"));
    assert!(n("diamond") > 0 && n("diamond") < n("chromite"), "diamond {}", n("diamond"));
}

#[test]
fn deposits_favour_where_they_form() {
    let t = table();
    let worldgen = WorldGen::new(1337);
    let bauxite = t.veins.iter().position(|v| v.def.id == "bauxite").unwrap();
    let mut by_biome: HashMap<Biome, (usize, usize)> = HashMap::new();
    for v in sample_veins(&t, &worldgen, 160) {
        if worldgen.ground(v.center.x, v.center.z).height - v.center.y > 90.0 {
            continue;
        }
        let biome = worldgen.ground(v.center.x, v.center.z).biome;
        let e = by_biome.entry(biome).or_default();
        e.0 += usize::from(v.kind == bauxite);
        e.1 += 1;
    }
    let share = |b: Biome| {
        let (k, n) = by_biome.get(&b).copied().unwrap_or((0, 0));
        k as f32 / n.max(1) as f32
    };
    println!("bauxite share: jungle {:.3}, tundra {:.3}", share(Biome::Jungle), share(Biome::Tundra));
    assert!(share(Biome::Jungle) > 4.0 * share(Biome::Tundra));
}

#[test]
fn veins_are_layered_top_to_bottom() {
    let vein = Vein { kind: 0, center: Vec3::ZERO, radius: 20.0, thickness: 6.0, normal: Vec3::Y, seed: 99 };
    let mut seen = HashMap::new();
    for i in 0..4000 {
        let (x, z) = ((i % 63) as f32 * 0.3 - 9.0, (i / 63) as f32 * 0.3 - 9.0);
        for (y, expect) in [(2.5, Layer::Primary), (0.0, Layer::Between), (-2.5, Layer::Secondary)] {
            if let Some(layer) = vein.layer_at(Vec3::new(x, y, z)) {
                if layer != Layer::Sporadic {
                    assert_eq!(layer, expect, "at y {y}");
                }
                *seen.entry(layer).or_insert(0) += 1;
            }
        }
    }
    assert!(seen.len() == 4, "{seen:?}");
    assert!(vein.layer_at(Vec3::new(0.0, 3.5, 0.0)).is_none());
    assert!(vein.layer_at(Vec3::new(21.0, 0.0, 0.0)).is_none());
}

#[test]
fn shallow_veins_show_at_the_surface() {
    let t = table();
    let worldgen = WorldGen::new(1337);
    let mut shallow = 0;
    let mut hints = 0;
    for v in sample_veins(&t, &worldgen, 60) {
        let h = t.surface_hints(&worldgen, &v);
        let top_depth = worldgen.height(v.center.x, v.center.z) - (v.center.y + v.thickness * 0.5);
        if top_depth <= asc_world::ores::HINT_DEPTH {
            shallow += 1;
            assert!(!h.is_empty());
            for (ore, p) in &h {
                let def = &t.veins[v.kind];
                assert!(*ore == def.ores[0] || *ore == def.ores[1]);
                assert!((p.y - worldgen.height(p.x, p.z)).abs() < 1e-3);
            }
            hints += h.len();
        } else {
            assert!(h.is_empty());
        }
    }
    assert!(shallow > 20, "{shallow}");
    println!("{shallow} shallow veins, {hints} surface hints");
}
