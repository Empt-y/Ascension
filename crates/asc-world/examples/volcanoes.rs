//! Lists the volcanoes within REACH (default 8 km) of the origin, with their crater, lava and
//! features. `SEED=.. REACH=.. cargo run --release -p asc-world --example volcanoes`.
use asc_world::volcano::VolcanoKind;
use asc_world::{Material, WorldGen};
use glam::Vec2;

fn main() {
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1337);
    let reach: f32 = std::env::var("REACH").ok().and_then(|s| s.parse().ok()).unwrap_or(8000.0);
    let wg = WorldGen::new(seed);
    let mut list = wg.volcanoes_near(Vec2::ZERO, reach);
    list.sort_by(|a, b| a.center.length().total_cmp(&b.center.length()));
    let count = |k: VolcanoKind| list.iter().filter(|v| v.kind == k).count();
    println!(
        "seed {seed}: {} volcanoes within {:.0} km ({} active, {} dormant, {} supervolcanoes)",
        list.len(),
        reach / 1000.0,
        count(VolcanoKind::Active),
        count(VolcanoKind::Dormant),
        count(VolcanoKind::Super)
    );
    for v in &list {
        let c = v.center;
        let summit = wg.height(c.x + v.crater_radius, c.y);
        let floor = wg.height(c.x, c.y);
        println!(
            "{:?} at ({:.0}, {:.0}): radius {:.0} m, base {:.0}, rim {:.0}, crater floor {:.0} (r {:.0}){}{}",
            v.kind,
            c.x,
            c.y,
            v.radius,
            v.base,
            summit,
            floor,
            v.crater_radius,
            v.lava.map_or(String::new(), |l| format!(", lava lake at {l:.0} (here {:?})", wg.lava_level(c.x, c.y).map(|l| l.round()))),
            v.crater_lake.map_or(String::new(), |l| format!(", lake at {l:.0}")),
        );
        if v.kind == VolcanoKind::Super {
            for f in &v.features {
                let p = f.at;
                println!(
                    "    {:?} at ({:.0}, {:.0}) r {:.0}: ground {:.1}, water {:?}, lava {:?}, hot {}, surface {:?}",
                    f.kind,
                    p.x,
                    p.y,
                    f.radius,
                    f.ground,
                    wg.water_level(p.x, p.y).map(|l| (l * 10.0).round() / 10.0),
                    wg.lava_level(p.x, p.y).map(|l| (l * 10.0).round() / 10.0),
                    wg.hot_spring(p.x, p.y),
                    wg.surface_material(p.x + f.radius * 1.3, p.y),
                );
            }
        } else {
            let flank = Vec2::new(c.x + v.radius * 0.5, c.y);
            let m: Material = wg.surface_material(flank.x, flank.y);
            println!("    flank surface {m:?}, ground barren {:.2}", wg.ground(flank.x, flank.y).barren);
        }
    }
}
