//! Dig into a riverbed under water, run the fluid simulation, and compare the simulated water
//! surface with the river's natural level column by column (and look for dry pockets left under
//! the surface). `SEED=.. cargo run --release -p asc-sim --example riverdig`.
use asc_sim::fluids::FluidSim;
use asc_world::fluid::{FULL, FluidKind};
use asc_world::{EditKind, EditOp, Material, World};
use glam::{IVec3, Vec3};

fn main() {
    let seed = std::env::var("SEED").ok().and_then(|s| s.parse().ok()).unwrap_or(1337);
    let build = std::env::var("BUILD").is_ok();
    let mut world = World::new(seed);
    let wg = world.worldgen.clone();
    // Deep river water above sea level.
    let (x, z, level) = (0..60000)
        .map(|i| ((i % 300) as f32 * 2.0 - 300.0, (i / 300) as f32 * 2.0 - 200.0))
        .find_map(|(x, z)| {
            let l = wg.water_level(x, z)?;
            let h = wg.height(x, z);
            (l > asc_world::SEA_LEVEL + 4.0 && l - h > 1.5).then_some((x, z, l))
        })
        .expect("a river");
    let bed = wg.height(x, z);
    println!("river at ({x}, {z}): level {level:.2}, bed {bed:.2}");
    let mut sim = FluidSim::new();
    let edit = |world: &mut World, sim: &mut FluidSim, center: Vec3, kind: EditKind, material: Material| {
        let op = EditOp { center, radius: 1.6, kind, material };
        world.edits.push(op);
        sim.disturb(world, &op);
    };
    if build {
        // Build a lump in the water, let it settle, then dig it out again.
        edit(&mut world, &mut sim, Vec3::new(x, bed + 0.5, z), EditKind::Fill, Material::Rock);
        for _ in 0..200 {
            sim.tick(&world);
        }
    }
    edit(&mut world, &mut sim, Vec3::new(x, bed - 0.5, z), EditKind::Dig, Material::Air);
    edit(&mut world, &mut sim, Vec3::new(x + 1.5, bed - 0.5, z), EditKind::Dig, Material::Air);
    for t in 0..=1200 {
        if t % 300 == 0 {
            report(&world, &sim, x, z, t);
        }
        sim.tick(&world);
    }
}

fn report(world: &World, sim: &FluidSim, x: f32, z: f32, t: u32) {
    let wg = &world.worldgen;
    let (mut worst, mut sum, mut n, mut pockets) = (0.0f32, 0.0f32, 0, 0);
    for dz in -12..=12 {
        for dx in -12..=12 {
            let (cx, cz) = (x as i32 + dx, z as i32 + dz);
            let Some(natural) = wg.water_level(cx as f32, cz as f32) else { continue };
            // The simulated surface: top of the highest water cell under natural + 3.
            let mut surface = None;
            let mut y = (natural + 3.0).floor() as i32;
            while y > natural as i32 - 12 {
                let (k, a) = sim.fluids.get(&world.worldgen, IVec3::new(cx, y, cz));
                if k == FluidKind::Water && a > 0 {
                    surface = Some(y as f32 - 0.5 + a.min(FULL) as f32 / FULL as f32);
                    break;
                }
                y -= 1;
            }
            let Some(s) = surface else { continue };
            // Open cells under the surface without water: pockets.
            for yy in (natural as i32 - 10)..(s.floor() as i32) {
                let c = IVec3::new(cx, yy, cz);
                let open = world.density_at(c.as_vec3()) > 0.0;
                let (k, a) = sim.fluids.get(&world.worldgen, c);
                if open && !(k == FluidKind::Water && a > FULL / 2) {
                    pockets += 1;
                }
            }
            let d = s - natural;
            worst = if d.abs() > worst.abs() { d } else { worst };
            sum += d;
            n += 1;
        }
    }
    println!(
        "t={t}: {} materialised, surface - natural: mean {:.2} worst {:.2} over {n} columns; {pockets} dry open cells under the surface",
        sim.fluids.chunks.len(),
        sum / n.max(1) as f32,
        worst
    );
}
