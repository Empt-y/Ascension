use ahash::AHashMap;
use asc_world::*;
use glam::{IVec3, Vec3};

const SEED: i32 = 1337;

/// A chunk coordinate whose column passes through the terrain surface near (x, z).
fn surface_chunk(worldgen: &WorldGen, x: f32, z: f32) -> ChunkCoord {
    let h = worldgen.height(x, z);
    chunk_of(Vec3::new(x, h, z))
}

#[test]
fn worldgen_is_deterministic() {
    let a = WorldGen::new(SEED);
    let b = WorldGen::new(SEED);
    let c = WorldGen::new(SEED + 1);
    let coord = surface_chunk(&a, 10.0, 10.0);
    let (ca, cb, cc) = (a.generate(coord), b.generate(coord), c.generate(coord));
    assert_eq!(ca.density, cb.density);
    assert_eq!(ca.material, cb.material);
    assert_ne!(ca.density, cc.density);
}

#[test]
fn point_samples_match_chunk_samples() {
    let worldgen = WorldGen::new(SEED);
    let coord = surface_chunk(&worldgen, -40.0, 70.0);
    let chunk = worldgen.generate(coord);
    for (x, y, z) in [(0, 0, 0), (1, 5, 9), (17, 20, 3), (34, 34, 34), (12, 30, 33)] {
        let p = ChunkData::sample_pos(coord, x, y, z);
        let i = ChunkData::index(x, y, z);
        assert_eq!(worldgen.sample(p), (chunk.density[i], chunk.material[i]), "sample {x},{y},{z}");
    }
}

#[test]
fn uniform_chunks_have_empty_meshes() {
    let n = SAMPLES * SAMPLES * SAMPLES;
    for value in [-1.0f32, 1.0] {
        let mut chunk = ChunkData {
            coord: IVec3::ZERO,
            density: vec![value; n],
            material: vec![Material::Rock as u8; n],
            surface: vec![0.0; SAMPLES * SAMPLES],
            water: vec![f32::NAN; SAMPLES * SAMPLES],
            uniform: None,
        };
        chunk.update_uniform();
        assert!(chunk.uniform.is_some());
        assert!(mesh_chunk(&chunk).is_empty());
    }
    let worldgen = WorldGen::new(SEED);
    let sky = worldgen.generate(IVec3::new(0, 40, 0));
    assert!(mesh_chunk(&sky).is_empty());
}

#[test]
fn surface_chunk_produces_triangles() {
    let worldgen = WorldGen::new(SEED);
    let coord = surface_chunk(&worldgen, 16.0, 16.0);
    let mesh = mesh_chunk(&worldgen.generate(coord));
    assert!(mesh.indices.len() > 300, "only {} indices", mesh.indices.len());
    assert_eq!(mesh.indices.len() % 3, 0);
    assert!(mesh.indices.iter().all(|i| (*i as usize) < mesh.vertices.len()));
}

/// Mesh a 2×2×2 block of chunks and check the union is a closed, consistently wound surface away
/// from the block's outer boundary: every edge is used equally often in each direction. A crack or a
/// mismatched vertex along a chunk border would leave unbalanced edges.
#[test]
fn neighbouring_chunks_join_without_seams() {
    let worldgen = WorldGen::new(SEED);
    let base = surface_chunk(&worldgen, 0.0, 0.0) - IVec3::new(1, 1, 1);
    let lo = chunk_origin(base) + Vec3::splat(2.0);
    let hi = chunk_origin(base + IVec3::splat(2)) - Vec3::splat(2.0);

    // Weld vertices that coincide up to float rounding (chunk-local + origin sums differ by an ulp).
    let mut welded: AHashMap<IVec3, Vec<(Vec3, IVec3)>> = AHashMap::new();
    let mut key = |p: Vec3| {
        let cell = (p * 64.0).floor().as_ivec3();
        for dz in -1..=1 {
            for dy in -1..=1 {
                for dx in -1..=1 {
                    if let Some(list) = welded.get(&(cell + IVec3::new(dx, dy, dz)))
                        && let Some((_, id)) = list.iter().find(|(q, _)| q.distance(p) < 1e-3)
                    {
                        return *id;
                    }
                }
            }
        }
        let id = (p * 4096.0).round().as_ivec3();
        welded.entry(cell).or_default().push((p, id));
        id
    };
    let mut directed: AHashMap<(IVec3, IVec3), i32> = AHashMap::new();
    let mut triangles = 0;
    for dz in 0..2 {
        for dy in 0..2 {
            for dx in 0..2 {
                let coord = base + IVec3::new(dx, dy, dz);
                let mesh = mesh_chunk(&worldgen.generate(coord));
                let origin = chunk_origin(coord);
                let world = |i: u32| origin + Vec3::from(mesh.vertices[i as usize].position);
                for tri in mesh.indices.as_chunks::<3>().0 {
                    let p = tri.map(world);
                    triangles += 1;
                    for k in 0..3 {
                        let (a, b) = (p[k], p[(k + 1) % 3]);
                        let inside = |v: Vec3| v.cmpge(lo).all() && v.cmple(hi).all();
                        if inside(a) && inside(b) {
                            let (ka, kb) = (key(a), key(b));
                            if ka == kb {
                                continue;
                            }
                            // +1 for a→b, -1 for b→a under a canonical ordering of the pair.
                            let (pair, sign) = if (ka.x, ka.y, ka.z) < (kb.x, kb.y, kb.z) {
                                ((ka, kb), 1)
                            } else {
                                ((kb, ka), -1)
                            };
                            *directed.entry(pair).or_default() += sign;
                        }
                    }
                }
            }
        }
    }
    assert!(triangles > 500, "block has too little surface to test ({triangles} triangles)");
    let open: Vec<_> = directed.iter().filter(|(_, n)| **n != 0).collect();
    assert!(open.is_empty(), "{} unbalanced edges, e.g. {:?}", open.len(), open.first());
}

#[test]
fn dig_marks_affected_chunks_and_carves_a_cavity() {
    let mut world = World::new(SEED);
    let h = world.worldgen.height(5.0, 5.0);
    // Two metres below the surface, right on a chunk corner region if it happens to be one.
    let center = Vec3::new(5.0, h - 2.0, 5.0);
    assert!(world.density_at(center) < 0.0, "test point should start solid");
    let op = EditOp { center, radius: 3.0, kind: EditKind::Dig, material: Material::Air };
    let before: AHashMap<ChunkCoord, Vec<f32>> =
        neighbourhood(center).map(|c| (c, world.build_chunk(c).density)).collect();

    let dirty = world.edits.push(op);
    assert!(dirty.contains(&chunk_of(center)));
    assert!(world.density_at(center) > 0.0, "centre of a dig must be air");

    // Soundness: no chunk outside the dirty set may change.
    for (coord, density) in &before {
        let after = world.build_chunk(*coord).density;
        if !dirty.contains(coord) {
            assert_eq!(&after, density, "chunk {coord} changed but was not marked dirty");
        }
    }
    // And the chunk holding the centre has a sample that flipped to air.
    let coord = chunk_of(center);
    let after = world.build_chunk(coord);
    let changed = after.density.iter().zip(&before[&coord]).any(|(a, b)| *a >= 0.0 && *b < 0.0);
    assert!(changed);
}

#[test]
fn fill_turns_air_solid_with_the_given_material() {
    let mut world = World::new(SEED);
    let h = world.worldgen.height(-20.0, 30.0);
    let center = Vec3::new(-20.0, h + 6.0, 30.0);
    assert!(world.density_at(center) > 0.0);
    world.edits.push(EditOp { center, radius: 2.5, kind: EditKind::Fill, material: Material::Rock });
    assert!(world.density_at(center) < 0.0);
}

fn neighbourhood(p: Vec3) -> impl Iterator<Item = ChunkCoord> {
    let c = chunk_of(p);
    (-2..=2).flat_map(move |z| (-2..=2).flat_map(move |y| (-2..=2).map(move |x| c + IVec3::new(x, y, z))))
}

#[test]
fn raycast_hits_a_plane() {
    let hit = raycast(|p| p.y - 10.0, Vec3::new(3.0, 50.0, -2.0), Vec3::new(0.3, -1.0, 0.1), 200.0)
        .expect("ray should hit the plane");
    assert!((hit.point.y - 10.0).abs() < 0.01, "hit at {}", hit.point);
    assert!(hit.normal.dot(Vec3::Y) > 0.99);
    assert!(raycast(|p| p.y - 10.0, Vec3::new(0.0, 50.0, 0.0), Vec3::Y, 200.0).is_none());
}

#[test]
fn raycast_finds_the_terrain_surface() {
    let world = World::new(SEED);
    let h = world.worldgen.height(40.0, -40.0);
    let hit = raycast(|p| world.density_at(p), Vec3::new(40.0, h + 60.0, -40.0), -Vec3::Y, 200.0)
        .expect("ray should hit the ground");
    // Overhangs and caves can shift the surface a little from the height field.
    assert!((hit.point.y - h).abs() < 8.0, "hit {} vs height {h}", hit.point.y);
}

#[test]
fn bounding_box_replay_matches_full_replay() {
    let world = World::new(SEED);
    let h = world.worldgen.height(12.0, -7.0);
    let coord = chunk_of(Vec3::new(12.0, h, -7.0));
    let base = chunk_origin(coord);
    // A spread of digs and fills, some straddling the chunk border, one well outside it.
    let ops: Vec<EditOp> = (0..40)
        .map(|i| {
            let t = i as f32;
            let center = base
                + Vec3::new((t * 7.3) % 40.0 - 4.0, h - base.y + (t % 5.0) - 2.0, (t * 3.1) % 38.0 - 3.0);
            let kind = if i % 3 == 0 { EditKind::Fill } else { EditKind::Dig };
            EditOp { center, radius: 1.0 + (t % 4.0), kind, material: Material::Rock }
        })
        .chain([EditOp {
            center: base + Vec3::splat(200.0),
            radius: 3.0,
            kind: EditKind::Dig,
            material: Material::Air,
        }])
        .collect();
    let fast = ChunkData::build(&world.worldgen, coord, &ops);
    let mut slow = world.worldgen.generate(coord);
    for z in 0..SAMPLES {
        for y in 0..SAMPLES {
            for x in 0..SAMPLES {
                let p = ChunkData::sample_pos(coord, x, y, z);
                let i = ChunkData::index(x, y, z);
                for op in &ops {
                    (slow.density[i], slow.material[i]) = op.apply(p, slow.density[i], slow.material[i]);
                }
            }
        }
    }
    assert_eq!(fast.density, slow.density);
    assert_eq!(fast.material, slow.material);
}

#[test]
fn census_measures_a_filled_sphere() {
    let mut world = World::new(SEED);
    let center = Vec3::new(0.0, 600.0, 0.0);
    let empty = world.census(center, 2.0, 0.25);
    let expected = 4.0 / 3.0 * std::f32::consts::PI * 8.0;
    assert!(empty.total_solid() == 0.0);
    assert!((empty.air - expected).abs() / expected < 0.06, "air {} vs {expected}", empty.air);

    world.edits.push(EditOp { center, radius: 4.0, kind: EditKind::Fill, material: Material::Rock });
    let full = world.census(center, 2.0, 0.25);
    assert!(full.air == 0.0);
    let rock = full.solid[Material::Rock as usize];
    assert!((rock - expected).abs() / expected < 0.06, "rock {rock} vs {expected}");
    assert_eq!(rock, full.total_solid());
}

#[test]
fn material_names_round_trip() {
    for m in Material::ALL {
        assert_eq!(Material::from_name(m.name()), Some(m));
        assert_eq!(Material::from_u8(m as u8), m);
    }
}

#[test]
fn flatten_levels_the_footprint() {
    let mut world = World::new(SEED);
    // A spot on a hillside.
    let (x, z) = (37.0, -12.0);
    let h = world.worldgen.height(x, z);
    let target = (h * 2.0).round() / 2.0;
    for op in EditOp::flatten(glam::Vec2::new(x, z), 1.0, target, 6.0, 3.0, Material::Soil) {
        world.edits.push(op);
    }
    for (dx, dz) in [(-0.9, -0.9), (0.9, -0.9), (-0.9, 0.9), (0.9, 0.9), (0.0, 0.0)] {
        let p = Vec3::new(x + dx, target, z + dz);
        assert!(world.density_at(p + Vec3::Y * 0.3) > 0.0, "air just above the flattened top at {p}");
        assert!(world.density_at(p - Vec3::Y * 0.3) < 0.0, "ground just below the flattened top at {p}");
    }
}

#[test]
fn neighbouring_flattens_leave_no_membrane_between_them() {
    // Two adjacent cells flattened separately, in a hillside so the old ground above the target is
    // solid: the seam between them must be open air, not a density-0 sheet.
    let mut world = World::new(SEED);
    let (x, z) = (37.0, -12.0);
    let target = (world.worldgen.height(x, z) * 2.0).round() / 2.0 - 3.0;
    for cx in [x - 1.0, x + 1.0] {
        for op in EditOp::flatten(glam::Vec2::new(cx, z), 1.4, target, 6.0, 3.0, Material::Soil) {
            world.edits.push(op);
        }
    }
    // Anywhere across the seam, a player-sized sphere resting on the floor sees the floor 0.35 m
    // away, not a phantom wall: density must read the true distance to the floor.
    for dx in [-0.5, -0.2, 0.0, 0.2, 0.5] {
        let p = Vec3::new(x + dx, target + 0.35, z);
        assert!(world.density_at(p) > 0.34, "phantom wall at {p}: {}", world.density_at(p));
    }
}

#[test]
fn biomes_cover_the_world_in_sensible_shares() {
    use asc_world::Biome;
    let worldgen = WorldGen::new(SEED);
    let mut counts = std::collections::HashMap::new();
    let mut land = 0usize;
    let n = 200;
    for i in 0..n {
        for j in 0..n {
            let g = worldgen.ground(i as f32 * 503.0 - 50000.0, j as f32 * 503.0 - 50000.0);
            if g.underwater {
                continue;
            }
            land += 1;
            *counts.entry(g.biome).or_insert(0usize) += 1;
            assert!((0.0..=1.0).contains(&g.forest));
        }
    }
    let share = |b: Biome| *counts.get(&b).unwrap_or(&0) as f32 / land as f32;
    for b in Biome::ALL {
        println!("{}: {:.1}%", b.name(), 100.0 * share(b));
    }
    // Over 100 km every biome turns up on land, none takes over, and nothing is vanishingly rare.
    for b in Biome::ALL {
        assert!(share(b) > 0.01, "{} is too rare: {:.2}%", b.name(), 100.0 * share(b));
        assert!(share(b) < 0.4, "{} takes over: {:.1}%", b.name(), 100.0 * share(b));
    }
}

#[test]
fn chunk_seams_are_bit_exact_on_the_gpu() {
    // The renderer adds each vertex's chunk-local position to its chunk's offset from the (snapped)
    // camera. Two chunks sharing a border vertex must get exactly the same sum, or hairline cracks
    // open along the seam and the sky sparkles through.
    let worldgen = WorldGen::new(SEED);
    let a = surface_chunk(&worldgen, 10.0, 10.0);
    // For the vertical seam, a spot where the ground crosses a horizontal chunk boundary.
    let (vx, vz) = (0..10000)
        .map(|i| ((i % 100) as f32 * 7.0 + 3.0, (i / 100) as f32 * 7.0 + 3.0))
        .find(|&(x, z)| {
            let h = worldgen.height(x, z);
            let off = h - (h / 32.0).round() * 32.0;
            (0.5..2.0).contains(&off.abs()) && worldgen.ground(x, z).slope < 0.3
        })
        .expect("ground crossing a chunk boundary");
    let low = asc_world::chunk_of(Vec3::new(vx, worldgen.height(vx, vz) - 2.5, vz));
    let snap = |v: f32| (v * POSITION_GRID).round() / POSITION_GRID;
    let camera = Vec3::new(snap(1234.567), snap(-31.25), snap(-987.654));
    for (a, b) in [(a, a + IVec3::X), (a, a + IVec3::Z), (low, low + IVec3::Y)] {
        let gpu = |coord: ChunkCoord| -> Vec<Vec3> {
            let offset = chunk_origin(coord) - camera;
            mesh_chunk(&worldgen.generate(coord))
                .vertices
                .iter()
                .map(|v| Vec3::from(v.position) + offset)
                .collect()
        };
        let (va, vb) = (gpu(a), gpu(b));
        let mut exact = 0;
        for p in &va {
            if let Some(q) = vb.iter().find(|q| q.distance(*p) < 1e-3) {
                assert_eq!(p.to_array().map(f32::to_bits), q.to_array().map(f32::to_bits), "{p} vs {q}");
                exact += 1;
            }
        }
        assert!(exact > 10, "only {exact} shared vertices between {a:?} and {b:?}");
    }
}
