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
