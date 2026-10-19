use super::*;
use asc_build::{BuildingSpec, PartKind, Piece, Target};

fn lattice() -> Lattice {
    let spec = BuildingSpec::parse(include_str!("../../../../assets/data/building.toml")).unwrap();
    Lattice::new(&spec)
}

fn temp_dir(name: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("ascension-server-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

fn server(name: &str) -> Server {
    let config =
        Config { save_dir: temp_dir(name), seed: Some(1337), new_world: false, validate: true, cheats: false, stress: None };
    Server::open(config, lattice()).unwrap()
}

/// Put the player at `feet` (a test setup step, not a move: moves are checked).
fn stand(s: &mut Server, feet: Vec3) {
    s.player.feet = feet;
}

fn ground(s: &Server, x: f32, z: f32) -> asc_world::RayHit {
    let h = s.world.worldgen.height(x, z);
    asc_world::raycast(|p| s.world.density_at(p), Vec3::new(x, h + 3.0, z), -Vec3::Y, 20.0).unwrap()
}

#[test]
fn a_bite_edits_the_world_and_fills_the_inventory_within_reach_only() {
    let mut s = server("bite");
    let hit = ground(&s, 40.0, 40.0);
    stand(&mut s, hit.point + Vec3::new(30.0, 0.0, 0.0));
    s.handle(ClientMsg::Bite { point: hit.point, normal: hit.normal });
    assert!(s.drain().next().is_none(), "out of reach: nothing happens");
    assert!(s.world.edits.is_empty());

    stand(&mut s, hit.point + Vec3::new(2.0, 0.0, 0.0));
    s.handle(ClientMsg::Bite { point: hit.point, normal: hit.normal });
    let out: Vec<ServerMsg> = s.drain().collect();
    assert!(matches!(out[0], ServerMsg::Edited(_)), "{out:?}");
    let ServerMsg::Inventory(inv) = &out[1] else { panic!("{out:?}") };
    let items: u32 = inv.slots.iter().flatten().map(|s| s.count).sum();
    let fraction: f32 = inv.remainders.iter().map(|r| r.1).sum();
    assert!(items as f32 + fraction > 0.5);
    assert_eq!(s.world.edits.len(), 1);
}

#[test]
fn pieces_are_checked_placed_and_dismantled() {
    let mut s = server("pieces");
    let hit = ground(&s, 12.0, 8.0);
    stand(&mut s, hit.point + Vec3::new(3.0, 0.0, 0.0));
    let l = s.structures.lattice.clone();
    let cell = l.cell_of(hit.point.x, hit.point.z);
    let center = (cell.as_vec2() + 0.5) * l.cell;
    let height = (hit.point.y / l.unit).round() * l.unit;
    s.handle(ClientMsg::Flatten { center, height });
    assert_eq!(s.drain().filter(|m| matches!(m, ServerMsg::Edited(_))).count(), 2);
    let world = &s.world;
    let terrain = |x: f32, z: f32| {
        asc_world::raycast(|p| world.density_at(p), Vec3::new(x, height + 10.0, z), -Vec3::Y, 40.0)
            .map_or(height, |h| h.point.y)
    };
    let target = Vec3::new(center.x, height, center.y);
    let snap = s.structures.snap(PartKind::Foundation, Target::Terrain(target), Vec3::ZERO, &terrain);
    let piece = Piece { key: snap.key, tier: 1, grounded: snap.grounded };
    s.handle(ClientMsg::Place { piece, gap: snap.gap });
    assert_eq!(s.drain().collect::<Vec<_>>(), vec![ServerMsg::PiecePlaced(piece)]);
    // The same again is refused, with a notice.
    s.handle(ClientMsg::Place { piece, gap: snap.gap });
    assert!(matches!(s.drain().next(), Some(ServerMsg::Notice(_))));
    s.handle(ClientMsg::Dismantle { key: piece.key });
    assert_eq!(s.drain().collect::<Vec<_>>(), vec![ServerMsg::PieceRemoved(piece.key)]);
    assert!(s.structures.is_empty());
}

#[test]
fn an_ore_node_breaks_once() {
    let mut s = server("node");
    let seed = ore_nodes::seed_of(s.seed());
    let node = (0..400)
        .find_map(|i| {
            let column = glam::IVec2::new(i % 20 - 10, i / 20 - 10);
            ore_nodes::surface(seed, &s.world.worldgen, column).into_iter().next()
        })
        .expect("an outcrop near spawn");
    let id = ore_nodes::node_id(node.position);
    s.handle(ClientMsg::MineNode { id: id + IVec3::X * 7 });
    assert!(s.drain().next().is_none(), "no node there");
    stand(&mut s, node.position + Vec3::new(1.5, 0.0, 0.0));
    s.handle(ClientMsg::MineNode { id });
    let out: Vec<ServerMsg> = s.drain().collect();
    assert_eq!(out[0], ServerMsg::NodeMined(id));
    let ServerMsg::Inventory(inv) = &out[1] else { panic!("{out:?}") };
    let item = s.content.ores[node.ore].drop.unwrap();
    let got: u32 = inv.slots.iter().flatten().filter(|st| st.item == item).map(|st| st.count).sum();
    assert_eq!(got, rules::SURFACE_NODE_YIELD);
    s.handle(ClientMsg::MineNode { id });
    assert!(s.drain().next().is_none(), "already mined");
}

#[test]
fn inventory_ops_echo_their_sequence_number() {
    let mut s = server("inventory");
    let rock = s.content.item_id("rock").unwrap();
    s.player.inventory.add(&s.content, rock, 10);
    s.handle(ClientMsg::Inventory { seq: 1, op: crate::inventory::InventoryOp::Click(0) });
    let Some(ServerMsg::Inventory(inv)) = s.drain().next() else { panic!() };
    assert_eq!(inv.seq, 1);
    assert_eq!(inv.held.map(|h| h.count), Some(10));
    assert!(inv.slots[0].is_none());
}

#[test]
fn a_saved_world_reopens_the_same() {
    let dir = temp_dir("reopen");
    let config =
        |dir: &PathBuf| Config { save_dir: dir.clone(), seed: Some(77), new_world: false, validate: false, cheats: true, stress: None };
    let mut s = Server::open(config(&dir), lattice()).unwrap();
    let hit = ground(&s, 5.0, 5.0);
    s.handle(ClientMsg::Bite { point: hit.point, normal: hit.normal });
    s.mined.insert([1, 2, 3]);
    for _ in 0..5 {
        s.tick();
    }
    s.handle(ClientMsg::Save);
    let before = s.welcome();
    let again = Server::open(config(&dir), lattice()).unwrap();
    assert_eq!(again.welcome(), before);
    assert_eq!(again.seed(), 77);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn every_message_survives_postcard() {
    use crate::inventory::InventoryOp;
    let s = server("postcard");
    let op = asc_world::EditOp {
        center: Vec3::new(1.0, 2.0, 3.0),
        radius: 0.9,
        kind: asc_world::EditKind::Dig,
        material: asc_world::Material::Air,
    };
    let key = asc_build::PieceKey::foundation(glam::IVec2::new(3, -2), 4);
    let piece = Piece { key, tier: 2, grounded: true };
    let client = [
        ClientMsg::Move { feet: Vec3::ONE, yaw: 0.5, pitch: -0.2, mode: Mode::Fly },
        ClientMsg::Bite { point: Vec3::X, normal: Vec3::Y },
        ClientMsg::Fill { point: Vec3::Z, normal: Vec3::Y, fresh: true },
        ClientMsg::MineNode { id: IVec3::new(4, -8, 12) },
        ClientMsg::Place { piece, gap: 0.25 },
        ClientMsg::Dismantle { key },
        ClientMsg::Flatten { center: glam::Vec2::new(1.0, 3.0), height: 7.5 },
        ClientMsg::Edit(op),
        ClientMsg::Inventory { seq: 9, op: InventoryOp::RightClick(4) },
        ClientMsg::SetTime(0.5),
        ClientMsg::Save,
        ClientMsg::Ping(3),
    ];
    let _ = ServerMsg::Correct { feet: Vec3::ONE, mode: Mode::Walk };
    for m in client {
        let bytes = postcard::to_stdvec(&m).unwrap();
        assert_eq!(postcard::from_bytes::<ClientMsg>(&bytes).unwrap(), m);
    }
    let server_msgs = [
        ServerMsg::Welcome(Box::new(s.welcome())),
        ServerMsg::Edited(op),
        ServerMsg::PiecePlaced(piece),
        ServerMsg::PieceRemoved(key),
        ServerMsg::NodeMined(IVec3::ONE),
        ServerMsg::Inventory(s.player.inventory.snapshot(None, 3)),
        ServerMsg::Time { ticks: 9, time_of_day: 0.3 },
        ServerMsg::Notice("hi".into()),
        ServerMsg::Saved { ok: true },
        ServerMsg::Pong(3),
        ServerMsg::Stats { tps: 19.8, mspt: 0.4 },
        ServerMsg::Correct { feet: Vec3::new(1.0, 2.0, 3.0), mode: Mode::Walk },
    ];
    for m in server_msgs {
        let bytes = postcard::to_stdvec(&m).unwrap();
        assert_eq!(postcard::from_bytes::<ServerMsg>(&bytes).unwrap(), m);
    }
}

#[test]
fn bites_come_no_faster_than_the_material_allows() {
    let mut s = server("rate");
    let hit = ground(&s, 40.0, 40.0);
    s.player.feet = hit.point + Vec3::new(2.0, 0.0, 0.0);
    s.handle(ClientMsg::Bite { point: hit.point, normal: hit.normal });
    s.handle(ClientMsg::Bite { point: hit.point - hit.normal * 0.5, normal: hit.normal });
    assert_eq!(s.world.edits.len(), 1, "the second bite came at once and was refused");
}

#[test]
fn development_tools_and_flying_need_permission() {
    let mut s = server("cheats");
    let op = asc_world::EditOp {
        center: s.player.feet,
        radius: 3.0,
        kind: asc_world::EditKind::Dig,
        material: asc_world::Material::Air,
    };
    let feet = s.player.feet;
    s.handle(ClientMsg::Move { feet, yaw: 0.0, pitch: 0.0, mode: Mode::Fly });
    assert_eq!(s.player.mode, Mode::Walk);
    s.handle(ClientMsg::Edit(op));
    s.handle(ClientMsg::SetTime(0.9));
    assert!(s.world.edits.is_empty());
    assert!((s.sim.time_of_day - 0.9).abs() > 0.01);
    s.player.cheats = true;
    s.handle(ClientMsg::Edit(op));
    assert_eq!(s.world.edits.len(), 1);
}

#[test]
fn teleporting_and_walking_into_the_ground_are_refused() {
    let mut s = server("moves");
    let start = s.player.feet;
    let away = start + Vec3::new(40.0, 0.0, 0.0);
    let away = Vec3::new(away.x, s.world.worldgen.height(away.x, away.z) + 0.05, away.z);
    s.handle(ClientMsg::Move { feet: away, yaw: 0.0, pitch: 0.0, mode: Mode::Walk });
    assert_eq!(s.player.feet, start);
    assert!(s.drain().any(|m| matches!(m, ServerMsg::Correct { feet, .. } if feet == start)));
    let buried = start - Vec3::Y * 3.0;
    s.handle(ClientMsg::Move { feet: buried, yaw: 0.0, pitch: 0.0, mode: Mode::Walk });
    assert_eq!(s.player.feet, start, "inside the ground");
    // A small step is fine.
    let step = start + Vec3::new(0.5, 0.0, 0.0);
    let step = Vec3::new(step.x, s.world.worldgen.height(step.x, step.z) + 0.05, step.z);
    s.handle(ClientMsg::Move { feet: step, yaw: 0.0, pitch: 0.0, mode: Mode::Walk });
    assert_eq!(s.player.feet, step);
}
