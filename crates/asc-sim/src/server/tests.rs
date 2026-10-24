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
    s.player.inventory = Inventory::default();
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
        ServerMsg::Vitals(crate::vitals::Vitals::default()),
        ServerMsg::Hurt { amount: 120, cause: crate::vitals::Cause::Fall },
        ServerMsg::Died { cause: crate::vitals::Cause::Lava },
        ServerMsg::StashPlaced { id: 2, at: Vec3::ONE },
        ServerMsg::StashTaken { id: 2 },
        ServerMsg::Fluid {
            coord: IVec3::new(1, 0, -1),
            data: asc_world::fluid::PackedFluid::pack(&asc_world::fluid::FluidChunk::empty()),
        },
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

/// Walk the player down a column of air from `from` to the ground, a step at a time (as the client
/// would report a fall).
fn fall(s: &mut Server, from: Vec3) {
    let ground = rules::find_ground(&s.world, from.x, from.z).y;
    s.player.feet = from;
    let mut y = from.y;
    while y > ground {
        y = (y - 1.0).max(ground);
        s.player.budget = Budget::new();
        s.handle(ClientMsg::Move { feet: Vec3::new(from.x, y, from.z), yaw: 0.0, pitch: 0.0, mode: Mode::Walk });
    }
}

#[test]
fn falls_hurt_by_their_height_and_kill_when_long() {
    let mut s = server("fall");
    let ground = rules::find_ground(&s.world, 30.0, 30.0);
    s.player.feet = ground;
    fall(&mut s, ground + Vec3::Y * 3.0);
    assert_eq!(s.player.vitals.health, crate::vitals::MAX_HEALTH, "a short drop is free");
    fall(&mut s, ground + Vec3::Y * 12.0);
    // (Measured from the first airborne position reported, a metre below the start here.)
    assert_eq!(s.player.vitals.health, crate::vitals::MAX_HEALTH - Vitals::fall_damage(11.0));
    assert!(s.drain().any(|m| matches!(m, ServerMsg::Hurt { cause: Cause::Fall, .. })));
}

#[test]
fn dying_leaves_a_stash_to_come_back_for() {
    let mut s = server("death");
    let rock = s.content.item_id("rock").unwrap();
    s.player.inventory.add(&s.content, rock, 20);
    let ground = rules::find_ground(&s.world, 30.0, 30.0);
    s.player.feet = ground;
    fall(&mut s, ground + Vec3::Y * 40.0);
    let out: Vec<ServerMsg> = s.drain().collect();
    assert!(out.iter().any(|m| matches!(m, ServerMsg::Died { cause: Cause::Fall })));
    assert!(out.iter().any(|m| matches!(m, ServerMsg::Correct { .. })));
    assert_eq!(s.player.feet, s.spawn);
    assert_eq!(s.player.inventory.count(rock), 0);
    assert_eq!(s.player.vitals.health, crate::vitals::MAX_HEALTH);
    assert_eq!(s.stashes.len(), 1);
    // Back at the stash: the things come back.
    s.player.feet = ground + Vec3::X * 0.5;
    s.player.budget = Budget::new();
    s.handle(ClientMsg::Move { feet: ground, yaw: 0.0, pitch: 0.0, mode: Mode::Walk });
    assert_eq!(s.player.inventory.count(rock), 20);
    assert!(s.stashes.is_empty());
    assert!(s.drain().any(|m| matches!(m, ServerMsg::StashTaken { .. })));
}

/// Hold the tool `kind` (select its hotbar slot).
fn hold(s: &mut Server, kind: ToolKind) {
    let item = s.content.tool_item(kind).unwrap();
    let slot = s.player.inventory.slots.iter().position(|x| x.is_some_and(|x| x.item == item)).expect("the tool in the inventory");
    assert!(slot < crate::inventory::HOTBAR, "{kind:?} isn't on the hotbar");
    s.handle(ClientMsg::Inventory { seq: 1, op: crate::inventory::InventoryOp::Select(slot) });
    assert_eq!(s.held_tool(), Some(kind));
}

#[test]
fn a_new_player_gets_the_starter_kit_once() {
    let mut s = server("kit");
    for kind in ToolKind::ALL {
        let item = s.content.tool_item(kind).unwrap();
        assert!(s.player.inventory.count(item) > 0, "no {kind:?}");
    }
    let dir = s.save_dir.clone();
    let multitool = s.content.tool_item(ToolKind::Multitool).unwrap();
    s.player.inventory.take(multitool, 1);
    s.save("saved");
    let config = Config { save_dir: dir, seed: Some(1337), new_world: false, validate: true, cheats: false, stress: None };
    let again = Server::open(config, lattice()).unwrap();
    assert_eq!(again.player.inventory.count(multitool), 0, "the kit came again");
}

#[test]
fn the_laser_mines_from_further_off_but_only_held() {
    let mut s = server("laser");
    let hit = ground(&s, 40.0, 40.0);
    stand(&mut s, hit.point + Vec3::new(15.0, 0.0, 0.0));
    hold(&mut s, ToolKind::Scanner);
    s.handle(ClientMsg::Tool(ToolUse::Laser { point: hit.point, normal: hit.normal }));
    assert!(s.world.edits.is_empty(), "the scanner doesn't mine");
    hold(&mut s, ToolKind::Multitool);
    s.handle(ClientMsg::Tool(ToolUse::Laser { point: hit.point, normal: hit.normal }));
    assert_eq!(s.world.edits.len(), 1, "the laser reaches 15 m");
    stand(&mut s, hit.point + Vec3::new(30.0, 0.0, 0.0));
    s.player.last_dig = None;
    s.handle(ClientMsg::Tool(ToolUse::Laser { point: hit.point, normal: hit.normal }));
    assert_eq!(s.world.edits.len(), 1, "not 30 m");
}

#[test]
fn holding_the_trigger_overheats_the_multitool() {
    let mut s = server("heat");
    hold(&mut s, ToolKind::Multitool);
    for _ in 0..(tools::HEAT_SECONDS * 20.0) as u32 + 10 {
        s.handle(ClientMsg::Tool(ToolUse::Hold));
        s.tick();
    }
    assert!(s.player.tools.locked(ToolKind::Multitool));
    let hit = ground(&s, 40.0, 40.0);
    stand(&mut s, hit.point + Vec3::new(5.0, 0.0, 0.0));
    s.handle(ClientMsg::Tool(ToolUse::Laser { point: hit.point, normal: hit.normal }));
    assert!(s.world.edits.is_empty(), "locked: no mining");
    assert!(s.drain().any(|m| matches!(m, ServerMsg::ToolState(t) if t.locked(ToolKind::Multitool))));
}

/// A land animal (standing the player by the nearest herd and letting them come).
fn a_creature(s: &mut Server) -> crate::creatures::Creature {
    let home = (Vec2::new(s.player.feet.x, s.player.feet.z) / crate::creatures::CELL).floor().as_ivec2();
    let at = (0..400)
        .find_map(|i| {
            let cell = home + glam::IVec2::new(i % 20 - 10, i / 20 - 10);
            crate::creatures::herds_in(&s.world, 1337, cell).into_iter().find(|h| !h.2[0].kind().swims()).map(|h| h.2[0].at)
        })
        .expect("a herd on land somewhere near");
    stand(s, at + Vec3::new(40.0, 0.0, 0.0));
    for _ in 0..60 {
        s.tick();
    }
    *s.creatures.all().find(|c| !c.kind().swims()).expect("animals near spawn")
}

#[test]
fn the_rifle_and_flamer_hurt_what_they_hit() {
    let mut s = server("rifle");
    let c = a_creature(&mut s);
    hold(&mut s, ToolKind::Multitool);
    // From above it, looking down: nothing between.
    let body = c.body();
    stand(&mut s, body + Vec3::Y * 5.0 - Vec3::Y * EYE_HEIGHT);
    s.handle(ClientMsg::Tool(ToolUse::Shoot { dir: -Vec3::Y }));
    let after = s.creatures.get(c.id).unwrap().health;
    assert_eq!(after, c.health - tools::RIFLE_DAMAGE);
    // Shooting off to the side misses.
    s.tick();
    s.tick();
    s.tick();
    s.handle(ClientMsg::Tool(ToolUse::Shoot { dir: Vec3::X }));
    assert_eq!(s.creatures.get(c.id).unwrap().health, after);
    assert!(s.player.tools.charge < tools::RIFLE_CHARGE - 1.5);
    // The flame, from closer.
    stand(&mut s, body + Vec3::Y * 3.0 - Vec3::Y * EYE_HEIGHT);
    s.tick();
    s.tick();
    s.handle(ClientMsg::Tool(ToolUse::Flame { dir: -Vec3::Y }));
    assert_eq!(s.creatures.get(c.id).unwrap().health, after - tools::FLAME_DAMAGE);
}

#[test]
fn beacons_are_placed_taken_and_saved() {
    let mut s = server("beacons");
    hold(&mut s, ToolKind::Beacon);
    let at = s.player.feet + Vec3::new(2.0, 0.0, 0.0);
    let beacon = s.content.tool_item(ToolKind::Beacon).unwrap();
    let before = s.player.inventory.count(beacon);
    s.handle(ClientMsg::Tool(ToolUse::PlaceBeacon { at }));
    s.handle(ClientMsg::Tool(ToolUse::PlaceBeacon { at: at + Vec3::X * 40.0 }));
    assert_eq!(s.beacons.len(), 1, "not 40 m off");
    assert_eq!(s.player.inventory.count(beacon), before - 1);
    let dir = s.save_dir.clone();
    s.save("saved");
    let config = Config { save_dir: dir, seed: Some(1337), new_world: false, validate: true, cheats: false, stress: None };
    let mut s = Server::open(config, lattice()).unwrap();
    assert_eq!(s.beacons, vec![(0, at)]);
    s.handle(ClientMsg::Tool(ToolUse::TakeBeacon { id: 0 }));
    assert!(s.beacons.is_empty());
    assert_eq!(s.player.inventory.count(beacon), before);
}

#[test]
fn the_injector_heals_for_meat() {
    let mut s = server("inject");
    hold(&mut s, ToolKind::Injector);
    s.player.vitals.hurt(500, Cause::Fall);
    s.handle(ClientMsg::Tool(ToolUse::Inject));
    assert_eq!(s.player.vitals.healing, 0, "no meat, no healing");
    let meat = s.content.item_id(tools::INJECT_COST).unwrap();
    s.player.inventory.add(&s.content, meat, 2);
    s.handle(ClientMsg::Tool(ToolUse::Inject));
    assert_eq!(s.player.inventory.count(meat), 1);
    for _ in 0..(tools::INJECT_SECONDS * 20.0) as u32 + 5 {
        s.tick();
    }
    assert!(s.player.vitals.health >= 500 + tools::INJECT_HEAL, "health {}", s.player.vitals.health);
}

#[test]
fn the_manipulator_digs_a_big_ball_and_fills_it_back() {
    let mut s = server("terraform");
    let hit = ground(&s, 40.0, 40.0);
    stand(&mut s, hit.point + Vec3::new(8.0, 0.0, 0.0));
    hold(&mut s, ToolKind::Manipulator);
    s.handle(ClientMsg::Tool(ToolUse::Terraform { point: hit.point, normal: hit.normal, radius: 2.0, fill: false }));
    assert_eq!(s.world.edits.ops()[0].radius, 2.0);
    let got: u32 = s.player.inventory.slots.iter().flatten().filter(|x| s.content.item(x.item).places.is_some()).map(|x| x.count).sum();
    assert!(got > 4, "a 2 m ball gave {got} items");
    for _ in 0..4 {
        s.tick();
    }
    s.handle(ClientMsg::Tool(ToolUse::Terraform { point: hit.point - Vec3::Y, normal: hit.normal, radius: 1.0, fill: true }));
    assert_eq!(s.world.edits.len(), 2);
}
