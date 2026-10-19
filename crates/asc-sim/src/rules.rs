//! What a player's actions do to the world: mining bites, filling with carried material, levelling
//! ground. The server applies these; the client only decides when to ask for them.

use asc_world::{Capsule, Census, EditKind, EditOp, Material, World};
use glam::{Vec2, Vec3};

use crate::content::Content;
use crate::inventory::Inventory;

/// Radius of one bite (and of one placed lump).
pub const BITE_RADIUS: f32 = 0.9;
/// How far into the surface a bite is centred (and out of it for placing), as a fraction of the radius.
const INSET: f32 = 0.3;
const CENSUS_STEP: f32 = 0.25;
/// How far a flatten reaches past its cell (metres). Combining cut boxes with `max` gives the right
/// surface but, inside the combined volume, only a lower bound on the distance to it: within this
/// far of an internal box face it under-reports, which collision feels as an invisible wall. So
/// neighbouring flattens overlap by more than the player's radius (0.35 m).
pub const FLATTEN_OVERLAP: f32 = 0.4;
/// Ore items from breaking a surface outcrop and an underground node.
pub const SURFACE_NODE_YIELD: u32 = 3;
pub const DEEP_NODE_YIELD: u32 = 8;
/// Node mining time as a multiple of its ore's bite time.
pub const SURFACE_NODE_HARDNESS: f32 = 1.5;
pub const DEEP_NODE_HARDNESS: f32 = 2.5;

/// What one completed bite produced.
#[derive(Debug)]
pub struct Bite {
    pub op: EditOp,
    pub removed: Census,
    /// Whole items that didn't fit in the inventory.
    pub overflow: u32,
}

/// Take a bite out of the surface at `point` (facing `normal`): the edit goes into the world and
/// its yield into the inventory.
pub fn bite(world: &mut World, content: &Content, inventory: &mut Inventory, point: Vec3, normal: Vec3) -> Bite {
    let center = point - normal * BITE_RADIUS * INSET;
    let removed = world.census(center, BITE_RADIUS, CENSUS_STEP);
    let op = EditOp { center, radius: BITE_RADIUS, kind: EditKind::Dig, material: Material::Air };
    world.edits.push(op);
    let mut overflow = 0;
    for m in Material::ALL.into_iter().filter(|m| *m != Material::Ore) {
        let volume = removed.solid[m as usize];
        if volume > 0.0
            && let Some(item) = content.material(m).drop
        {
            overflow += inventory.add_volume(content, item, volume);
        }
    }
    for (ore, &volume) in removed.ores.iter().enumerate() {
        if volume > 0.0
            && let Some(item) = content.ores[ore].drop
        {
            overflow += inventory.add_volume(content, item, volume);
        }
    }
    Bite { op, removed, overflow }
}

#[derive(Debug, PartialEq)]
pub enum FillOutcome {
    Placed(EditOp),
    NothingSelected,
    NotPlaceable,
    NotEnough,
    WouldBury,
}

/// Place a lump of the selected item's material at `point`, paying for the air it fills.
pub fn fill(
    world: &mut World,
    content: &Content,
    inventory: &mut Inventory,
    point: Vec3,
    normal: Vec3,
    player_feet: Vec3,
) -> FillOutcome {
    let Some(stack) = inventory.selected_stack() else { return FillOutcome::NothingSelected };
    let Some(material) = content.item(stack.item).places else { return FillOutcome::NotPlaceable };
    let center = point + normal * BITE_RADIUS * INSET;
    if Capsule::PLAYER.overlaps_sphere(player_feet, center, BITE_RADIUS) {
        return FillOutcome::WouldBury;
    }
    let filled = world.census(center, BITE_RADIUS, CENSUS_STEP).air;
    if !inventory.pay_volume(stack.item, filled) {
        return FillOutcome::NotEnough;
    }
    let op = EditOp { center, radius: BITE_RADIUS, kind: EditKind::Fill, material };
    world.edits.push(op);
    FillOutcome::Placed(op)
}

/// The edits that level the building cell centred at `center` (half-width `half_cell`) to `height`.
pub fn flatten_ops(center: Vec2, half_cell: f32, height: f32) -> [EditOp; 2] {
    // Deep fill so the levelled pad meets the ground on downhill slopes instead of overhanging.
    EditOp::flatten(center, half_cell + FLATTEN_OVERLAP, height, 6.0, 8.0, Material::Soil)
}

/// Feet position standing on the ground in column (x, z).
pub fn find_ground(world: &World, x: f32, z: f32) -> Vec3 {
    let top = Vec3::new(x, 400.0, z);
    match asc_world::raycast(|p| world.density_at(p), top, -Vec3::Y, 800.0) {
        Some(hit) => hit.point + Vec3::Y * 0.05,
        None => Vec3::new(x, world.worldgen.height(x, z) + 2.0, z),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asc_world::raycast;

    fn ground(world: &World) -> (Vec3, Vec3) {
        let h = world.worldgen.height(40.0, 40.0);
        let hit = raycast(|p| world.density_at(p), Vec3::new(40.0, h + 3.0, 40.0), -Vec3::Y, 20.0)
            .expect("ground below");
        (hit.point, hit.normal)
    }

    #[test]
    fn a_bite_yields_what_it_removed() {
        let content = Content::load().unwrap();
        let mut world = World::new(1337);
        let mut inv = Inventory::default();
        let (point, normal) = ground(&world);
        let bite = bite(&mut world, &content, &mut inv, point, normal);
        assert_eq!(world.edits.len(), 1);
        let gained: f32 = inv.slots.iter().flatten().map(|s| s.count as f32).sum::<f32>()
            + inv.remainders.values().sum::<f32>();
        assert!((gained - bite.removed.total_solid()).abs() < 1e-3, "yield {gained} vs {:?}", bite.removed);
        assert!(bite.removed.total_solid() > 0.5);
    }

    #[test]
    fn filling_costs_the_volume_it_fills_and_refuses_to_bury() {
        let content = Content::load().unwrap();
        let mut world = World::new(1337);
        let mut inv = Inventory::default();
        let (point, normal) = ground(&world);
        let far_away = point + Vec3::new(20.0, 0.0, 0.0);
        assert_eq!(fill(&mut world, &content, &mut inv, point, normal, far_away), FillOutcome::NothingSelected);
        let rock = content.item_id("rock").unwrap();
        inv.add(&content, rock, 1);
        assert_eq!(fill(&mut world, &content, &mut inv, point, normal, point), FillOutcome::WouldBury);
        assert_eq!(fill(&mut world, &content, &mut inv, point, normal, far_away), FillOutcome::NotEnough);
        inv.add(&content, rock, 9);
        let FillOutcome::Placed(op) = fill(&mut world, &content, &mut inv, point, normal, far_away) else {
            panic!("expected to place")
        };
        let paid = 10.0 - (inv.count(rock) as f32 + inv.remainders[&rock]);
        let expected = 0.5 * 4.0 / 3.0 * std::f32::consts::PI * BITE_RADIUS.powi(3);
        assert!(paid > expected * 0.4 && paid < expected * 1.6, "paid {paid}, about half a sphere is {expected}");
        assert!(world.density_at(op.center + normal * 0.3) < 0.0, "placed material is solid");
    }
}
