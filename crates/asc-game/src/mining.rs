//! Timed mining by material hardness, and placing material back from the inventory.

use asc_world::{Capsule, Census, EditKind, EditOp, Material, RayHit, World};
use glam::Vec3;

use crate::content::Content;
use crate::inventory::Inventory;

/// Radius of one bite (and of one placed lump).
pub const BITE_RADIUS: f32 = 0.9;
/// How far into the surface a bite is centred (and out of it for placing), as a fraction of the radius.
const INSET: f32 = 0.3;
/// Moving the crosshair further than this from where mining started restarts the bite.
const RETARGET_DISTANCE: f32 = 0.5;
const PLACE_INTERVAL: f32 = 0.2;
const CENSUS_STEP: f32 = 0.25;

#[derive(Default)]
pub struct Mining {
    target: Option<(Vec3, Material)>,
    progress: f32,
    place_cooldown: f32,
}

/// What one completed bite produced.
#[derive(Debug)]
pub struct Bite {
    pub op: EditOp,
    pub removed: Census,
    /// Whole items that didn't fit in the inventory.
    pub overflow: u32,
}

#[derive(Debug, PartialEq)]
pub enum PlaceOutcome {
    Placed(EditOp),
    NothingSelected,
    NotPlaceable,
    NotEnough,
    WouldBury,
    OnCooldown,
}

impl Mining {
    /// 0..1 progress of the current bite, for the HUD.
    pub fn progress(&self, content: &Content) -> f32 {
        match self.target {
            Some((_, m)) => (self.progress / content.material(m).hardness.max(1e-3)).min(1.0),
            None => 0.0,
        }
    }

    pub fn target_material(&self) -> Option<Material> {
        self.target.map(|(_, m)| m)
    }

    pub fn reset(&mut self) {
        self.target = None;
        self.progress = 0.0;
    }

    pub fn tick_cooldown(&mut self, dt: f32) {
        self.place_cooldown = (self.place_cooldown - dt).max(0.0);
    }

    /// Advance mining while the button is held. Returns a bite when one completes; the edit has
    /// already been added to the world and its yield to the inventory.
    pub fn mine(
        &mut self,
        dt: f32,
        hit: Option<RayHit>,
        world: &mut World,
        content: &Content,
        inventory: &mut Inventory,
    ) -> Option<Bite> {
        let Some(hit) = hit else {
            self.reset();
            return None;
        };
        // What's just under the surface at the crosshair.
        let (_, material) = world.sample_at(hit.point - hit.normal * 0.3);
        if material == Material::Air {
            self.reset();
            return None;
        }
        let same =
            self.target.is_some_and(|(p, m)| m == material && p.distance(hit.point) < RETARGET_DISTANCE);
        if !same {
            self.target = Some((hit.point, material));
            self.progress = 0.0;
        }
        self.progress += dt;
        if self.progress < content.material(material).hardness {
            return None;
        }
        self.progress = 0.0;

        let center = hit.point - hit.normal * BITE_RADIUS * INSET;
        let removed = world.census(center, BITE_RADIUS, CENSUS_STEP);
        let op = EditOp { center, radius: BITE_RADIUS, kind: EditKind::Dig, material: Material::Air };
        world.edits.push(op);
        let mut overflow = 0;
        for m in Material::ALL {
            let volume = removed.solid[m as usize];
            if volume > 0.0
                && let Some(item) = content.material(m).drop
            {
                overflow += inventory.add_volume(content, item, volume);
            }
        }
        Some(Bite { op, removed, overflow })
    }

    /// Place a lump of the selected item's material at the crosshair, paying for the air it fills.
    pub fn place(
        &mut self,
        hit: RayHit,
        player_feet: Vec3,
        world: &mut World,
        content: &Content,
        inventory: &mut Inventory,
    ) -> PlaceOutcome {
        if self.place_cooldown > 0.0 {
            return PlaceOutcome::OnCooldown;
        }
        let Some(stack) = inventory.selected_stack() else { return PlaceOutcome::NothingSelected };
        let Some(material) = content.item(stack.item).places else { return PlaceOutcome::NotPlaceable };
        let center = hit.point + hit.normal * BITE_RADIUS * INSET;
        if Capsule::PLAYER.overlaps_sphere(player_feet, center, BITE_RADIUS) {
            return PlaceOutcome::WouldBury;
        }
        let filled = world.census(center, BITE_RADIUS, CENSUS_STEP).air;
        if !inventory.pay_volume(stack.item, filled) {
            return PlaceOutcome::NotEnough;
        }
        let op = EditOp { center, radius: BITE_RADIUS, kind: EditKind::Fill, material };
        world.edits.push(op);
        self.place_cooldown = PLACE_INTERVAL;
        PlaceOutcome::Placed(op)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asc_world::raycast;

    fn ground_hit(world: &World) -> RayHit {
        let h = world.worldgen.height(40.0, 40.0);
        raycast(|p| world.density_at(p), Vec3::new(40.0, h + 3.0, 40.0), -Vec3::Y, 20.0)
            .expect("ground below")
    }

    #[test]
    fn bites_take_the_material_hardness_and_yield_items() {
        let content = Content::load().unwrap();
        let mut world = World::new(1337);
        let mut inv = Inventory::default();
        let mut mining = Mining::default();
        let hit = ground_hit(&world);
        let material = mining.target_material();
        assert_eq!(material, None);
        let hardness = {
            let (_, m) = world.sample_at(hit.point - hit.normal * 0.3);
            content.material(m).hardness
        };
        let dt = 1.0 / 60.0;
        let mut elapsed = 0.0;
        let bite = loop {
            elapsed += dt;
            if let Some(b) = mining.mine(dt, Some(hit), &mut world, &content, &mut inv) {
                break b;
            }
            assert!(elapsed < 10.0, "never finished a bite");
        };
        assert!((elapsed - hardness).abs() < 2.0 * dt, "took {elapsed}s for hardness {hardness}");
        assert_eq!(world.edits.len(), 1);
        let gained: f32 = inv.slots.iter().flatten().map(|s| s.count as f32).sum::<f32>()
            + inv.remainders.values().sum::<f32>();
        assert!(
            (gained - bite.removed.total_solid()).abs() < 1e-3,
            "yield {gained} vs removed {:?}",
            bite.removed
        );
        assert!(bite.removed.total_solid() > 0.5);
    }

    #[test]
    fn placing_costs_the_volume_it_fills() {
        let content = Content::load().unwrap();
        let mut world = World::new(1337);
        let mut inv = Inventory::default();
        let rock = content.item_id("rock").unwrap();
        inv.add(&content, rock, 10);
        let mut mining = Mining::default();
        let hit = ground_hit(&world);
        let far_away = hit.point + Vec3::new(20.0, 0.0, 0.0);
        let outcome = mining.place(hit, far_away, &mut world, &content, &mut inv);
        let PlaceOutcome::Placed(op) = outcome else { panic!("expected to place, got {outcome:?}") };
        let paid = 10.0 - (inv.count(rock) as f32 + inv.remainders[&rock]);
        let expected = 0.5 * 4.0 / 3.0 * std::f32::consts::PI * BITE_RADIUS.powi(3);
        assert!(
            paid > expected * 0.4 && paid < expected * 1.6,
            "paid {paid}, roughly half a sphere is {expected}"
        );
        assert!(world.density_at(op.center + hit.normal * 0.3) < 0.0, "placed material is solid");
        assert_eq!(mining.place(hit, far_away, &mut world, &content, &mut inv), PlaceOutcome::OnCooldown);
    }

    #[test]
    fn placing_refuses_to_bury_the_player_or_overspend() {
        let content = Content::load().unwrap();
        let mut world = World::new(1337);
        let mut inv = Inventory::default();
        let mut mining = Mining::default();
        let hit = ground_hit(&world);
        assert_eq!(
            mining.place(hit, hit.point, &mut world, &content, &mut inv),
            PlaceOutcome::NothingSelected
        );
        inv.add(&content, content.item_id("rock").unwrap(), 1);
        assert_eq!(mining.place(hit, hit.point, &mut world, &content, &mut inv), PlaceOutcome::WouldBury);
        let far = hit.point + Vec3::X * 20.0;
        assert_eq!(mining.place(hit, far, &mut world, &content, &mut inv), PlaceOutcome::NotEnough);
        assert!(world.edits.is_empty());
    }
}
