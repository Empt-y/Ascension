//! Timed mining by material hardness, on the client: how long the button has been held on what.
//! When a bite (or a node) is done the client asks the server, which applies it (`asc_sim::rules`).

use asc_world::{Material, RayHit, World};
use glam::Vec3;

use crate::content::Content;
use crate::scatter::{NodeHit, NodeId};

pub use asc_sim::rules::BITE_RADIUS;

/// Moving the crosshair further than this from where mining started restarts the bite.
const RETARGET_DISTANCE: f32 = 0.5;
const PLACE_INTERVAL: f32 = 0.2;

#[derive(Default)]
pub struct Mining {
    /// Where mining started and the sample byte there (which tells ores apart).
    target: Option<(Vec3, u8)>,
    /// Or the ore node being mined, its hardness and ore.
    node: Option<(NodeId, f32, usize)>,
    progress: f32,
    place_cooldown: f32,
}

impl Mining {
    /// 0..1 progress of the current bite, for the HUD.
    pub fn progress(&self, content: &Content) -> f32 {
        if let Some((_, hardness, _)) = self.node {
            return (self.progress / hardness.max(1e-3)).min(1.0);
        }
        match self.target {
            Some((_, m)) => (self.progress / content.sample(m).hardness.max(1e-3)).min(1.0),
            None => 0.0,
        }
    }

    /// The sample byte being mined (see [`Content::sample_name`]).
    pub fn target_material(&self) -> Option<u8> {
        self.target.map(|(_, m)| m)
    }

    /// What's being mined, for the HUD.
    pub fn target_name(&self) -> Option<&'static str> {
        match self.node {
            Some((_, _, ore)) => Some(asc_world::ores::builtin().ores[ore].name.as_str()),
            None => self.target_material().map(Content::sample_name),
        }
    }

    pub fn reset(&mut self) {
        self.target = None;
        self.node = None;
        self.progress = 0.0;
    }

    pub fn tick_cooldown(&mut self, dt: f32) {
        self.place_cooldown = (self.place_cooldown - dt).max(0.0);
    }

    /// Advance mining an ore node while the button is held; true when it breaks.
    pub fn mine_node(&mut self, dt: f32, node: NodeHit, hardness: f32) -> bool {
        if self.node.map(|(id, _, _)| id) != Some(node.id) {
            self.reset();
            self.node = Some((node.id, hardness, node.ore));
        }
        self.progress += dt;
        if self.progress < hardness {
            return false;
        }
        self.reset();
        true
    }

    /// Advance mining while the button is held. Returns the hit to bite when a bite completes.
    pub fn mine(&mut self, dt: f32, hit: Option<RayHit>, world: &World, content: &Content) -> Option<RayHit> {
        let Some(hit) = hit else {
            self.reset();
            return None;
        };
        if self.node.is_some() {
            self.reset();
        }
        // What's just under the surface at the crosshair.
        let (_, material) = world.sample_raw(hit.point - hit.normal * 0.3);
        if material == Material::Air as u8 {
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
        if self.progress < content.sample(material).hardness {
            return None;
        }
        self.progress = 0.0;
        Some(hit)
    }

    /// Whether placing may happen now (it repeats at a fixed rate while held); starts the cooldown.
    pub fn try_place(&mut self) -> bool {
        if self.place_cooldown > 0.0 {
            return false;
        }
        self.place_cooldown = PLACE_INTERVAL;
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use asc_world::raycast;

    #[test]
    fn a_bite_takes_the_material_hardness() {
        let content = Content::load().unwrap();
        let world = World::new(1337);
        let h = world.worldgen.height(40.0, 40.0);
        let hit = raycast(|p| world.density_at(p), Vec3::new(40.0, h + 3.0, 40.0), -Vec3::Y, 20.0).unwrap();
        let hardness = content.sample(world.sample_raw(hit.point - hit.normal * 0.3).1).hardness;
        let mut mining = Mining::default();
        let dt = 1.0 / 60.0;
        let mut elapsed = 0.0;
        loop {
            elapsed += dt;
            if mining.mine(dt, Some(hit), &world, &content).is_some() {
                break;
            }
            assert!(elapsed < 10.0, "never finished a bite");
        }
        assert!((elapsed - hardness).abs() < 2.0 * dt, "took {elapsed}s for hardness {hardness}");
    }

    #[test]
    fn placing_repeats_at_a_fixed_rate() {
        let mut mining = Mining::default();
        assert!(mining.try_place());
        assert!(!mining.try_place());
        mining.tick_cooldown(PLACE_INTERVAL);
        assert!(mining.try_place());
    }
}
