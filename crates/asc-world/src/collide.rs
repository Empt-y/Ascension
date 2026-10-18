//! Collision of simple shapes against any density field.
//!
//! Density is (roughly) a signed distance, so a sphere of radius `r` whose centre has density `d < r`
//! overlaps the ground by `r - d`, and the density gradient points the way out. A capsule is three
//! stacked spheres, which is plenty for walking on smooth terrain.

use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Contact {
    /// Unit vector pointing out of the ground.
    pub normal: Vec3,
    /// How far the sphere overlapped the ground before it was pushed out.
    pub depth: f32,
}

/// An upright capsule made of spheres, positioned by its feet (the bottom of the lowest sphere).
#[derive(Clone, Copy, Debug)]
pub struct Capsule {
    pub radius: f32,
    /// Heights of the sphere centres above the feet.
    pub spheres: [f32; 3],
}

impl Capsule {
    pub const PLAYER: Self = Self { radius: 0.35, spheres: [0.35, 0.9, 1.45] };

    pub fn height(&self) -> f32 {
        self.spheres[2] + self.radius
    }

    /// Whether a sphere at `center` with `radius` overlaps the capsule standing at `feet`.
    pub fn overlaps_sphere(&self, feet: Vec3, center: Vec3, radius: f32) -> bool {
        self.spheres.iter().any(|h| (feet + Vec3::Y * *h).distance(center) < self.radius + radius)
    }
}

/// Density gradient from four samples on a tetrahedron (cheaper than six-sample central differences).
pub fn gradient(density: &impl Fn(Vec3) -> f32, p: Vec3, eps: f32) -> Vec3 {
    const K: [Vec3; 4] = [
        Vec3::new(1.0, -1.0, -1.0),
        Vec3::new(-1.0, -1.0, 1.0),
        Vec3::new(-1.0, 1.0, -1.0),
        Vec3::new(1.0, 1.0, 1.0),
    ];
    K.iter().map(|k| *k * density(p + *k * eps)).sum()
}

/// If a sphere overlaps the ground, how to push it out.
pub fn push_sphere(density: &impl Fn(Vec3) -> f32, center: Vec3, radius: f32) -> Option<Contact> {
    let d = density(center);
    if d >= radius {
        return None;
    }
    let normal = gradient(density, center, 0.1).normalize_or(Vec3::Y);
    Some(Contact { normal, depth: radius - d })
}

/// Push a capsule out of the ground. Returns the corrected feet position and the contacts found
/// (one per sphere at most: the deepest).
pub fn resolve_capsule(
    density: &impl Fn(Vec3) -> f32,
    mut feet: Vec3,
    capsule: &Capsule,
) -> (Vec3, Vec<Contact>) {
    const ITERATIONS: usize = 4;
    // A hair of extra push so the next step doesn't start exactly on the surface.
    const SKIN: f32 = 0.002;
    let mut deepest: [Option<Contact>; 3] = [None; 3];
    for _ in 0..ITERATIONS {
        let mut moved = false;
        for (i, h) in capsule.spheres.iter().enumerate() {
            if let Some(c) = push_sphere(density, feet + Vec3::Y * *h, capsule.radius) {
                feet += c.normal * (c.depth + SKIN);
                moved = true;
                if deepest[i].is_none_or(|d| c.depth > d.depth) {
                    deepest[i] = Some(c);
                }
            }
        }
        if !moved {
            break;
        }
    }
    (feet, deepest.into_iter().flatten().collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn plane(p: Vec3) -> f32 {
        p.y
    }

    #[test]
    fn capsule_sunk_into_a_plane_is_lifted_onto_it() {
        let (feet, contacts) = resolve_capsule(&plane, Vec3::new(3.0, -0.2, 1.0), &Capsule::PLAYER);
        assert!((feet.y - 0.0).abs() < 0.01, "feet at {feet}");
        assert!((feet.x - 3.0).abs() < 1e-4 && (feet.z - 1.0).abs() < 1e-4, "no sideways push: {feet}");
        assert!(contacts.iter().all(|c| c.normal.y > 0.99));
    }

    #[test]
    fn capsule_clear_of_the_ground_is_untouched() {
        let start = Vec3::new(0.0, 0.5, 0.0);
        let (feet, contacts) = resolve_capsule(&plane, start, &Capsule::PLAYER);
        assert_eq!(feet, start);
        assert!(contacts.is_empty());
    }

    #[test]
    fn wall_pushes_sideways() {
        // Solid for x > 1.
        let wall = |p: Vec3| (1.0 - p.x).min(p.y + 10.0);
        let (feet, contacts) = resolve_capsule(&wall, Vec3::new(0.8, 0.0, 0.0), &Capsule::PLAYER);
        assert!(feet.x <= 1.0 - Capsule::PLAYER.radius + 0.01, "feet at {feet}");
        assert!(contacts.iter().all(|c| c.normal.x < -0.99));
    }

    #[test]
    fn overlap_test_matches_sphere_distance() {
        let feet = Vec3::ZERO;
        assert!(Capsule::PLAYER.overlaps_sphere(feet, Vec3::new(0.0, 1.0, 1.0), 0.9));
        assert!(!Capsule::PLAYER.overlaps_sphere(feet, Vec3::new(0.0, 1.0, 1.5), 0.9));
    }
}
