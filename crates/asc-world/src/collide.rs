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

/// Signed distance from `p` to an axis-aligned box.
pub fn box_distance(p: Vec3, lo: Vec3, hi: Vec3) -> f32 {
    let center = (lo + hi) * 0.5;
    let half = (hi - lo) * 0.5;
    let q = (p - center).abs() - half;
    q.max(Vec3::ZERO).length() + q.max_element().min(0.0)
}

/// Density gradient by central differences.
///
/// Six samples along the axes rather than four on a tetrahedron: the diagonal tetrahedral offsets
/// straddle sharp creases (a floor meeting a wall) and invent a sideways component there, which made
/// players slide along walls they were pushing into. Returns the gradient scaled by `2 * eps`.
pub fn gradient(density: &impl Fn(Vec3) -> f32, p: Vec3, eps: f32) -> Vec3 {
    Vec3::new(
        density(p + Vec3::X * eps) - density(p - Vec3::X * eps),
        density(p + Vec3::Y * eps) - density(p - Vec3::Y * eps),
        density(p + Vec3::Z * eps) - density(p - Vec3::Z * eps),
    )
}

/// Estimated true distance to the surface: density divided by its gradient's length.
pub fn distance_estimate(density: &impl Fn(Vec3) -> f32, p: Vec3) -> f32 {
    // Small, so the samples only straddle a crease (floor meeting wall) within 2 cm of it.
    const EPS: f32 = 0.02;
    let len = (gradient(density, p, EPS) / (2.0 * EPS)).length();
    density(p) / len.max(1.0)
}

/// If a sphere overlaps the ground, how to push it out.
///
/// Worldgen density is only roughly a distance: near caves its gradient can be 1.5x steeper, and deep
/// inside the ground it is clamped flat. Dividing by the gradient's length turns it into a proper
/// distance estimate; a flat (clamped) field means "deep inside", reported with `depth > radius`.
pub fn push_sphere(density: &impl Fn(Vec3) -> f32, center: Vec3, radius: f32) -> Option<Contact> {
    // Small, so the samples only straddle a crease (floor meeting wall) within 2 cm of it.
    const EPS: f32 = 0.02;
    let d = density(center);
    // Cheap early out, assuming the field is never less than half as steep as a true distance.
    if d >= radius * 2.0 {
        return None;
    }
    let g = gradient(density, center, EPS) / (2.0 * EPS);
    let len = g.length();
    if len < 0.1 {
        // Clamped region: far inside solid ground (or far out in air, which the early out caught).
        return (d < 0.0).then_some(Contact { normal: Vec3::Y, depth: radius + 1.0 });
    }
    // Only ever shrink the distance (worldgen fields can be steeper than a true distance). Where
    // the gradient is shallower than 1, at the crease where two surfaces meet, density is already
    // a safe distance; dividing would push it out and let the capsule sink into the corner.
    let distance = d / len.max(1.0);
    (distance < radius).then(|| Contact { normal: g / len, depth: radius - distance })
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
