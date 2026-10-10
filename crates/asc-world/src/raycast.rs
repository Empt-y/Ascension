//! Ray queries against any density function.

use glam::Vec3;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub point: Vec3,
    /// Outward surface normal (towards air).
    pub normal: Vec3,
    pub distance: f32,
}

const MIN_STEP: f32 = 0.25;

/// March from `origin` along `dir` until the density turns negative, then bisect to the surface.
///
/// Density is roughly a distance, so steps grow with it (half of it, to stay safe where noise makes the
/// field steeper than a true distance), but never below a quarter metre. Returns `None` if the ray
/// starts inside solid ground or reaches `max_distance` without a hit.
pub fn raycast(density: impl Fn(Vec3) -> f32, origin: Vec3, dir: Vec3, max_distance: f32) -> Option<RayHit> {
    let dir = dir.normalize_or_zero();
    if dir == Vec3::ZERO || density(origin) < 0.0 {
        return None;
    }
    let mut t_prev = 0.0;
    let mut t = 0.0;
    while t < max_distance {
        let d = density(origin + dir * t);
        if d < 0.0 {
            let (mut lo, mut hi) = (t_prev, t);
            for _ in 0..10 {
                let mid = 0.5 * (lo + hi);
                if density(origin + dir * mid) < 0.0 {
                    hi = mid;
                } else {
                    lo = mid;
                }
            }
            let point = origin + dir * hi;
            return Some(RayHit { point, normal: normal_at(&density, point), distance: hi });
        }
        t_prev = t;
        t += (d * 0.5).max(MIN_STEP);
    }
    None
}

fn normal_at(density: &impl Fn(Vec3) -> f32, p: Vec3) -> Vec3 {
    const E: f32 = 0.25;
    let g = Vec3::new(
        density(p + Vec3::X * E) - density(p - Vec3::X * E),
        density(p + Vec3::Y * E) - density(p - Vec3::Y * E),
        density(p + Vec3::Z * E) - density(p - Vec3::Z * E),
    );
    g.normalize_or(Vec3::Y)
}
