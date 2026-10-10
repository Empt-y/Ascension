//! Camera maths and frustum culling. All matrices are camera-relative: the camera sits at the origin and
//! world positions are offset by `-camera.position` before they reach the GPU.

use glam::{Mat4, Vec3, Vec4};

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    pub position: Vec3,
    /// Radians around +Y; 0 looks along +X.
    pub yaw: f32,
    /// Radians up from the horizon.
    pub pitch: f32,
    pub fov_y: f32,
    pub near: f32,
}

impl Default for Camera {
    fn default() -> Self {
        Self { position: Vec3::ZERO, yaw: 0.0, pitch: 0.0, fov_y: 70f32.to_radians(), near: 0.1 }
    }
}

impl Camera {
    pub fn forward(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp)
    }

    /// Horizontal right vector.
    pub fn right(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        Vec3::new(-sy, 0.0, cy)
    }

    pub fn view_rotation(&self) -> Mat4 {
        Mat4::look_to_rh(Vec3::ZERO, self.forward(), Vec3::Y)
    }

    /// Reversed-Z infinite perspective: depth 1 at the near plane, 0 at infinity.
    pub fn projection(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_infinite_reverse_rh(self.fov_y, aspect, self.near)
    }

    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.projection(aspect) * self.view_rotation()
    }
}

/// Side planes plus the near plane of a view-projection (the far plane is at infinity).
pub struct Frustum {
    planes: [Vec4; 5],
}

impl Frustum {
    pub fn from_view_proj(m: Mat4) -> Self {
        let (r0, r1, r2, r3) = (m.row(0), m.row(1), m.row(2), m.row(3));
        let planes = [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r3 - r2].map(|p| p / p.truncate().length());
        Self { planes }
    }

    /// Conservative box test in the same (camera-relative) space as the matrix.
    pub fn intersects_box(&self, min: Vec3, max: Vec3) -> bool {
        self.planes.iter().all(|p| {
            let n = p.truncate();
            let positive = Vec3::select(n.cmpge(Vec3::ZERO), max, min);
            n.dot(positive) + p.w >= 0.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frustum_keeps_what_is_ahead_and_drops_what_is_behind() {
        let cam = Camera { yaw: 0.0, ..Default::default() };
        let f = Frustum::from_view_proj(cam.view_proj(16.0 / 9.0));
        let ahead = Vec3::new(50.0, 0.0, 0.0);
        assert!(f.intersects_box(ahead - 1.0, ahead + 1.0));
        let behind = Vec3::new(-50.0, 0.0, 0.0);
        assert!(!f.intersects_box(behind - 1.0, behind + 1.0));
        let far_ahead = Vec3::new(10_000.0, 0.0, 0.0);
        assert!(f.intersects_box(far_ahead - 1.0, far_ahead + 1.0));
        // A box around the camera always intersects.
        assert!(f.intersects_box(Vec3::splat(-1.0), Vec3::splat(1.0)));
    }

    #[test]
    fn forward_and_right_are_orthonormal() {
        let cam = Camera { yaw: 0.7, pitch: 0.3, ..Default::default() };
        assert!((cam.forward().length() - 1.0).abs() < 1e-5);
        assert!(cam.forward().dot(cam.right()).abs() < 1e-5);
    }
}
