//! First-person player: a kinematic capsule walking on the density field, or flying through it.
//!
//! Physics runs at a fixed 60 Hz, separate from both the frame rate and the 20 Hz game tick, and the
//! camera is interpolated between the last two physics states so motion stays smooth at any fps.

use asc_world::{Capsule, push_sphere};
use glam::Vec3;

pub const STEP: f32 = 1.0 / 60.0;
pub use asc_sim::EYE_HEIGHT;
const WALK_SPEED: f32 = 5.0;
const SPRINT_SPEED: f32 = 8.0;
const GROUND_ACCEL: f32 = 60.0;
const AIR_CONTROL: f32 = 0.3;
const GRAVITY: f32 = 25.0;
const JUMP_SPEED: f32 = 7.5;
const TERMINAL_SPEED: f32 = 60.0;
/// Contacts whose normal points at least this far up count as ground you can stand on.
const WALKABLE: f32 = 0.7;
const MAX_SUBSTEP: f32 = 0.25;
const SNAP_DISTANCE: f32 = 0.3;
const RESOLVE_ITERATIONS: usize = 4;
/// Most one contact may push the capsule per iteration, so a bad gradient can never fling it.
const MAX_PUSH: f32 = 0.3;
/// Deeper than this and the player counts as buried and is lifted out instead of pushed.
const BURIED_DEPTH: f32 = 0.6;
/// How fast a buried player (e.g. after filling on themselves) is lifted out.
const UNBURY_SPEED: f32 = 6.0;
/// Ledges up to this high (a foundation's rim, a stair) are climbed automatically when walking.
const STEP_HEIGHT: f32 = 0.55;
/// Below this the player is put back at the spawn point.
pub const KILL_Y: f32 = -400.0;

pub use asc_sim::Mode;

/// Intent for one update, already mapped from keys.
#[derive(Clone, Copy, Debug, Default)]
pub struct MoveInput {
    pub forward: f32,
    pub right: f32,
    /// Fly mode: up/down. Walk mode: `up > 0` means jump.
    pub up: f32,
    pub sprint: bool,
    pub fly_speed: f32,
}

#[derive(Clone, Debug)]
pub struct Player {
    pub feet: Vec3,
    pub velocity: Vec3,
    pub yaw: f32,
    pub pitch: f32,
    pub mode: Mode,
    pub grounded: bool,
    prev_feet: Vec3,
    accumulator: f32,
}

impl Player {
    pub fn new(feet: Vec3) -> Self {
        Self {
            feet,
            velocity: Vec3::ZERO,
            yaw: 0.0,
            pitch: 0.0,
            mode: Mode::Walk,
            grounded: false,
            prev_feet: feet,
            accumulator: 0.0,
        }
    }

    /// Move to a new spot without interpolating from the old one.
    pub fn teleport(&mut self, feet: Vec3) {
        self.feet = feet;
        self.prev_feet = feet;
        self.velocity = Vec3::ZERO;
    }

    pub fn look_dir(&self) -> Vec3 {
        let (sy, cy) = self.yaw.sin_cos();
        let (sp, cp) = self.pitch.sin_cos();
        Vec3::new(cy * cp, sp, sy * cp)
    }

    /// Interpolated eye position for rendering.
    pub fn eye(&self) -> Vec3 {
        let alpha = self.accumulator / STEP;
        self.prev_feet.lerp(self.feet, alpha) + Vec3::Y * EYE_HEIGHT
    }

    /// Advance by a frame's worth of time. Returns how many physics steps ran.
    pub fn update(&mut self, dt: f32, input: &MoveInput, density: &impl Fn(Vec3) -> f32) -> u32 {
        self.accumulator += dt.min(0.1);
        let mut steps = 0;
        while self.accumulator >= STEP {
            self.prev_feet = self.feet;
            self.step(input, density);
            self.accumulator -= STEP;
            steps += 1;
        }
        steps
    }

    pub fn step(&mut self, input: &MoveInput, density: &impl Fn(Vec3) -> f32) {
        match self.mode {
            Mode::Fly => self.fly(input),
            Mode::Walk => self.walk(input, density),
        }
    }

    fn fly(&mut self, input: &MoveInput) {
        let (sy, cy) = self.yaw.sin_cos();
        let right = Vec3::new(-sy, 0.0, cy);
        let wish = self.look_dir() * input.forward + right * input.right + Vec3::Y * input.up;
        let boost = if input.sprint { 5.0 } else { 1.0 };
        self.velocity = wish.normalize_or_zero() * input.fly_speed * boost;
        self.feet += self.velocity * STEP;
        self.grounded = false;
    }

    fn walk(&mut self, input: &MoveInput, density: &impl Fn(Vec3) -> f32) {
        let (sy, cy) = self.yaw.sin_cos();
        let forward = Vec3::new(cy, 0.0, sy);
        let right = Vec3::new(-sy, 0.0, cy);
        let wish = (forward * input.forward + right * input.right).normalize_or_zero();
        let speed = if input.sprint { SPRINT_SPEED } else { WALK_SPEED };

        // Accelerate the horizontal velocity toward the wished velocity; on the ground this also acts
        // as friction when there's no input.
        let horizontal = Vec3::new(self.velocity.x, 0.0, self.velocity.z);
        let target = wish * speed;
        let accel = GROUND_ACCEL * if self.grounded { 1.0 } else { AIR_CONTROL };
        let change = target - horizontal;
        let max_change = accel * STEP;
        let horizontal = horizontal + change.clamp_length_max(max_change);
        self.velocity.x = horizontal.x;
        self.velocity.z = horizontal.z;

        let jumped = self.grounded && input.up > 0.0;
        if jumped {
            self.velocity.y = JUMP_SPEED;
        }
        self.velocity.y = (self.velocity.y - GRAVITY * STEP).max(-TERMINAL_SPEED);

        if self.buried(density) {
            // Rise straight up through the ground until the capsule is clear again.
            self.feet.y += UNBURY_SPEED * STEP;
            self.velocity = Vec3::ZERO;
            self.grounded = false;
            return;
        }

        let was_grounded = self.grounded;
        let motion = self.velocity * STEP;
        let substeps = (motion.length() / MAX_SUBSTEP).ceil().max(1.0) as u32;
        let mut grounded = false;
        for _ in 0..substeps {
            let before = self.feet;
            let step = motion / substeps as f32;
            self.feet += step;
            let r = self.resolve(density);
            if was_grounded && !jumped && self.try_step_up(before, step, density) {
                grounded = true;
                continue;
            }
            if r.squeezed {
                // Floor and ceiling at once: the gap is too low. Undo the horizontal part of the move,
                // as if a wall stopped it, and settle again.
                self.feet = Vec3::new(before.x, self.feet.y, before.z);
                self.velocity.x = 0.0;
                self.velocity.z = 0.0;
                grounded |= r.grounded | self.resolve(density).grounded;
            } else {
                grounded |= r.grounded;
            }
        }

        // Stay glued to the ground walking downhill instead of hopping off every bump.
        if was_grounded && !grounded && !jumped && self.velocity.y <= 0.0 {
            let saved = (self.feet, self.velocity);
            self.feet -= Vec3::Y * SNAP_DISTANCE;
            if self.resolve(density).grounded {
                grounded = true;
            } else {
                (self.feet, self.velocity) = saved;
            }
        }
        self.grounded = grounded;
        if grounded && self.velocity.y < 0.0 {
            self.velocity.y = 0.0;
        }
    }

    /// If a low ledge stopped this substep's horizontal move, climb it: from `STEP_HEIGHT` up, probe
    /// a capsule radius ahead, and if there's floor to land on, move onto it. Returns whether it
    /// stepped. (Moving only this substep's few centimetres wouldn't get the capsule over the ledge.)
    fn try_step_up(&mut self, before: Vec3, step: Vec3, density: &impl Fn(Vec3) -> f32) -> bool {
        let wanted = Vec3::new(step.x, 0.0, step.z);
        let moved = Vec3::new(self.feet.x - before.x, 0.0, self.feet.z - before.z);
        if wanted.length_squared() < 1e-8 || moved.length() > wanted.length() * 0.5 {
            return false;
        }
        let saved = (self.feet, self.velocity);
        let velocity = self.velocity;
        let fail = |this: &mut Self| {
            (this.feet, this.velocity) = saved;
            false
        };
        // Lift, and make sure there's headroom up there.
        self.feet = before + Vec3::Y * STEP_HEIGHT;
        if self.resolve(density).squeezed || self.feet.y < before.y + STEP_HEIGHT - 0.01 {
            return fail(self);
        }
        let reach = Capsule::PLAYER.radius + 0.05;
        let lifted = self.feet;
        self.feet += wanted.normalize() * reach;
        self.resolve(density);
        let advanced = Vec3::new(self.feet.x - lifted.x, 0.0, self.feet.z - lifted.z).length();
        if advanced < reach * 0.6 {
            return fail(self);
        }
        // Settle back down onto the ledge in small steps (one big drop would pass through its top
        // and end up inside it).
        let mut landed = false;
        let mut dropped = 0.0;
        while dropped < STEP_HEIGHT + 0.05 && !landed {
            self.feet.y -= 0.1;
            dropped += 0.1;
            landed = self.resolve(density).grounded;
        }
        if !landed || self.feet.y <= before.y + 0.02 {
            return fail(self);
        }
        self.velocity = Vec3::new(velocity.x, 0.0, velocity.z);
        true
    }

    /// Whether any part of the capsule is deeper in the ground than pushing should handle. Normal
    /// movement never gets this deep (substeps are 0.25 m); it happens after filling terrain onto the
    /// player, or spawning inside it.
    fn buried(&self, density: &impl Fn(Vec3) -> f32) -> bool {
        let capsule = Capsule::PLAYER;
        capsule.spheres.iter().any(|h| {
            push_sphere(density, self.feet + Vec3::Y * *h, capsule.radius)
                .is_some_and(|c| c.depth > BURIED_DEPTH)
        })
    }

    /// Push the capsule out of the ground and cancel velocity into it.
    fn resolve(&mut self, density: &impl Fn(Vec3) -> f32) -> Resolved {
        let capsule = Capsule::PLAYER;
        let mut out = Resolved::default();
        let mut ceiling = false;
        let mut support = Vec3::ZERO;
        for _ in 0..RESOLVE_ITERATIONS {
            let mut moved = false;
            for h in capsule.spheres {
                let Some(contact) = push_sphere(density, self.feet + Vec3::Y * h, capsule.radius) else {
                    continue;
                };
                let n = contact.normal;
                let depth = contact.depth.min(MAX_PUSH);
                if n.y > WALKABLE {
                    // Lift straight up rather than along the slope, so standing still on a hill
                    // doesn't slowly slide you down it.
                    self.feet.y += depth / n.y + 0.001;
                    out.grounded = true;
                } else {
                    self.feet += n * (depth + 0.001);
                    ceiling |= n.y < -0.5;
                }
                let into = self.velocity.dot(n);
                if into < 0.0 {
                    self.velocity -= n * into;
                }
                support += n;
                moved = true;
            }
            if !moved {
                break;
            }
        }
        // Wedged between two steep walls (a narrow ravine): no single contact is walkable, but
        // together they hold you up, so you can stand, jump and walk out.
        if !out.grounded && support.normalize_or_zero().y > WALKABLE {
            out.grounded = true;
        }
        out.squeezed = out.grounded && ceiling;
        out
    }
}

#[derive(Clone, Copy, Debug, Default)]
struct Resolved {
    /// Touching ground that can be stood on.
    grounded: bool,
    /// Touching floor and ceiling at once.
    squeezed: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn run(player: &mut Player, input: MoveInput, seconds: f32, density: impl Fn(Vec3) -> f32) {
        for _ in 0..(seconds / STEP) as usize {
            player.step(&input, &density);
        }
    }

    fn plane(p: Vec3) -> f32 {
        p.y
    }

    #[test]
    fn falls_and_stands_on_flat_ground() {
        let mut p = Player::new(Vec3::new(0.0, 5.0, 0.0));
        run(&mut p, MoveInput::default(), 2.0, plane);
        assert!(p.grounded);
        assert!(p.feet.y.abs() < 0.02, "feet at {}", p.feet);
        assert!(p.velocity.length() < 1e-3);
    }

    #[test]
    fn walks_forward_at_walking_speed() {
        let mut p = Player::new(Vec3::ZERO);
        run(&mut p, MoveInput::default(), 0.5, plane);
        let start = p.feet;
        run(&mut p, MoveInput { forward: 1.0, ..Default::default() }, 2.0, plane);
        let moved = p.feet - start;
        assert!((moved.x - 9.6).abs() < 0.6, "moved {moved}"); // ~2 s at 5 m/s, minus spin-up
        assert!(p.feet.y.abs() < 0.02 && p.grounded);
    }

    #[test]
    fn stands_still_on_a_walkable_slope() {
        // 25 degree hill rising along +x.
        let slope = |p: Vec3| (p.y - p.x * 25f32.to_radians().tan()) * 25f32.to_radians().cos();
        let mut p = Player::new(Vec3::new(0.0, 1.0, 0.0));
        run(&mut p, MoveInput::default(), 3.0, slope);
        assert!(p.grounded);
        assert!(p.feet.x.abs() < 0.05, "slid to {}", p.feet);
    }

    #[test]
    fn slides_off_a_cliff_face() {
        // 70 degrees: too steep to stand on.
        let a = 70f32.to_radians();
        let slope = |p: Vec3| (p.y - p.x * a.tan()) * a.cos();
        let mut p = Player::new(Vec3::new(0.0, 0.5, 0.0));
        run(&mut p, MoveInput::default(), 1.0, slope);
        assert!(p.feet.x < -0.3 && p.feet.y < 0.0, "should slide down and away: {}", p.feet);
    }

    #[test]
    fn walls_stop_you() {
        // Floor, plus a wall for x > 3.
        let room = |p: Vec3| p.y.min(3.0 - p.x);
        let mut p = Player::new(Vec3::ZERO);
        run(&mut p, MoveInput { forward: 1.0, sprint: true, ..Default::default() }, 3.0, room);
        assert!(p.feet.x <= 3.0 - Capsule::PLAYER.radius + 0.02, "went through the wall: {}", p.feet);
        assert!(p.feet.x > 2.4, "should reach the wall: {}", p.feet);
    }

    #[test]
    fn fast_falls_do_not_tunnel_through_a_thin_floor() {
        // A floor half a metre thick, with air below it.
        let slab = |p: Vec3| p.y.abs() - 0.25;
        let mut p = Player::new(Vec3::new(0.0, 300.0, 0.0));
        p.velocity.y = -TERMINAL_SPEED;
        run(&mut p, MoveInput::default(), 8.0, slab);
        assert!(p.feet.y > 0.0 && p.grounded, "fell through: {}", p.feet);
    }

    #[test]
    fn walks_up_a_low_ledge_but_not_a_high_one() {
        // Ground rises 0.5 m at x = 2 (like a foundation's rim) and another 1 m at x = 6, built
        // from exact box distance fields (a sloppy field misreports which face is nearest inside
        // the risers).
        let block = |p: Vec3, x0: f32, top: f32| {
            asc_world::box_distance(p, Vec3::new(x0, -10.0, -50.0), Vec3::new(1000.0, top, 50.0))
        };
        let steps = move |p: Vec3| p.y.min(block(p, 2.0, 0.5)).min(block(p, 6.0, 1.5));
        let mut p = Player::new(Vec3::ZERO);
        run(&mut p, MoveInput { forward: 1.0, ..Default::default() }, 3.0, steps);
        assert!(p.feet.x > 2.5 && p.feet.x < 6.0, "should climb 0.5 m but stop at 1 m: {}", p.feet);
        assert!((p.feet.y - 0.5).abs() < 0.03 && p.grounded, "{}", p.feet);
    }

    #[test]
    fn a_buried_player_rises_out() {
        let mut p = Player::new(Vec3::new(0.0, -5.0, 0.0));
        run(&mut p, MoveInput::default(), 3.0, plane);
        assert!(p.feet.y.abs() < 0.05 && p.grounded, "feet at {}", p.feet);
    }

    #[test]
    fn a_gap_too_low_to_fit_blocks_you() {
        // Floor at y = 0 everywhere; for x > 2 a ceiling at y = 1.2 (lower than the 1.8 m capsule).
        let tunnel = |p: Vec3| if p.x > 2.0 { p.y.min(1.2 - p.y) } else { p.y };
        let mut p = Player::new(Vec3::ZERO);
        run(&mut p, MoveInput { forward: 1.0, sprint: true, ..Default::default() }, 3.0, tunnel);
        assert!(p.feet.x < 2.1, "squeezed into the gap: {}", p.feet);
        assert!(p.feet.y.abs() < 0.05, "should stay on the floor: {}", p.feet);
    }

    #[test]
    fn wedged_in_a_narrow_v_counts_as_standing() {
        // Two 60 degree walls meeting at x = 0: too narrow for the capsule to reach the bottom.
        let a = 60f32.to_radians();
        let v = |p: Vec3| (p.y - p.x.abs() * a.tan()) * a.cos();
        let mut p = Player::new(Vec3::new(0.0, 2.0, 0.0));
        run(&mut p, MoveInput::default(), 2.0, v);
        assert!(p.grounded, "wedged player should be standing: {}", p.feet);
        let y = p.feet.y;
        run(&mut p, MoveInput { up: 1.0, ..Default::default() }, 0.1, v);
        assert!(p.feet.y > y + 0.3, "should be able to jump out: {} -> {}", y, p.feet.y);
    }

    #[test]
    fn jump_leaves_the_ground_and_lands() {
        let mut p = Player::new(Vec3::ZERO);
        run(&mut p, MoveInput::default(), 0.2, plane);
        let mut peak = 0.0f32;
        let jump = MoveInput { up: 1.0, ..Default::default() };
        p.step(&jump, &plane);
        for _ in 0..90 {
            p.step(&MoveInput::default(), &plane);
            peak = peak.max(p.feet.y);
        }
        // v²/2g = 7.5² / 50 = 1.125 m.
        assert!((peak - 1.125).abs() < 0.1, "peak {peak}");
        assert!(p.grounded && p.feet.y.abs() < 0.02);
    }

    #[test]
    fn interpolated_eye_lies_between_physics_states() {
        let mut p = Player::new(Vec3::ZERO);
        p.mode = Mode::Fly;
        let input = MoveInput { forward: 1.0, fly_speed: 6.0, ..Default::default() };
        p.update(STEP * 1.5, &input, &plane);
        let eye = p.eye() - Vec3::Y * EYE_HEIGHT;
        assert!(eye.x > p.prev_feet.x && eye.x < p.feet.x, "{eye} between {} and {}", p.prev_feet, p.feet);
    }
}

#[cfg(test)]
mod corner_tests {
    use super::*;

    #[test]
    fn pushing_into_a_wall_from_a_floor_does_not_slide_or_sink() {
        // Floor at y = 0 and a wall face at z = 2 (solid beyond), meeting in a sharp inside corner,
        // like a foundation and a wall.
        let room = |p: Vec3| p.y.min(2.0 - p.z);
        let mut p = Player::new(Vec3::new(5.0, 0.0, 0.0));
        p.yaw = std::f32::consts::FRAC_PI_2;
        for _ in 0..(3.0 / STEP) as usize {
            p.step(&MoveInput { forward: 1.0, ..Default::default() }, &room);
        }
        assert!((p.feet.x - 5.0).abs() < 0.01, "slid sideways to {}", p.feet);
        assert!(p.feet.y.abs() < 0.02, "sank or rose: {}", p.feet);
        assert!(p.feet.z <= 2.0 - Capsule::PLAYER.radius + 0.02 && p.grounded, "{}", p.feet);
    }
}
