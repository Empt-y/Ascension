//! Using the tool in hand: input becomes `ToolUse`s for the server (which checks them and does the
//! work), and the effects are drawn here: beams, bolts, flame, sparks, the scan's sweep, the torch's
//! glow, beacons' light pillars. Heat and charge come from the server; the rifle's charge is
//! spent here too as it fires, so a held button doesn't outrun it.

use std::collections::BTreeMap;

use asc_render::{PARTICLE_GLOW, PARTICLE_SMOKE, Particle, PointLight};
use asc_sim::content::Content;
use asc_sim::protocol::ClientMsg;
use asc_sim::tools::{self, Mode, ToolKind, ToolState, ToolUse};
use asc_world::{RayHit, World};
use glam::Vec3;

use crate::avatar::Held;
use crate::creatures::ClientCreatures;

/// Uses that repeat while the button is held go ten times a second.
const REPEAT: f32 = 0.1;
/// The manipulator works a little slower.
const TERRAFORM_REPEAT: f32 = 0.15;
/// Holding the injector's button this long uses it.
const INJECT_HOLD: f32 = 1.0;

/// This frame's input and what the tools can see.
pub struct ToolInput<'a> {
    pub held: Option<ToolKind>,
    /// Can use the scanner (has one) without holding it.
    pub has_scanner: bool,
    pub lmb: (bool, bool),
    pub rmb: (bool, bool),
    pub mode_key: bool,
    pub scan_key: bool,
    pub wheel: f32,
    pub eye: Vec3,
    pub dir: Vec3,
    /// Where the held tool's business end is.
    pub muzzle: Vec3,
    /// The ground (and buildings, trees) along a ray: origin, unit direction, reach.
    pub ray: &'a dyn Fn(Vec3, Vec3, f32) -> Option<RayHit>,
    pub creatures: &'a ClientCreatures,
    pub beacons: &'a BTreeMap<u32, Vec3>,
    pub world: &'a World,
    pub content: &'a Content,
    pub dt: f32,
    pub clock: f64,
}

struct Spark {
    at: Vec3,
    vel: Vec3,
    age: f32,
    life: f32,
    color: [f32; 3],
    size: f32,
    smoke: bool,
}

/// A rifle bolt in flight.
struct Bolt {
    from: Vec3,
    to: Vec3,
    age: f32,
}

#[derive(Default)]
pub struct Tools {
    pub mode: Mode,
    /// The manipulator's brush radius.
    pub brush: f32,
    /// Heat and charge, from the server.
    pub state: ToolState,
    /// The laser's bite under way: where, the sample byte, and how long it's been on it.
    laser: Option<(Vec3, u8, f32)>,
    /// Time to the next repeat of a held use.
    repeat: f32,
    last_shot: f64,
    /// When the last shot and injection were (game seconds), for the arms.
    pub shot_at: Option<f64>,
    pub inject_at: Option<f64>,
    inject_hold: f32,
    /// The torch held high.
    pub raised: bool,
    /// The beam this frame: from, to, colour (the laser and manipulator).
    beam: Option<(Vec3, Vec3, [f32; 3])>,
    /// Throwing flame this frame, along which way.
    flaming: Option<Vec3>,
    bolts: Vec<Bolt>,
    sparks: Vec<Spark>,
    /// A muzzle flash: where, and how long ago.
    flash: Option<(Vec3, f32)>,
    /// The view model's kick (0..1), easing off.
    pub recoil: f32,
    /// The manipulator's brush where it'd land (for the preview sphere).
    pub brush_at: Option<(Vec3, f32)>,
    pub scanner: crate::scanner::Scanner,
    /// Sounds to play: name, volume, and where (or at the player).
    pub sounds: Vec<(&'static str, f32, Option<Vec3>)>,
    sound_due: f32,
    was_locked: bool,
    rng: u64,
}

impl Tools {
    pub fn new() -> Self {
        Self { brush: 1.5, rng: 0x7001_5eed, ..Default::default() }
    }

    fn random(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(0x1405_7B7E_F767_814F);
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    fn spray(&mut self, at: Vec3, n: usize, speed: f32, color: [f32; 3], smoke: bool) {
        for _ in 0..n {
            let v = Vec3::new(self.random() - 0.5, self.random() * 0.8, self.random() - 0.5).normalize_or(Vec3::Y) * speed * (0.4 + self.random());
            let life = if smoke { 0.8 + self.random() * 0.6 } else { 0.25 + self.random() * 0.35 };
            let size = if smoke { 0.15 + self.random() * 0.15 } else { 0.02 + self.random() * 0.03 };
            self.sparks.push(Spark { at, vel: v, age: 0.0, life, color, size, smoke });
        }
    }

    /// The server's word on heat and charge.
    pub fn set_state(&mut self, s: ToolState) {
        self.state = s;
    }

    /// What the arms are doing, for the avatar.
    pub fn held_pose(&self, held: Option<ToolKind>, clock: f64) -> Held {
        let since = |t: Option<f64>| t.map(|t| (clock - t) as f32);
        match held {
            None => Held::Nothing,
            Some(ToolKind::Multitool) => match since(self.shot_at) {
                Some(t) if t < 0.35 => Held::Shoot(t),
                _ => Held::Aim,
            },
            Some(ToolKind::Manipulator) => Held::Aim,
            Some(ToolKind::Scanner) => match self.scanner.since(clock) {
                Some(t) if t < 4.0 => Held::Scan(t),
                _ => Held::Carry,
            },
            Some(ToolKind::Torch) => Held::Torch,
            Some(ToolKind::Injector) => match since(self.inject_at) {
                Some(t) if t < 1.6 => Held::Inject(t),
                _ => Held::Carry,
            },
            Some(ToolKind::Beacon) => Held::Carry,
        }
    }

    /// One frame: what the player did with what they hold (messages for the server), and the
    /// effects moved on.
    pub fn update(&mut self, i: &ToolInput) -> Vec<ClientMsg> {
        let mut out = Vec::new();
        self.beam = None;
        self.flaming = None;
        self.brush_at = None;
        self.recoil = (self.recoil - i.dt * 5.0).max(0.0);
        self.repeat -= i.dt;
        self.sound_due -= i.dt;
        let locked = i.held.is_some_and(|k| self.state.locked(k));
        if locked && !self.was_locked {
            self.sounds.push(("tool_overheat", 0.7, None));
        }
        self.was_locked = locked;
        match i.held {
            Some(ToolKind::Multitool) => self.multitool(i, locked, &mut out),
            Some(ToolKind::Manipulator) => self.manipulator(i, locked, &mut out),
            Some(ToolKind::Torch) => {
                if i.lmb.0 {
                    self.raised = !self.raised;
                }
            }
            Some(ToolKind::Beacon) => {
                if i.rmb.0
                    && let Some(hit) = (i.ray)(i.eye, i.dir, 6.0)
                    && hit.normal.y > 0.5
                {
                    out.push(ClientMsg::Tool(ToolUse::PlaceBeacon { at: hit.point }));
                    self.sounds.push(("tool_beacon", 0.6, Some(hit.point)));
                }
                if i.lmb.0
                    && let Some((&id, _)) = i.beacons.iter().find(|(_, p)| {
                        let to = **p + Vec3::Y * 0.4 - i.eye;
                        to.length() < tools::BEACON_REACH + 1.6 && to.normalize_or_zero().dot(i.dir) > 0.93
                    })
                {
                    out.push(ClientMsg::Tool(ToolUse::TakeBeacon { id }));
                }
            }
            Some(ToolKind::Injector) => {
                if i.rmb.1 {
                    self.inject_hold += i.dt;
                    if self.inject_hold >= INJECT_HOLD {
                        self.inject_hold = f32::MIN;
                        self.inject_at = Some(i.clock);
                        out.push(ClientMsg::Tool(ToolUse::Inject));
                        self.sounds.push(("tool_inject", 0.7, None));
                    }
                } else {
                    self.inject_hold = 0.0;
                }
            }
            Some(ToolKind::Scanner) | None => {}
        }
        if locked && self.random() < i.dt * 12.0 {
            // Venting heat.
            self.spray(i.muzzle, 1, 0.6, [0.8, 0.8, 0.8], true);
        }
        self.step_effects(i.dt);
        out
    }

    /// Whether a scan should go now (the C key, or the scanner's trigger).
    pub fn wants_scan(&self, i: &ToolInput) -> bool {
        self.scanner.ready(i.clock)
            && ((i.scan_key && i.has_scanner) || (i.held == Some(ToolKind::Scanner) && i.lmb.0))
    }

    fn multitool(&mut self, i: &ToolInput, locked: bool, out: &mut Vec<ClientMsg>) {
        if i.mode_key {
            self.mode = self.mode.next();
            self.laser = None;
            self.sounds.push(("tool_switch", 0.5, None));
        }
        if locked {
            self.laser = None;
            return;
        }
        let repeat = self.repeat <= 0.0;
        match self.mode {
            Mode::Laser if i.lmb.1 => {
                if repeat {
                    self.repeat = REPEAT;
                    out.push(ClientMsg::Tool(ToolUse::Hold));
                }
                let ground = (i.ray)(i.eye, i.dir, tools::LASER_REACH);
                let ground_d = ground.map_or(tools::LASER_REACH, |h| h.point.distance(i.eye));
                let creature = i.creatures.pick(i.eye, i.dir, ground_d).filter(|c| !c.dead);
                let end = match (creature, ground) {
                    (Some(c), _) => {
                        if repeat {
                            out.push(ClientMsg::Tool(ToolUse::Zap { id: c.id }));
                        }
                        self.laser = None;
                        let at = i.eye + i.dir * c.distance;
                        if self.random() < i.dt * 20.0 {
                            self.spray(at, 2, 2.0, [1.0, 0.4, 0.3], false);
                        }
                        at
                    }
                    (None, Some(hit)) => {
                        let (_, byte) = i.world.sample_raw(hit.point - hit.normal * 0.3);
                        let same = self.laser.is_some_and(|(p, b, _)| p.distance(hit.point) < 0.6 && b == byte);
                        let progress = if same { self.laser.map_or(0.0, |l| l.2) } else { 0.0 } + i.dt;
                        let needed = i.content.sample(byte).hardness / tools::LASER_SPEED;
                        if byte != asc_world::Material::Air as u8 && progress >= needed {
                            out.push(ClientMsg::Tool(ToolUse::Laser { point: hit.point, normal: hit.normal }));
                            self.laser = None;
                            self.spray(hit.point, 6, 1.5, [0.45, 0.38, 0.3], true);
                        } else {
                            self.laser = Some((hit.point, byte, progress));
                        }
                        if self.random() < i.dt * 30.0 {
                            self.spray(hit.point + hit.normal * 0.05, 2, 3.0, [1.0, 0.8, 0.5], false);
                        }
                        hit.point
                    }
                    (None, None) => {
                        self.laser = None;
                        i.eye + i.dir * tools::LASER_REACH
                    }
                };
                self.beam = Some((i.muzzle, end, [0.3, 0.9, 1.0]));
                if self.sound_due <= 0.0 {
                    self.sound_due = 0.3;
                    self.sounds.push(("tool_laser", 0.3, None));
                }
            }
            Mode::Rifle if i.lmb.0 || (i.lmb.1 && i.clock - self.last_shot > 0.35) => {
                if i.clock - self.last_shot < tools::RIFLE_INTERVAL as f64 || self.state.charge < 1.0 {
                    return;
                }
                self.last_shot = i.clock;
                self.shot_at = Some(i.clock);
                self.state.charge -= 1.0;
                self.recoil = 1.0;
                out.push(ClientMsg::Tool(ToolUse::Shoot { dir: i.dir }));
                let ground_d = (i.ray)(i.eye, i.dir, tools::RIFLE_REACH).map_or(tools::RIFLE_REACH, |h| h.point.distance(i.eye));
                let d = i.creatures.pick(i.eye, i.dir, ground_d).map_or(ground_d, |c| c.distance);
                let to = i.eye + i.dir * d;
                self.bolts.push(Bolt { from: i.muzzle, to, age: 0.0 });
                self.flash = Some((i.muzzle, 0.0));
                self.sounds.push(("tool_shot", 0.8, None));
                if d < tools::RIFLE_REACH {
                    self.spray(to, 10, 4.0, [1.0, 0.65, 0.25], false);
                    self.spray(to, 2, 1.0, [0.4, 0.38, 0.35], true);
                }
            }
            Mode::Flamer if i.lmb.1 => {
                if repeat {
                    self.repeat = REPEAT;
                    out.push(ClientMsg::Tool(ToolUse::Hold));
                    out.push(ClientMsg::Tool(ToolUse::Flame { dir: i.dir }));
                }
                self.flaming = Some(i.dir);
                // Tongues of flame out along the aim, spreading and slowing.
                let n = (i.dt * 120.0).ceil() as usize;
                for _ in 0..n {
                    let spread = Vec3::new(self.random() - 0.5, self.random() - 0.5, self.random() - 0.5) * 0.35;
                    let vel = (i.dir + spread).normalize_or(i.dir) * (10.0 + self.random() * 4.0);
                    let life = 0.55 + self.random() * 0.2;
                    self.sparks.push(Spark { at: i.muzzle, vel, age: 0.0, life, color: [1.0, 0.55, 0.15], size: 0.08, smoke: false });
                }
                if self.random() < i.dt * 10.0 {
                    let at = i.muzzle + i.dir * tools::FLAME_REACH * 0.8;
                    self.spray(at, 1, 1.0, [0.2, 0.18, 0.17], true);
                }
                if self.sound_due <= 0.0 {
                    self.sound_due = 0.4;
                    self.sounds.push(("tool_flame", 0.5, None));
                }
            }
            _ => self.laser = None,
        }
    }

    fn manipulator(&mut self, i: &ToolInput, locked: bool, out: &mut Vec<ClientMsg>) {
        if i.wheel != 0.0 {
            self.brush = (self.brush + i.wheel.signum() * 0.2).clamp(tools::BRUSH[0], tools::BRUSH[1]);
        }
        let hit = (i.ray)(i.eye, i.dir, tools::TERRAFORM_REACH);
        if let Some(h) = hit {
            self.brush_at = Some((h.point, self.brush));
        }
        let (dig, fill) = (i.lmb.1, i.rmb.1);
        if locked || !(dig || fill) {
            return;
        }
        let Some(hit) = hit else { return };
        self.beam = Some((i.muzzle, hit.point, if dig { [0.3, 1.0, 0.6] } else { [0.9, 0.8, 0.3] }));
        if self.repeat <= 0.0 {
            self.repeat = TERRAFORM_REPEAT;
            out.push(ClientMsg::Tool(ToolUse::Hold));
            out.push(ClientMsg::Tool(ToolUse::Terraform { point: hit.point, normal: hit.normal, radius: self.brush, fill: fill && !dig }));
            self.spray(hit.point, 6, 2.0, [0.45, 0.38, 0.3], true);
        }
        if self.sound_due <= 0.0 {
            self.sound_due = 0.5;
            self.sounds.push(("tool_terraform", 0.35, Some(hit.point)));
        }
    }

    fn step_effects(&mut self, dt: f32) {
        for s in &mut self.sparks {
            s.age += dt;
            s.at += s.vel * dt;
            if s.smoke {
                s.vel = s.vel * (1.0 - dt) + Vec3::Y * dt * 1.5;
            } else {
                s.vel *= 1.0 - dt * 2.5;
                s.vel.y -= dt * 3.0;
            }
        }
        self.sparks.retain(|s| s.age < s.life);
        for b in &mut self.bolts {
            b.age += dt;
        }
        self.bolts.retain(|b| b.age < 0.15);
        if let Some((_, t)) = &mut self.flash {
            *t += dt;
            if *t > 0.08 {
                self.flash = None;
            }
        }
    }

    /// The tools' particles this frame.
    pub fn particles(&mut self, held: Option<ToolKind>, muzzle: Vec3, beacons: &BTreeMap<u32, Vec3>, clock: f64, out: &mut Vec<Particle>) {
        let glow = |p: Vec3, size: f32, c: [f32; 3], bright: f32, seed: f32| Particle {
            position: p.to_array(),
            size,
            color: [c[0], c[1], c[2], bright],
            params: [PARTICLE_GLOW, 0.0, 0.0, seed],
        };
        if let Some((from, to, c)) = self.beam {
            let len = from.distance(to);
            let n = (len / 0.12).ceil().max(1.0) as usize;
            for k in 0..=n {
                let flicker = 0.7 + 0.6 * self.random();
                out.push(glow(from.lerp(to, k as f32 / n as f32), 0.035, c, 2.5 * flicker, k as f32));
            }
            out.push(glow(to, 0.25, c, 3.0, 0.0));
            out.push(glow(from, 0.08, c, 3.0, 1.0));
        }
        for b in &self.bolts {
            let f = (b.age / 0.08).min(1.0);
            let head = b.from.lerp(b.to, f);
            let tail = b.from.lerp(b.to, (f - 0.25).max(0.0));
            for k in 0..8 {
                out.push(glow(tail.lerp(head, k as f32 / 7.0), 0.05, [1.0, 0.7, 0.3], 4.0, k as f32));
            }
        }
        if let Some((at, t)) = self.flash {
            out.push(glow(at, 0.2 * (1.0 - t / 0.08), [1.0, 0.75, 0.4], 6.0, 3.0));
        }
        for s in &self.sparks {
            let f = s.age / s.life;
            if s.smoke {
                out.push(Particle {
                    position: s.at.to_array(),
                    size: s.size * (1.0 + f * 3.0),
                    color: [s.color[0], s.color[1], s.color[2], 0.5 * (1.0 - f)],
                    params: [PARTICLE_SMOKE, 0.0, f * 2.0, s.size * 97.0],
                });
            } else {
                // Flame goes from yellow-orange to dull red as it grows and slows.
                let c = [s.color[0], s.color[1] * (1.0 - f * 0.6), s.color[2] * (1.0 - f)];
                let size = if s.size > 0.06 { s.size * (1.0 + f * 6.0) } else { s.size };
                out.push(glow(s.at, size, c, 3.0 * (1.0 - f), s.vel.x));
            }
        }
        if held == Some(ToolKind::Torch) {
            let flick = 0.8 + 0.4 * self.random();
            out.push(glow(muzzle, if self.raised { 0.12 } else { 0.08 }, [1.0, 0.8, 0.5], 3.0 * flick, 7.0));
        }
        // Beacons' light pillars and blinking tops.
        let blink = (clock * 2.0).fract() < 0.5;
        for (id, p) in beacons {
            for k in 0..24 {
                let h = 0.9 + k as f32 * 1.5;
                out.push(glow(*p + Vec3::Y * h, 0.08, [0.3, 0.85, 1.0], 1.5 * (1.0 - k as f32 / 24.0), *id as f32 + k as f32));
            }
            if blink {
                out.push(glow(*p + Vec3::Y * 0.8, 0.15, [0.4, 0.95, 1.0], 5.0, *id as f32));
            }
        }
        self.scanner.particles(clock, out);
    }

    /// The tools' lights this frame.
    pub fn lights(&mut self, held: Option<ToolKind>, muzzle: Vec3, beacons: &BTreeMap<u32, Vec3>, out: &mut Vec<PointLight>) {
        if let Some((from, to, c)) = self.beam {
            let c = Vec3::from(c);
            out.push(PointLight { position: to, radius: 6.0, color: c * 3.0 });
            out.push(PointLight { position: from, radius: 3.0, color: c * 1.5 });
        }
        if let Some((at, _)) = self.flash {
            out.push(PointLight { position: at, radius: 10.0, color: Vec3::new(1.0, 0.7, 0.4) * 6.0 });
        }
        if let Some(dir) = self.flaming {
            let flick = 0.8 + 0.4 * self.random();
            out.push(PointLight { position: muzzle + dir * 3.0, radius: 12.0, color: Vec3::new(1.0, 0.5, 0.15) * 6.0 * flick });
        }
        if held == Some(ToolKind::Torch) {
            let flick = 0.9 + 0.2 * self.random();
            let (r, i) = if self.raised { (22.0, 5.0) } else { (14.0, 3.0) };
            out.push(PointLight { position: muzzle + Vec3::Y * 0.1, radius: r, color: Vec3::new(1.0, 0.78, 0.5) * i * flick });
        }
        for p in beacons.values() {
            out.push(PointLight { position: *p + Vec3::Y * 0.9, radius: 6.0, color: Vec3::new(0.3, 0.85, 1.0) * 2.0 });
        }
    }
}
