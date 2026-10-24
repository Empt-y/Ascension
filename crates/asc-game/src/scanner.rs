//! The survey scanner: a pulse that sweeps out from the player and marks what it finds on the HUD
//! for a while: creatures near by, ore outcrops and underground nodes, ore veins further off (with
//! the ore they're mostly of), the nearest water, volcanoes that aren't dead, and the player's
//! stashes. Waypoint beacons are always marked, scanned or not.
//!
//! Marks are drawn where their places fall on the screen (projected through the camera), or
//! pinned to the screen's edge with an arrow when they're behind or off to the side.

use std::collections::{BTreeMap, HashSet};

use asc_render::{Camera, PARTICLE_GLOW, Particle};
use asc_sim::creatures::CreatureId;
use asc_sim::ore_nodes::{self, NodeId};
use asc_world::World;
use egui::{Color32, Pos2, Stroke};
use glam::{IVec2, Vec2, Vec3};

use crate::creatures::ClientCreatures;

/// A pulse sweeps out this fast (m/s) to this far; marks last this long (seconds) and the scanner
/// is ready again after this long.
const SWEEP_SPEED: f32 = 80.0;
const SWEEP: f32 = 120.0;
const MARK_SECONDS: f64 = 25.0;
pub const COOLDOWN: f64 = 3.0;
/// How far each kind of thing is looked for (m).
const CREATURES: f32 = 120.0;
const NODES: f32 = 60.0;
const VEINS: f32 = 600.0;
const WATER: f32 = 300.0;
const VOLCANOES: f32 = 4000.0;

/// What a mark points at.
#[derive(Clone, Copy, Debug)]
enum Place {
    At(Vec3),
    /// A creature, wherever it's got to.
    Creature(CreatureId),
}

#[derive(Clone, Debug)]
struct Mark {
    place: Place,
    label: String,
    color: Color32,
    /// When it was found: marks appear as the sweep passes them.
    found: f64,
}

#[derive(Default)]
pub struct Scanner {
    /// The last pulse: when, and from where.
    pulse: Option<(f64, Vec3)>,
    marks: Vec<Mark>,
}

const CREATURE: Color32 = Color32::from_rgb(140, 230, 140);
const PREDATOR: Color32 = Color32::from_rgb(255, 90, 70);
const ORE: Color32 = Color32::from_rgb(255, 200, 90);
const VEIN: Color32 = Color32::from_rgb(255, 160, 40);
const WATER_C: Color32 = Color32::from_rgb(90, 170, 255);
const VOLCANO: Color32 = Color32::from_rgb(255, 110, 40);
const STASH: Color32 = Color32::from_rgb(230, 230, 230);
const BEACON: Color32 = Color32::from_rgb(80, 220, 255);

/// What a scan needs to know about.
pub struct ScanFrom<'a> {
    pub at: Vec3,
    pub world: &'a World,
    pub seed: i32,
    pub creatures: &'a ClientCreatures,
    pub mined: &'a HashSet<NodeId>,
    pub stashes: &'a BTreeMap<u32, Vec3>,
}

impl Scanner {
    pub fn ready(&self, clock: f64) -> bool {
        self.pulse.is_none_or(|(t, _)| clock - t >= COOLDOWN)
    }

    /// Seconds since the last pulse.
    pub fn since(&self, clock: f64) -> Option<f32> {
        self.pulse.map(|(t, _)| (clock - t) as f32)
    }

    /// Send a pulse out from `from.at` at `clock`.
    pub fn scan(&mut self, from: ScanFrom, clock: f64) {
        let at = from.at;
        self.pulse = Some((clock, at));
        self.marks.clear();
        let found = |p: Vec3| clock + (p.distance(at) / SWEEP_SPEED) as f64;
        let mut add = |place: Place, p: Vec3, label: String, color: Color32| {
            self.marks.push(Mark { place, label, color, found: found(p) });
        };
        // Creatures.
        for (id, p, c) in from.creatures.near(at, CREATURES) {
            if c.dead() {
                continue;
            }
            let k = c.kind();
            add(Place::Creature(id), p, k.name.clone(), if k.predator() { PREDATOR } else { CREATURE });
        }
        // Ore nodes: outcrops and (deeper) underground ones.
        let wg = &from.world.worldgen;
        let seed = ore_nodes::seed_of(from.seed);
        let table = asc_world::ores::builtin();
        let lo = (Vec2::new(at.x, at.z) - NODES) / ore_nodes::COLUMN;
        let hi = (Vec2::new(at.x, at.z) + NODES) / ore_nodes::COLUMN;
        for cz in lo.y.floor() as i32..=hi.y.floor() as i32 {
            for cx in lo.x.floor() as i32..=hi.x.floor() as i32 {
                let column = IVec2::new(cx, cz);
                let nodes = ore_nodes::surface(seed, wg, column).into_iter().chain(ore_nodes::deep(seed, wg, column));
                for n in nodes {
                    if n.position.distance(at) > NODES || from.mined.contains(&ore_nodes::node_id(n.position)) {
                        continue;
                    }
                    let name = &table.ores[n.ore].name;
                    let depth = wg.height(n.position.x, n.position.z) - n.position.y;
                    let label = if depth > 2.0 { format!("{name} ({depth:.0} m down)") } else { name.clone() };
                    add(Place::At(n.position), n.position, label, ORE);
                }
            }
        }
        // Veins, further: the nearest few, by their main ore.
        let mut veins = wg.veins_touching(at - Vec3::splat(VEINS), at + Vec3::splat(VEINS));
        veins.sort_by(|a, b| a.center.distance(at).total_cmp(&b.center.distance(at)));
        for v in veins.iter().filter(|v| v.center.distance(at) < VEINS).take(6) {
            let ore = &table.ores[table.veins[v.kind].ores[0]].name;
            add(Place::At(v.center), v.center, format!("{ore} vein"), VEIN);
        }
        // The nearest water (rivers, lakes, the sea), searched on a 10 m grid.
        let mut water: Option<(f32, Vec3)> = None;
        let steps = (WATER / 10.0) as i32;
        for dz in -steps..=steps {
            for dx in -steps..=steps {
                let p = Vec2::new(at.x, at.z) + Vec2::new(dx as f32, dz as f32) * 10.0;
                let d = p.distance(Vec2::new(at.x, at.z));
                if d > WATER || water.is_some_and(|(best, _)| d >= best) {
                    continue;
                }
                if let Some(level) = wg.water_level(p.x, p.y) {
                    water = Some((d, Vec3::new(p.x, level, p.y)));
                }
            }
        }
        if let Some((_, p)) = water {
            add(Place::At(p), p, "Water".into(), WATER_C);
        }
        for v in wg.volcanoes_near(Vec2::new(at.x, at.z), VOLCANOES) {
            if v.kind == asc_world::volcano::VolcanoKind::Dormant {
                continue;
            }
            let p = Vec3::new(v.center.x, v.base + v.height, v.center.y);
            add(Place::At(p), p, format!("{:?} volcano", v.kind), VOLCANO);
        }
        for p in from.stashes.values() {
            add(Place::At(*p), *p, "Your things".into(), STASH);
        }
    }

    /// The pulse's shell sweeping out: glowing points on a sphere, fading as it goes.
    pub fn particles(&self, clock: f64, out: &mut Vec<Particle>) {
        let Some((t, at)) = self.pulse else { return };
        let age = (clock - t) as f32;
        let r = age * SWEEP_SPEED;
        if r > SWEEP {
            return;
        }
        let fade = 1.0 - r / SWEEP;
        // A Fibonacci sphere's upper half (the half below ground isn't seen).
        let n = 400;
        for i in 0..n {
            let y = 1.0 - i as f32 / n as f32;
            let ring = (1.0 - y * y).sqrt();
            let a = i as f32 * 2.399_963;
            let p = at + Vec3::new(a.cos() * ring, y, a.sin() * ring) * r;
            out.push(Particle {
                position: p.to_array(),
                size: 0.25 + r * 0.01,
                color: [0.3, 0.9, 1.0, 1.6 * fade],
                params: [PARTICLE_GLOW, 0.0, 0.0, i as f32],
            });
        }
    }

    /// Draw the marks (and beacons, always) over the view.
    pub fn overlay(&self, ctx: &egui::Context, camera: &Camera, clock: f64, creatures: &ClientCreatures, beacons: &BTreeMap<u32, Vec3>) {
        let screen = ctx.content_rect();
        let aspect = screen.width() / screen.height().max(1.0);
        let view_proj = camera.view_proj(aspect);
        let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("scanner")));
        let draw = |p: Vec3, label: &str, color: Color32, alpha: f32| {
            let rel = p - camera.position;
            let clip = view_proj * rel.extend(1.0);
            let distance = rel.length();
            let text = format!("{label}  {distance:.0} m");
            let color = color.gamma_multiply(alpha);
            let in_front = clip.w > 0.0;
            let ndc = Vec2::new(clip.x, clip.y) / clip.w.abs().max(1e-4);
            let on_screen = in_front && ndc.x.abs() < 0.95 && ndc.y.abs() < 0.92;
            if on_screen {
                let at = Pos2::new((ndc.x * 0.5 + 0.5) * screen.width(), (0.5 - ndc.y * 0.5) * screen.height()) + screen.min.to_vec2();
                painter.circle_stroke(at, 5.0, Stroke::new(1.5, color));
                painter.text(at + egui::vec2(9.0, -2.0), egui::Align2::LEFT_CENTER, text, egui::FontId::proportional(13.0), color);
            } else {
                // Pinned to the edge, pointing the way to turn.
                let dir = if in_front { ndc } else { -ndc };
                let dir = Vec2::new(dir.x, -dir.y).normalize_or(Vec2::X);
                let half = Vec2::new(screen.width(), screen.height()) * 0.5 - 40.0;
                let scale = (half.x / dir.x.abs().max(1e-4)).min(half.y / dir.y.abs().max(1e-4));
                let c = screen.center();
                let at = c + egui::vec2(dir.x, dir.y) * scale;
                let tip = at + egui::vec2(dir.x, dir.y) * 10.0;
                let side = egui::vec2(-dir.y, dir.x) * 6.0;
                painter.add(egui::Shape::convex_polygon(vec![tip, at + side, at - side], color, Stroke::NONE));
                painter.text(at - egui::vec2(dir.x, dir.y) * 14.0, egui::Align2::CENTER_CENTER, text, egui::FontId::proportional(12.0), color);
            }
        };
        for m in &self.marks {
            let age = clock - m.found;
            if !(0.0..MARK_SECONDS).contains(&age) {
                continue;
            }
            let alpha = (age * 4.0).min(1.0).min((MARK_SECONDS - age) / 3.0) as f32;
            let p = match m.place {
                Place::At(p) => p,
                Place::Creature(id) => match creatures.drawn_at(id) {
                    Some(p) => p + Vec3::Y * 1.2,
                    None => continue,
                },
            };
            draw(p, &m.label, m.color, alpha);
        }
        for (id, p) in beacons {
            draw(*p + Vec3::Y * 1.0, &format!("Beacon {}", id + 1), BEACON, 1.0);
        }
    }
}
