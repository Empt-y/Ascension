//! Ore nodes: where the surface outcrops and underground nodes stand, as a pure function of the seed.
//!
//! Outcrops are the veins' surface hints (`OreTable::surface_hints`): a few above every vein whose top
//! comes near the surface, so they point at what lies below, as GregTech's indicator rocks do.
//! Underground nodes stand on the natural cave floors inside veins: a jittered grid over each vein's
//! footprint is scanned down through the vein for air-over-solid crossings whose rock is ore, and
//! some of those floors get a node of that ore. Both are found per scatter column, so they stream
//! with the rest of the scatter.

use asc_world::{WorldGen, ore_index};
use glam::{IVec2, Vec3};

/// Side of a scatter column in metres (the terrain chunk size): nodes are found per column.
pub const COLUMN: f32 = 32.0;

/// Grid spacing of underground node candidates, metres.
const DEEP_SPACING: f32 = 3.0;
/// Share of candidate cave floors that get a node.
const DEEP_CHANCE: f32 = 0.5;

/// Which ore nodes: the outcrops on the surface, or the nodes in caves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NodeKind {
    Surface,
    Deep,
}

/// A node's identity: its position on a quarter-metre grid (placement is deterministic, so this
/// names the same node in every session).
pub type NodeId = glam::IVec3;

pub fn node_id(p: Vec3) -> NodeId {
    (p * 4.0).round().as_ivec3()
}

/// The scatter seed nodes are placed with, from the world seed.
pub fn seed_of(world_seed: i32) -> u64 {
    world_seed as u32 as u64
}

/// The node called `id`, if there is one: its kind and ore.
pub fn find(seed: u64, worldgen: &WorldGen, id: NodeId) -> Option<(NodeKind, Node)> {
    let p = id.as_vec3() / 4.0;
    let column = (glam::Vec2::new(p.x, p.z) / COLUMN).floor().as_ivec2();
    let hit = |nodes: Vec<Node>| nodes.into_iter().find(|n| node_id(n.position) == id);
    if let Some(n) = hit(surface(seed, worldgen, column)) {
        return Some((NodeKind::Surface, n));
    }
    hit(deep(seed, worldgen, column)).map(|n| (NodeKind::Deep, n))
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node {
    pub ore: usize,
    pub position: Vec3,
    pub yaw: f32,
    pub scale: f32,
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

struct Rng(u64);

impl Rng {
    fn new(seed: u64, a: i64, b: i64, c: u64) -> Self {
        Self(splitmix(seed ^ splitmix(a as u64 ^ splitmix(b as u64 ^ splitmix(c)))))
    }

    fn next(&mut self) -> f32 {
        self.0 = splitmix(self.0);
        (self.0 >> 40) as f32 / (1u64 << 24) as f32
    }
}

fn in_column(column: IVec2, p: Vec3) -> bool {
    let lo = column.as_vec2() * COLUMN;
    p.x >= lo.x && p.x < lo.x + COLUMN && p.z >= lo.y && p.z < lo.y + COLUMN
}

/// The veins that might reach into `column`.
fn veins(worldgen: &WorldGen, column: IVec2) -> Vec<asc_world::ores::Vein> {
    let lo = column.as_vec2() * COLUMN;
    worldgen.veins_touching(Vec3::new(lo.x, -1e5, lo.y), Vec3::new(lo.x + COLUMN, 1e5, lo.y + COLUMN))
}

/// Surface outcrops in `column`.
pub fn surface(seed: u64, worldgen: &WorldGen, column: IVec2) -> Vec<Node> {
    let table = worldgen.ores();
    let mut out = Vec::new();
    for vein in veins(worldgen, column) {
        for (ore, p) in table.surface_hints(worldgen, &vein) {
            if !in_column(column, p) || worldgen.ground(p.x, p.z).underwater {
                continue;
            }
            // Not where a cave mouth breaks the height field (it would float or be buried).
            if worldgen.sample(p + Vec3::Y * 0.25).0 <= 0.0 || worldgen.sample(p - Vec3::Y * 0.25).0 >= 0.0 {
                continue;
            }
            let mut r = Rng::new(seed, (p.x * 16.0) as i64, (p.z * 16.0) as i64, 0x0ce);
            out.push(Node {
                ore,
                position: p,
                yaw: r.next() * std::f32::consts::TAU,
                scale: 0.8 + 0.4 * r.next(),
            });
        }
    }
    out
}

/// Underground nodes in `column`: on cave floors inside veins, of the ore under them.
pub fn deep(seed: u64, worldgen: &WorldGen, column: IVec2) -> Vec<Node> {
    let lo = column.as_vec2() * COLUMN;
    let mut out = Vec::new();
    for vein in veins(worldgen, column) {
        let reach = vein.reach();
        let g0 = ((vein.center.x - reach).max(lo.x) / DEEP_SPACING).floor() as i64;
        let g1 = ((vein.center.x + reach).min(lo.x + COLUMN) / DEEP_SPACING).floor() as i64;
        let h0 = ((vein.center.z - reach).max(lo.y) / DEEP_SPACING).floor() as i64;
        let h1 = ((vein.center.z + reach).min(lo.y + COLUMN) / DEEP_SPACING).floor() as i64;
        for gz in h0..=h1 {
            for gx in g0..=g1 {
                let mut r = Rng::new(seed, gx, gz, vein.seed as u64);
                let x = (gx as f32 + r.next()) * DEEP_SPACING;
                let z = (gz as f32 + r.next()) * DEEP_SPACING;
                let (yaw, scale, roll) = (r.next() * std::f32::consts::TAU, 0.7 + 0.4 * r.next(), r.next());
                if roll >= DEEP_CHANCE || !in_column(column, Vec3::new(x, 0.0, z)) {
                    continue;
                }
                if Vec3::new(x - vein.center.x, 0.0, z - vein.center.z).length() > reach {
                    continue;
                }
                let line = worldgen.line(x, z);
                // Every floor inside the vein, top down: solid at y, air a metre above.
                let mut above = line(vein.center.y + reach + 1.0).0;
                let mut y = vein.center.y + reach;
                while y > vein.center.y - reach {
                    let here = line(y).0;
                    if above > 0.0 && here < 0.0 {
                        let (mut solid, mut air) = (y, y + 1.0);
                        for _ in 0..6 {
                            let mid = 0.5 * (solid + air);
                            if line(mid).0 < 0.0 { solid = mid } else { air = mid }
                        }
                        let floor = 0.5 * (solid + air);
                        if let Some(node) = floor_node(worldgen, &line, Vec3::new(x, floor, z), yaw, scale) {
                            out.push(node);
                        }
                    }
                    above = here;
                    y -= 1.0;
                }
            }
        }
    }
    out
}

/// A node standing at `p` on a cave floor, if the floor is flat enough, has headroom and is ore.
fn floor_node(
    worldgen: &WorldGen,
    line: &impl Fn(f32) -> (f32, u8),
    p: Vec3,
    yaw: f32,
    scale: f32,
) -> Option<Node> {
    let ore = ore_index(line(p.y - 0.4).1)?;
    if line(p.y + 1.6).0 <= 0.0 {
        return None;
    }
    let e = 0.5;
    let d = |q: Vec3| worldgen.sample(q).0;
    let normal = Vec3::new(
        d(p + Vec3::X * e) - d(p - Vec3::X * e),
        d(p + Vec3::Y * e) - d(p - Vec3::Y * e),
        d(p + Vec3::Z * e) - d(p - Vec3::Z * e),
    )
    .normalize_or_zero();
    if normal.y < 0.8 {
        return None;
    }
    // Sunk a little so the flat base doesn't hover over the bumps of the floor.
    Some(Node { ore, position: p - Vec3::Y * 0.12, yaw, scale })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nodes_are_deterministic_and_sit_on_ore() {
        let worldgen = WorldGen::new(1337);
        let (mut surface_count, mut deep_count) = (0, 0);
        for i in 0..60 {
            let column = IVec2::new(i * 5 - 150, (i * 17) % 90 - 45);
            let s = surface(7, &worldgen, column);
            assert_eq!(s, surface(7, &worldgen, column));
            for n in &s {
                assert!(in_column(column, n.position));
                assert!((n.position.y - worldgen.height(n.position.x, n.position.z)).abs() < 1e-3);
            }
            surface_count += s.len();
            let d = deep(7, &worldgen, column);
            assert_eq!(d, deep(7, &worldgen, column));
            for n in &d {
                assert!(in_column(column, n.position));
                let below = worldgen.sample(n.position + Vec3::Y * (0.12 - 0.4));
                assert_eq!(ore_index(below.1), Some(n.ore), "node stands on its own ore");
                assert!(worldgen.sample(n.position + Vec3::Y * 1.0).0 > 0.0, "room above");
            }
            deep_count += d.len();
        }
        println!("{surface_count} outcrops, {deep_count} underground nodes in 60 columns");
        assert!(surface_count > 0 && deep_count > 0);
    }
}
