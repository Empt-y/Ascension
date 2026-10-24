//! Wildfire: the ground in `CELL`-metre cells catches fire (from lava, or a bomb landing), burns
//! for a while depending on how much fuel grows there, sets its neighbours alight as it goes, and
//! goes out burnt. Woodland burns hottest and longest, grassland fast, and bare volcanic ground,
//! desert, snow, ice and water not at all.
//!
//! Each cell's stage is what clients draw plants by (scorched as it catches, charred while it
//! burns hard, burnt out after), and what hurts a player standing in it. Ticks every `EVERY` ticks;
//! deterministic from the seed and the tick.

use std::collections::BTreeMap;

use asc_world::{Biome, WorldGen};
use glam::{IVec2, Vec2, Vec3};
use serde::{Deserialize, Serialize};

/// Fire cells are this many metres across.
pub const CELL: f32 = 4.0;
/// The fire runs every this many ticks.
const EVERY: u64 = 10;
/// How long a cell burns (ticks): this long, plus up to `BURN_MORE` with more fuel.
const BURN_TICKS: u64 = 400;
const BURN_MORE: u64 = 800;
/// Chance each step that a burning cell lights a neighbour, at full fuel on both sides.
const SPREAD: f32 = 0.12;
/// No more than this many cells burn at once (a world on fire would cost without end).
const MAX_BURNING: usize = 6000;
/// Damage each fire step to a player standing in a burning cell (at full fuel).
const BURN_DAMAGE: i32 = 30;

/// How burnt a cell is, as plants show it: 0 untouched, 1 scorched (catching), 2 charred (burning
/// hard), 3 burnt out.
pub type Stage = u8;

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
struct Cell {
    /// When it caught (while burning), or `None` once it's burnt out.
    since: Option<u64>,
    fuel: f32,
}

#[derive(Default)]
pub struct Fire {
    cells: BTreeMap<[i32; 2], Cell>,
    /// Cells whose stage changed since the last `take_changed`.
    changed: BTreeMap<[i32; 2], Stage>,
    /// How many of `cells` are burning (the rest are burnt out).
    burning: usize,
    seed: u64,
}

/// The cell holding world point (x, z).
pub fn cell_of(x: f32, z: f32) -> IVec2 {
    (Vec2::new(x, z) / CELL).floor().as_ivec2()
}

/// How much burns at (x, z), 0..1.
pub fn fuel(worldgen: &WorldGen, x: f32, z: f32) -> f32 {
    let g = worldgen.ground(x, z);
    if g.underwater {
        return 0.0;
    }
    let base = match g.biome {
        Biome::Forest | Biome::Taiga | Biome::Jungle => 0.55 + 0.45 * g.forest,
        Biome::Meadow | Biome::Savanna => 0.55,
        Biome::Swamp => 0.3,
        Biome::Tundra => 0.25,
        Biome::Shore => 0.12,
        Biome::Desert => 0.04,
        Biome::Arctic => 0.0,
    };
    base * (1.0 - g.barren)
}

fn hash(seed: u64, cell: [i32; 2], tick: u64, salt: u64) -> f32 {
    let mut h = seed ^ salt ^ tick.wrapping_mul(0x9E37_79B9_7F4A_7C15);
    for v in cell {
        h ^= v as u32 as u64;
        h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9).rotate_left(31);
    }
    (h >> 40) as f32 / (1u64 << 24) as f32
}

impl Fire {
    pub fn new(seed: i32) -> Self {
        Self { seed: seed as u32 as u64 ^ 0x6669_7265, ..Default::default() }
    }

    /// The middles of the cells burning now.
    pub fn burning_cells(&self) -> impl Iterator<Item = glam::Vec2> + '_ {
        self.cells
            .iter()
            .filter(|(_, c)| c.since.is_some())
            .map(|(k, _)| (glam::Vec2::new(k[0] as f32, k[1] as f32) + 0.5) * CELL)
    }

    /// Restore from a save: burnt-out cells (and any still burning, as burnt).
    pub fn restore(&mut self, burnt: impl IntoIterator<Item = [i32; 2]>) {
        for c in burnt {
            self.cells.insert(c, Cell { since: None, fuel: 0.0 });
        }
    }

    /// Every cell that has burnt or is burning (for saves and newly joined clients), with stages.
    pub fn stages(&self, ticks: u64) -> Vec<([i32; 2], Stage)> {
        self.cells.iter().map(|(k, c)| (*k, Self::stage(c, ticks))).collect()
    }

    fn stage(c: &Cell, ticks: u64) -> Stage {
        match c.since {
            None => 3,
            Some(since) => {
                let duration = BURN_TICKS + (BURN_MORE as f32 * c.fuel) as u64;
                if ticks.saturating_sub(since) < duration / 4 { 1 } else { 2 }
            }
        }
    }

    pub fn burning(&self) -> usize {
        self.burning
    }

    /// Set the ground at (x, z) alight, if it'll burn and isn't already burning or burnt.
    pub fn ignite(&mut self, worldgen: &WorldGen, x: f32, z: f32, ticks: u64) {
        let c = cell_of(x, z);
        let key = c.to_array();
        if self.cells.contains_key(&key) || self.burning() >= MAX_BURNING {
            return;
        }
        let centre = (c.as_vec2() + 0.5) * CELL;
        let fuel = fuel(worldgen, centre.x, centre.y);
        if fuel < 0.05 {
            return;
        }
        self.cells.insert(key, Cell { since: Some(ticks), fuel });
        self.burning += 1;
        self.changed.insert(key, 1);
    }

    /// Stages that changed since the last call.
    pub fn take_changed(&mut self) -> Vec<([i32; 2], Stage)> {
        std::mem::take(&mut self.changed).into_iter().collect()
    }

    /// One tick: every `EVERY` ticks, burning cells spread to their neighbours and burn out.
    /// Returns damage to a player at `player` standing in fire.
    pub fn tick(&mut self, worldgen: &WorldGen, ticks: u64, player: Vec3) -> i32 {
        if !ticks.is_multiple_of(EVERY) {
            return 0;
        }
        let burning: Vec<([i32; 2], Cell)> = self.cells.iter().filter(|(_, c)| c.since.is_some()).map(|(k, c)| (*k, *c)).collect();
        let mut lit = Vec::new();
        for (key, c) in &burning {
            let since = c.since.expect("burning");
            let duration = BURN_TICKS + (BURN_MORE as f32 * c.fuel) as u64;
            let age = ticks.saturating_sub(since);
            if age >= duration {
                self.cells.insert(*key, Cell { since: None, fuel: c.fuel });
                self.burning -= 1;
                self.changed.insert(*key, 3);
                continue;
            }
            if age >= duration / 4 && age < duration / 4 + EVERY {
                self.changed.insert(*key, 2);
            }
            // Burning hardest in the middle of its life.
            let heat = c.fuel * (1.0 - ((age as f32 / duration as f32) * 2.0 - 1.0).abs()).max(0.3);
            for (i, d) in [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (1, -1), (-1, 1), (1, 1)].into_iter().enumerate() {
                let n = [key[0] + d.0, key[1] + d.1];
                if self.cells.contains_key(&n) {
                    continue;
                }
                let diagonal = if i >= 4 { 0.6 } else { 1.0 };
                // Neighbours' fuel is looked up only when the dice might light them.
                let roll = hash(self.seed, n, ticks, i as u64);
                if roll >= SPREAD * heat * diagonal {
                    continue;
                }
                let centre = (IVec2::from_array(n).as_vec2() + 0.5) * CELL;
                let f = fuel(worldgen, centre.x, centre.y);
                if roll < SPREAD * heat * diagonal * f {
                    lit.push((n, f));
                }
            }
        }
        for (n, f) in lit {
            if self.burning >= MAX_BURNING {
                break;
            }
            if let std::collections::btree_map::Entry::Vacant(e) = self.cells.entry(n) {
                e.insert(Cell { since: Some(ticks), fuel: f });
                self.burning += 1;
                self.changed.insert(n, 1);
            }
        }
        // Standing in fire.
        let here = cell_of(player.x, player.z).to_array();
        match self.cells.get(&here) {
            Some(c) if c.since.is_some() && player.y < worldgen.height(player.x, player.z) + 4.0 => {
                (BURN_DAMAGE as f32 * (0.4 + 0.6 * c.fuel)) as i32
            }
            _ => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A spot whose cell and every neighbour within `reach` cells has at least `min` fuel.
    fn find(wg: &WorldGen, min: f32, max: f32) -> (f32, f32) {
        (0..20000)
            .map(|i| ((i % 140) as f32 * 37.0 - 2600.0, (i / 140) as f32 * 41.0 - 2900.0))
            .find(|&(x, z)| (0..9).all(|k| { let f = fuel(wg, x + (k % 3) as f32 * 8.0 - 8.0, z + (k / 3) as f32 * 8.0 - 8.0); f >= min && f <= max }))
            .expect("such ground")
    }

    #[test]
    fn fire_spreads_through_woods_and_burns_out() {
        let wg = WorldGen::new(1337);
        let (x, z) = find(&wg, 0.7, 1.0);
        let mut fire = Fire::new(1337);
        fire.ignite(&wg, x, z, 0);
        let far = Vec3::new(x + 5000.0, 0.0, z);
        let mut most = 0;
        for t in 1..=3000 {
            fire.tick(&wg, t, far);
            most = most.max(fire.burning());
        }
        let burnt = fire.stages(3000).iter().filter(|(_, s)| *s == 3).count();
        assert!(most > 15, "only {most} cells burning at once");
        assert!(burnt > 30, "only {burnt} cells burnt");
        // Deterministic.
        let mut again = Fire::new(1337);
        again.ignite(&wg, x, z, 0);
        for t in 1..=3000 {
            again.tick(&wg, t, far);
        }
        assert_eq!(again.stages(3000), fire.stages(3000));
    }

    #[test]
    fn bare_ground_doesnt_burn_and_standing_in_fire_hurts() {
        let wg = WorldGen::new(1337);
        // Desert, ice or water: nothing to burn.
        let (x, z) = find(&wg, 0.0, 0.03);
        let mut fire = Fire::new(1);
        fire.ignite(&wg, x, z, 0);
        assert_eq!(fire.burning(), 0);
        // Woods: standing in the fire hurts.
        let (x, z) = find(&wg, 0.7, 1.0);
        fire.ignite(&wg, x, z, 0);
        let player = Vec3::new(x, wg.height(x, z) + 0.5, z);
        let hurt: i32 = (1..=100).map(|t| fire.tick(&wg, t, player)).sum();
        assert!(hurt > 100, "only {hurt} damage standing in fire");
    }
}
