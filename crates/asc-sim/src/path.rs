//! Finding the way: A* over the navigation grid (`nav`), a little at a time.
//!
//! Animals ask for paths (`PathFinder::request`); a short way in a straight line is answered at
//! once, anything else is searched over the following ticks within a shared budget of cells
//! (`PathFinder::step`), so one long search never stalls a tick. A search that runs out of cells,
//! or can't reach its goal, gives the way to the nearest place it found: good enough for a chase or
//! a long walk, which ask again as they go. Paths are smoothed by cutting corners wherever the
//! straight line is clear.

use std::collections::{BinaryHeap, VecDeque};

use ahash::AHashMap;
use glam::{IVec2, Vec2, Vec3};

use crate::nav::{COVER, Cell, NEIGHBOURS, NavGrid, Profile, cell_at, centre};

/// Straight lines shorter than this are taken without searching, when clear.
pub const DIRECT: f32 = 12.0;
/// Each search gets at least this many cells per turn.
const SLICE: usize = 250;

/// What a path is for, which changes what it prefers.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Prefs {
    /// Keep to woodland where it can (fleeing, stalking).
    pub cover: bool,
    /// How much danger (fire, lava, a flow's way) is avoided: its cost is multiplied by this.
    pub caution: f32,
    /// The most cells to search (0: `MAX_NODES`).
    pub max_nodes: usize,
}

/// Searches stop after this many cells unless told otherwise.
pub const MAX_NODES: usize = 4000;

/// A way to go: points on the ground (smoothed), the first where it starts.
#[derive(Clone, Debug, PartialEq)]
pub struct Path {
    pub points: Vec<Vec3>,
    /// It reaches the goal (else it ends as near as the search got).
    pub complete: bool,
}

struct Search {
    profile: Profile,
    prefs: Prefs,
    start: IVec2,
    goal: IVec2,
    /// The goal itself can't be stood on (water's edge for a drink): its neighbours will do.
    goal_unfit: bool,
    max_nodes: usize,
    open: BinaryHeap<Open>,
    best: AHashMap<IVec2, (f32, IVec2)>,
    expanded: usize,
    /// The nearest to the goal found so far (by straight distance), and its cost.
    closest: (IVec2, f32),
    from: Vec2,
    to: Vec2,
}

#[derive(PartialEq)]
struct Open {
    f: f32,
    g: f32,
    cell: IVec2,
}

impl Eq for Open {}

impl Ord for Open {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.f.total_cmp(&self.f).then_with(|| self.g.total_cmp(&other.g)).then_with(|| self.cell.to_array().cmp(&other.cell.to_array()))
    }
}

impl PartialOrd for Open {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Octile distance: what the cheapest way between two cells would cost on open flat ground.
fn octile(a: IVec2, b: IVec2) -> f32 {
    let d = (a - b).abs();
    let (lo, hi) = (d.x.min(d.y) as f32, d.x.max(d.y) as f32);
    hi + (std::f32::consts::SQRT_2 - 1.0) * lo
}

/// The cost of going from `a` to its neighbour `b` (`run` metres), with `danger` there.
fn step_cost(a: &Cell, b: &Cell, run: f32, prefs: &Prefs, danger: f32, profile: &Profile) -> f32 {
    let slope = if profile.swims { 0.0 } else { (b.height - a.height).abs() / run };
    let wading = if profile.swims { 0.0 } else { b.depth() * 1.5 };
    let open = if prefs.cover && !b.has(COVER) { 0.4 } else { 0.0 };
    run * (b.roughness() + slope * 2.0 + wading + open) + danger * prefs.caution
}

impl Search {
    fn new(nav: &NavGrid, from: Vec2, to: Vec2, profile: Profile, prefs: Prefs) -> Option<Self> {
        let max_nodes = if prefs.max_nodes == 0 { MAX_NODES } else { prefs.max_nodes };
        let mut start = cell_at(from);
        // Standing somewhere it doesn't fit (pushed against a trunk, wading in too deep): start from
        // the nearest cell it does, a few metres round at most.
        if !nav.cell(start).is_some_and(|c| profile.fits(c)) {
            let here = start;
            start = (1..=4)
                .flat_map(|r| (-r..=r).flat_map(move |dz| (-r..=r).map(move |dx| here + IVec2::new(dx, dz))))
                .filter(|c| nav.cell(*c).is_some_and(|c| profile.fits(c)))
                .min_by(|a, b| octile(*a, here).total_cmp(&octile(*b, here)).then(a.to_array().cmp(&b.to_array())))?;
        }
        let goal = cell_at(to);
        let goal_unfit = !nav.cell(goal).is_some_and(|c| profile.fits(c));
        let mut s = Self {
            profile,
            prefs,
            start,
            goal,
            goal_unfit,
            max_nodes,
            open: BinaryHeap::new(),
            best: AHashMap::new(),
            expanded: 0,
            closest: (start, octile(start, goal)),
            from,
            to,
        };
        s.best.insert(start, (0.0, start));
        s.open.push(Open { f: octile(start, goal), g: 0.0, cell: start });
        Some(s)
    }

    fn reached(&self, c: IVec2) -> bool {
        c == self.goal || (self.goal_unfit && (c - self.goal).abs().max_element() <= 1)
    }

    /// Expand up to `n` cells. `Some(end)` once finished: the cell the path ends at (the goal, or
    /// as near as it got) and whether that's the goal.
    fn run(&mut self, nav: &NavGrid, danger: &dyn Fn(IVec2) -> f32, n: usize, missing: &mut Vec<IVec2>) -> Option<(IVec2, bool)> {
        for _ in 0..n {
            let Some(Open { g, cell, .. }) = self.open.pop() else {
                return Some((self.closest.0, false));
            };
            if g > self.best.get(&cell).map_or(f32::MAX, |b| b.0) {
                continue;
            }
            if self.reached(cell) {
                return Some((cell, true));
            }
            self.expanded += 1;
            if self.expanded >= self.max_nodes {
                return Some((self.closest.0, false));
            }
            let Some(here) = nav.cell(cell).copied() else { continue };
            for (d, run) in NEIGHBOURS {
                let next = cell + d;
                let Some(nc) = nav.cell(next) else {
                    missing.push(next);
                    continue;
                };
                let goal_side = self.goal_unfit && next == self.goal;
                if goal_side || !self.profile.steps(&here, nc, run) {
                    continue;
                }
                // No cutting corners past obstacles.
                if run > 1.1 {
                    let sides = [cell + IVec2::new(d.x, 0), cell + IVec2::new(0, d.y)];
                    if !sides.iter().all(|s| nav.cell(*s).is_some_and(|c| self.profile.fits(c))) {
                        continue;
                    }
                }
                let ng = g + step_cost(&here, nc, run, &self.prefs, danger(next), &self.profile);
                if ng < self.best.get(&next).map_or(f32::MAX, |b| b.0) {
                    self.best.insert(next, (ng, cell));
                    let h = octile(next, self.goal);
                    if h < self.closest.1 {
                        self.closest = (next, h);
                    }
                    self.open.push(Open { f: ng + h, g: ng, cell: next });
                }
            }
        }
        None
    }

    /// The cells from the start to `end`.
    fn cells_to(&self, end: IVec2) -> Vec<IVec2> {
        let mut out = vec![end];
        let mut c = end;
        while c != self.start {
            let Some(&(_, prev)) = self.best.get(&c) else { break };
            c = prev;
            out.push(c);
        }
        out.reverse();
        out
    }
}

/// Points on the ground for a run of cells, smoothed: from each point, straight on to the furthest
/// one the line to is clear.
fn smooth(nav: &NavGrid, profile: &Profile, from: Vec2, cells: &[IVec2], to: Option<Vec2>) -> Vec<Vec3> {
    let mut pts: Vec<Vec2> = Vec::with_capacity(cells.len() + 2);
    pts.push(from);
    pts.extend(cells.iter().skip(1).map(|c| centre(*c)));
    if let Some(to) = to {
        pts.push(to);
    }
    let mut out = vec![from];
    let mut i = 0;
    while i + 1 < pts.len() {
        // The furthest point straight on from i (at least the next).
        let mut j = i + 1;
        for k in (i + 2..pts.len()).rev() {
            if k - i > 48 {
                continue;
            }
            if nav.line_clear(pts[i], pts[k], profile) {
                j = k;
                break;
            }
        }
        out.push(pts[j]);
        i = j;
    }
    out.into_iter()
        .map(|p| Vec3::new(p.x, nav.height_at(p).or_else(|| nav.cell(cell_at(p)).map(|c| c.height)).unwrap_or(0.0), p.y))
        .collect()
}

pub struct PathFinder<K> {
    searching: VecDeque<(K, Search)>,
    /// Statistics since the last `take_stats`: searches finished, cells expanded.
    finished: usize,
    expanded: usize,
}

impl<K> Default for PathFinder<K> {
    fn default() -> Self {
        Self { searching: VecDeque::new(), finished: 0, expanded: 0 }
    }
}

impl<K: Copy + PartialEq> PathFinder<K> {
    /// Find a way from `from` to `to` for `key`, replacing any search it already has. A clear short
    /// way comes back at once; otherwise the answer comes from a later `step`. `None` back and
    /// nothing searched if `from` isn't on the built grid.
    pub fn request(
        &mut self,
        nav: &NavGrid,
        key: K,
        from: Vec2,
        to: Vec2,
        profile: Profile,
        prefs: Prefs,
    ) -> Option<Path> {
        self.cancel(key);
        if from.distance(to) < DIRECT && nav.line_clear(from, to, &profile) {
            return Some(Path { points: smooth(nav, &profile, from, &[cell_at(from)], Some(to)), complete: true });
        }
        if let Some(s) = Search::new(nav, from, to, profile, prefs) {
            self.searching.push_back((key, s));
        }
        None
    }

    pub fn cancel(&mut self, key: K) {
        self.searching.retain(|(k, _)| *k != key);
    }

    pub fn searching(&self, key: K) -> bool {
        self.searching.iter().any(|(k, _)| *k == key)
    }

    /// Search on for up to `budget` cells, shared out between the searches in turn; the finished
    /// ones' paths, and the cells they found unbuilt (to ask the grid for).
    pub fn step(&mut self, nav: &NavGrid, danger: &dyn Fn(IVec2) -> f32, budget: usize) -> (Vec<(K, Path)>, Vec<IVec2>) {
        let mut done = Vec::new();
        let mut missing = Vec::new();
        let mut left = budget;
        let mut turns = self.searching.len();
        while left > 0 && turns > 0 {
            turns -= 1;
            let Some((key, mut s)) = self.searching.pop_front() else { break };
            let n = SLICE.max(budget / self.searching.len().max(1)).min(left);
            let before = s.expanded;
            let end = s.run(nav, danger, n, &mut missing);
            self.expanded += s.expanded - before;
            left = left.saturating_sub((s.expanded - before).max(1));
            match end {
                Some((end, complete)) => {
                    let cells = s.cells_to(end);
                    let to = complete.then_some(s.to).filter(|_| !s.goal_unfit);
                    let points = smooth(nav, &s.profile, s.from, &cells, to);
                    self.finished += 1;
                    done.push((key, Path { points, complete }));
                }
                None => self.searching.push_back((key, s)),
            }
        }
        missing.sort_by_key(|c| c.to_array());
        missing.dedup();
        (done, missing)
    }

    /// Searches finished and cells expanded since last asked, and searches under way.
    pub fn take_stats(&mut self) -> (usize, usize, usize) {
        (std::mem::take(&mut self.finished), std::mem::take(&mut self.expanded), self.searching.len())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nav::{BLOCKED, TILE};

    const HORSE: Profile = Profile { radius: 0.6, max_slope: 0.8, step: 0.5, water: 0.9, swims: false };

    fn flat() -> Cell {
        Cell { height: 0.0, water: 0, clear: 16, flags: 0, rough: 0 }
    }

    /// A 4x4-tile grid (128 m square from the origin) of flat ground, changed by `f`.
    fn grid(f: impl Fn(IVec2, &mut Cell)) -> NavGrid {
        let mut nav = NavGrid::default();
        let tiles: Vec<IVec2> = (0..4).flat_map(|z| (0..4).map(move |x| IVec2::new(x, z))).collect();
        nav.fill(&tiles, |c| {
            let mut cell = flat();
            f(c, &mut cell);
            cell
        });
        nav
    }

    fn find(nav: &NavGrid, from: Vec2, to: Vec2, max: usize) -> Path {
        let mut pf = PathFinder::default();
        if let Some(p) = pf.request(nav, 1u32, from, to, HORSE, Prefs { max_nodes: max, ..Default::default() }) {
            return p;
        }
        for _ in 0..1000 {
            let (done, _) = pf.step(nav, &|_| 0.0, 2000);
            if let Some((_, p)) = done.into_iter().next() {
                return p;
            }
        }
        panic!("never finished");
    }

    fn walkable(nav: &NavGrid, path: &Path) {
        for w in path.points.windows(2) {
            let (a, b) = (Vec2::new(w[0].x, w[0].z), Vec2::new(w[1].x, w[1].z));
            assert!(nav.line_clear(a, b, &HORSE), "{a} to {b} isn't clear");
        }
    }

    /// A wall along x = 40 from z = 0 to 100, with a gap at its end.
    fn wall(c: IVec2, cell: &mut Cell) {
        if c.x == 40 && c.y < 100 {
            cell.flags |= BLOCKED;
            cell.clear = 0;
        } else if (c.x - 40).abs() <= 2 && c.y < 102 {
            let d = ((c.x - 40).abs() as f32 - 0.5).max(0.0).min((c.y as f32 - 99.5).max(0.0) + 4.0);
            cell.clear = (d * 4.0) as u8;
        }
    }

    #[test]
    fn a_straight_short_way_is_taken_at_once() {
        let nav = grid(|_, _| {});
        let mut pf = PathFinder::default();
        let p = pf.request(&nav, 1u32, Vec2::new(10.0, 10.0), Vec2::new(18.0, 14.0), HORSE, Prefs::default()).unwrap();
        assert!(p.complete && p.points.len() == 2);
    }

    #[test]
    fn goes_round_a_wall() {
        let nav = grid(wall);
        let p = find(&nav, Vec2::new(20.0, 50.0), Vec2::new(60.0, 50.0), 20_000);
        assert!(p.complete);
        walkable(&nav, &p);
        let furthest = p.points.iter().map(|q| q.z).fold(0.0f32, f32::max);
        assert!(furthest > 99.0, "went round the end of the wall (got to z = {furthest})");
        // Smoothed to a few corners, not a cell-by-cell staircase.
        assert!(p.points.len() < 8, "{} points", p.points.len());
    }

    #[test]
    fn keeps_off_slopes_too_steep() {
        // A ridge too steep to climb across x = 60..64, passable only near z > 110.
        let nav = grid(|c, cell| {
            if (60..64).contains(&c.x) && c.y < 110 {
                cell.height = if c.x % 2 == 0 { 3.0 } else { 0.0 };
            }
        });
        let p = find(&nav, Vec2::new(50.0, 20.0), Vec2::new(75.0, 20.0), 30_000);
        assert!(p.complete);
        walkable(&nav, &p);
        assert!(p.points.iter().any(|q| q.z > 109.0), "it climbed the ridge");
    }

    #[test]
    fn out_of_budget_it_goes_as_far_as_it_got() {
        let nav = grid(wall);
        let p = find(&nav, Vec2::new(20.0, 50.0), Vec2::new(60.0, 50.0), 300);
        assert!(!p.complete);
        walkable(&nav, &p);
        let end = p.points.last().unwrap();
        assert!(Vec2::new(end.x, end.z).distance(Vec2::new(60.0, 50.0)) < Vec2::new(20.0, 50.0).distance(Vec2::new(60.0, 50.0)));
    }

    #[test]
    fn an_unreachable_goal_gets_as_near_as_it_can() {
        // A pond: the middle is deep water, which the horse doesn't wade; it stops at the edge.
        let nav = grid(|c, cell| {
            if centre(c).distance(Vec2::new(64.0, 64.0)) < 10.0 {
                cell.water = 30;
            }
        });
        let p = find(&nav, Vec2::new(20.0, 64.0), Vec2::new(64.0, 64.0), 20_000);
        let end = p.points.last().unwrap();
        let d = Vec2::new(end.x, end.z).distance(Vec2::new(64.0, 64.0));
        assert!((9.0..12.5).contains(&d), "stopped {d} m from the middle");
        assert_eq!(TILE, 32);
    }
}
