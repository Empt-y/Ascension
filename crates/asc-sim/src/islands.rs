//! Islands: groups of machines connected (by items, fluids or power) to each other and to nothing
//! else. Nothing in one island can affect another within a tick, so islands tick in parallel.
//!
//! Connecting two nodes merges their islands (the smaller one's members are relabelled, so a node
//! is relabelled at most log n times over a whole build). Disconnecting re-walks only the island it
//! was in. An island is named by its lowest node, so the names depend only on the graph, not on
//! the order it was built in; ticking islands in name order keeps the result independent of how
//! many threads ran it.

use std::collections::BTreeMap;

use rayon::prelude::*;

pub type Node = u32;
pub type IslandId = u32;

#[derive(Default)]
pub struct Islands {
    /// Neighbours of each node (empty for removed nodes).
    edges: Vec<Vec<Node>>,
    alive: Vec<bool>,
    island: Vec<IslandId>,
    /// Members of each island, by name.
    members: BTreeMap<IslandId, Vec<Node>>,
}

impl Islands {
    /// A new node, alone in its own island.
    pub fn add(&mut self) -> Node {
        let n = self.edges.len() as Node;
        self.edges.push(Vec::new());
        self.alive.push(true);
        self.island.push(n);
        self.members.insert(n, vec![n]);
        n
    }

    pub fn island_of(&self, n: Node) -> IslandId {
        self.island[n as usize]
    }

    pub fn count(&self) -> usize {
        self.members.len()
    }

    /// The islands in name order, with their members.
    pub fn iter(&self) -> impl Iterator<Item = (IslandId, &[Node])> {
        self.members.iter().map(|(id, m)| (*id, m.as_slice()))
    }

    pub fn connect(&mut self, a: Node, b: Node) {
        if a == b || self.edges[a as usize].contains(&b) {
            return;
        }
        self.edges[a as usize].push(b);
        self.edges[b as usize].push(a);
        let (ia, ib) = (self.island_of(a), self.island_of(b));
        if ia == ib {
            return;
        }
        // Relabel the smaller island into the larger, then name the result by its lowest node.
        let (keep, gone) = if self.members[&ia].len() >= self.members[&ib].len() { (ia, ib) } else { (ib, ia) };
        let moved = self.members.remove(&gone).unwrap_or_default();
        for &n in &moved {
            self.island[n as usize] = keep;
        }
        let mut list = self.members.remove(&keep).unwrap_or_default();
        list.extend(moved);
        let name = keep.min(gone);
        if name != keep {
            for &n in &list {
                self.island[n as usize] = name;
            }
        }
        self.members.insert(name, list);
    }

    pub fn disconnect(&mut self, a: Node, b: Node) {
        self.edges[a as usize].retain(|&n| n != b);
        self.edges[b as usize].retain(|&n| n != a);
        self.rebuild(self.island_of(a));
    }

    /// Take a node out (its links go with it).
    pub fn remove(&mut self, n: Node) {
        let island = self.island_of(n);
        for m in std::mem::take(&mut self.edges[n as usize]) {
            self.edges[m as usize].retain(|&x| x != n);
        }
        self.alive[n as usize] = false;
        self.rebuild(island);
    }

    /// Re-walk one island after links were cut: it may have split.
    fn rebuild(&mut self, island: IslandId) {
        let Some(old) = self.members.remove(&island) else { return };
        let mut seen: Vec<Node> = Vec::new();
        let mut done = std::collections::HashSet::new();
        for &start in &old {
            if !self.alive[start as usize] || done.contains(&start) {
                continue;
            }
            seen.clear();
            let mut stack = vec![start];
            done.insert(start);
            while let Some(n) = stack.pop() {
                seen.push(n);
                for &m in &self.edges[n as usize] {
                    if done.insert(m) {
                        stack.push(m);
                    }
                }
            }
            seen.sort_unstable();
            let name = seen[0];
            for &n in &seen {
                self.island[n as usize] = name;
            }
            self.members.insert(name, seen.clone());
        }
    }
}

/// Run `work` on every island's state in parallel. `states` must be in island name order (as
/// `Islands::iter` gives them); each call only sees its own island's state, so the result is the
/// same however many threads run it.
pub fn tick_parallel<S: Send>(states: &mut [S], work: impl Fn(&mut S) + Sync + Send) {
    states.par_iter_mut().for_each(work);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn names(islands: &Islands) -> Vec<Vec<Node>> {
        islands.iter().map(|(_, m)| {
            let mut m = m.to_vec();
            m.sort_unstable();
            m
        }).collect()
    }

    #[test]
    fn connecting_merges_and_cutting_splits() {
        let mut g = Islands::default();
        let n: Vec<Node> = (0..6).map(|_| g.add()).collect();
        assert_eq!(g.count(), 6);
        g.connect(n[4], n[5]);
        g.connect(n[1], n[2]);
        g.connect(n[2], n[4]);
        assert_eq!(names(&g), vec![vec![0], vec![1, 2, 4, 5], vec![3]]);
        assert_eq!(g.island_of(5), 1, "named by the lowest node");
        g.disconnect(n[2], n[4]);
        assert_eq!(names(&g), vec![vec![0], vec![1, 2], vec![3], vec![4, 5]]);
        g.connect(n[0], n[3]);
        g.remove(n[0]);
        assert_eq!(names(&g), vec![vec![1, 2], vec![3], vec![4, 5]]);
        assert_eq!(g.island_of(3), 3);
    }

    #[test]
    fn names_depend_on_the_graph_not_the_build_order() {
        let build = |order: &[(Node, Node)]| {
            let mut g = Islands::default();
            for _ in 0..8 {
                g.add();
            }
            for &(a, b) in order {
                g.connect(a, b);
            }
            (0..8).map(|n| g.island_of(n)).collect::<Vec<_>>()
        };
        let edges = [(0, 3), (3, 6), (1, 7), (2, 5), (5, 4)];
        let mut reversed = edges;
        reversed.reverse();
        assert_eq!(build(&edges), build(&reversed));
    }

    #[test]
    fn a_parallel_tick_matches_a_serial_one() {
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            let mut states: Vec<(u32, u64)> = (0..500).map(|i| (i, i as u64)).collect();
            pool.install(|| {
                for _ in 0..50 {
                    tick_parallel(&mut states, |(id, acc)| {
                        for k in 0..100 {
                            *acc = acc.wrapping_mul(6364136223846793005).wrapping_add(*id as u64 + k);
                        }
                    });
                }
            });
            states
        };
        assert_eq!(run(1), run(8));
    }
}
