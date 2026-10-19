//! A stand-in factory for measuring the tick, before there are real machines.
//!
//! `timers` invisible machines each run recipes of 1 to 120 seconds back to back, woken through
//! the [`Schedule`] only when a recipe ends; `islands` groups of `per_island` machines each do
//! `work` steps of integer work every tick, ticked in parallel. Both are deterministic, so a
//! checksum shows the result doesn't depend on the thread count.

use crate::islands::tick_parallel;
use crate::schedule::Schedule;

#[derive(Clone, Copy, Debug)]
pub struct StressConfig {
    pub timers: u32,
    pub islands: u32,
    pub per_island: u32,
    /// Integer work steps per island machine per tick.
    pub work: u32,
}

impl StressConfig {
    /// A big base that is mostly waiting on recipes, plus some busy networks.
    pub const LIGHT: Self = Self { timers: 100_000, islands: 64, per_island: 1_000, work: 4 };
    /// Busy enough to overrun a 50 ms tick on most machines, to show the clock slowing instead of
    /// the frame rate.
    pub const HEAVY: Self = Self { timers: 100_000, islands: 64, per_island: 4_000, work: 30_000 };
}

fn splitmix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// Recipe length for machine `id`'s `n`th recipe: 20 to 2400 ticks.
fn recipe_ticks(id: u32, n: u32) -> u64 {
    20 + splitmix((id as u64) << 32 | n as u64) % 2381
}

struct IslandState {
    machines: Vec<u64>,
}

pub struct Stress {
    config: StressConfig,
    schedule: Schedule<u32>,
    /// Recipes finished, per timer machine.
    finished: Vec<u32>,
    islands: Vec<IslandState>,
    woken: Vec<u32>,
    /// Wake-ups on the last tick.
    pub last_woken: usize,
}

impl Stress {
    pub fn new(config: StressConfig, now: u64) -> Self {
        let mut schedule = Schedule::new(now);
        for id in 0..config.timers {
            schedule.at(now + recipe_ticks(id, 0), id);
        }
        let islands = (0..config.islands)
            .map(|i| IslandState {
                machines: (0..config.per_island).map(|m| splitmix(((i as u64) << 32) | m as u64)).collect(),
            })
            .collect();
        Self {
            config,
            schedule,
            finished: vec![0; config.timers as usize],
            islands,
            woken: Vec::new(),
            last_woken: 0,
        }
    }

    pub fn tick(&mut self) {
        let now = self.schedule.now();
        self.woken.clear();
        self.schedule.advance(&mut self.woken);
        self.last_woken = self.woken.len();
        for &id in &self.woken {
            let n = &mut self.finished[id as usize];
            *n += 1;
            self.schedule.at(now + recipe_ticks(id, *n), id);
        }
        let work = self.config.work;
        tick_parallel(&mut self.islands, |island| {
            for m in &mut island.machines {
                let mut x = *m;
                for _ in 0..work {
                    x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                }
                *m = x;
            }
        });
    }

    /// A checksum of everything, to compare runs.
    pub fn checksum(&self) -> u64 {
        let mut h = 0u64;
        for (i, n) in self.finished.iter().enumerate() {
            h = splitmix(h ^ ((i as u64) << 32 | *n as u64));
        }
        for island in &self.islands {
            for m in &island.machines {
                h = splitmix(h ^ m);
            }
        }
        h
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipes_finish_on_schedule_and_threads_dont_change_the_result() {
        let config = StressConfig { timers: 2_000, islands: 16, per_island: 50, work: 10 };
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut s = Stress::new(config, 100);
                let mut woken = 0;
                for _ in 0..3_000 {
                    s.tick();
                    woken += s.last_woken;
                }
                (s.checksum(), woken, s.finished.clone())
            })
        };
        let (a, woken, finished) = run(1);
        assert_eq!(run(6).0, a);
        // Every machine has finished at least one recipe (the longest is 2400 ticks) and none more
        // than 3000 / 20.
        assert!(finished.iter().all(|&n| (1..=150).contains(&n)));
        assert_eq!(woken, finished.iter().map(|&n| n as usize).sum::<usize>());
    }
}
