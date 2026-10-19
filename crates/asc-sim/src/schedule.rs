//! Wake-ups at future ticks, so nothing has to be checked every tick.
//!
//! A machine that starts a 30 s recipe asks to be woken in 600 ticks and costs nothing until then;
//! a blocked machine isn't in here at all until something wakes it. A timer wheel keeps the near
//! future (`SLOTS` ticks) as one list per tick, so scheduling and waking are O(1); anything further
//! out waits in an ordered map and moves into the wheel as the horizon reaches it.
//!
//! Deterministic: everything due on a tick comes out in the order it was scheduled.

use std::collections::BTreeMap;

/// Ticks the wheel covers (a bit over 3 minutes at 20 Hz).
const SLOTS: u64 = 4096;

pub struct Schedule<T> {
    /// `wheel[t % SLOTS]` holds what's due on tick `t`, for `now <= t < now + SLOTS`.
    wheel: Vec<Vec<T>>,
    /// Further out, by tick.
    later: BTreeMap<u64, Vec<T>>,
    /// The next tick `advance` will hand out.
    now: u64,
    len: usize,
}

impl<T> Schedule<T> {
    /// A schedule whose next tick is `now`.
    pub fn new(now: u64) -> Self {
        Self { wheel: (0..SLOTS).map(|_| Vec::new()).collect(), later: BTreeMap::new(), now, len: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Wake `item` on `tick` (on the next tick, if that's already past).
    pub fn at(&mut self, tick: u64, item: T) {
        let tick = tick.max(self.now);
        self.len += 1;
        if tick < self.now + SLOTS {
            self.wheel[(tick % SLOTS) as usize].push(item);
        } else {
            self.later.entry(tick).or_default().push(item);
        }
    }

    /// Hand out everything due on the next tick (appended to `out`, in scheduling order) and step
    /// to the one after.
    pub fn advance(&mut self, out: &mut Vec<T>) {
        let slot = (self.now % SLOTS) as usize;
        self.len -= self.wheel[slot].len();
        out.append(&mut self.wheel[slot]);
        self.now += 1;
        // The wheel now reaches one tick further: bring in what's due there.
        let edge = self.now + SLOTS - 1;
        if let Some(entry) = self.later.first_entry()
            && *entry.key() == edge
        {
            let items = entry.remove();
            self.wheel[(edge % SLOTS) as usize].extend(items);
        }
    }

    /// The tick `advance` hands out next.
    pub fn now(&self) -> u64 {
        self.now
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn items_come_out_on_their_tick_in_order() {
        let mut s = Schedule::new(10);
        s.at(12, "b1");
        s.at(11, "a");
        s.at(12, "b2");
        s.at(5, "late"); // already past: next tick
        s.at(10 + SLOTS * 3 + 7, "far");
        assert_eq!(s.len(), 5);
        let mut seen = Vec::new();
        let mut out = Vec::new();
        while !s.is_empty() {
            let t = s.now();
            s.advance(&mut out);
            for item in out.drain(..) {
                seen.push((t, item));
            }
        }
        assert_eq!(seen, vec![(10, "late"), (11, "a"), (12, "b1"), (12, "b2"), (10 + SLOTS * 3 + 7, "far")]);
    }

    #[test]
    fn far_items_keep_their_order_with_near_ones() {
        let mut s = Schedule::new(0);
        let due = SLOTS + 100;
        s.at(due, 1); // far when scheduled
        for _ in 0..200 {
            let mut out = Vec::new();
            s.advance(&mut out);
            assert!(out.is_empty());
        }
        s.at(due, 2); // near now, scheduled later: comes out after
        let mut out = Vec::new();
        while s.now() <= due {
            s.advance(&mut out);
        }
        assert_eq!(out, vec![1, 2]);
    }

    #[test]
    fn every_item_comes_out_exactly_once_on_time() {
        let mut s = Schedule::new(0);
        let mut x = 12345u64;
        let mut expected = Vec::new();
        for i in 0..20_000u64 {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let tick = (x >> 33) % (SLOTS * 4);
            s.at(tick, (tick, i));
            expected.push((tick, i));
        }
        expected.sort();
        let mut got = Vec::new();
        let mut out = Vec::new();
        while !s.is_empty() {
            let t = s.now();
            s.advance(&mut out);
            for (tick, i) in out.drain(..) {
                assert_eq!(tick, t);
                got.push((tick, i));
            }
        }
        assert_eq!(got, expected);
    }
}
