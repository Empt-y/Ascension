//! The player's inventory: 36 slots (the first 9 are the hotbar) plus fractional remainders.
//!
//! Mining yields material by volume (1 item per m³), so a bite can be worth 2.8 items. The whole part
//! goes into slots; the fraction is kept per item in `remainders` so nothing is lost to rounding, and
//! placing pays from the remainder first.

use std::collections::BTreeMap;

use crate::content::{Content, ItemId};

pub const SLOTS: usize = 36;
pub const HOTBAR: usize = 9;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct Stack {
    pub item: ItemId,
    pub count: u32,
}

#[derive(Clone, Debug)]
pub struct Inventory {
    pub slots: [Option<Stack>; SLOTS],
    /// Hotbar slot in use (0..HOTBAR).
    pub selected: usize,
    pub remainders: BTreeMap<ItemId, f32>,
}

/// One thing a player does in their inventory. The client applies it at once (so the screen
/// responds immediately) and sends it; the server applies the same op and its result wins.
#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum InventoryOp {
    Click(usize),
    RightClick(usize),
    QuickMove(usize),
    /// Choose hotbar slot `n`.
    Select(usize),
    /// Put the stack on the cursor back (closing the screen).
    ReturnHeld,
}

/// An inventory as sent to the client: everything in it, and the last op it includes.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct InventorySnapshot {
    pub seq: u32,
    pub slots: Vec<Option<Stack>>,
    pub selected: usize,
    pub held: Option<Stack>,
    pub remainders: Vec<(ItemId, f32)>,
}

impl Default for Inventory {
    fn default() -> Self {
        Self { slots: [None; SLOTS], selected: 0, remainders: BTreeMap::new() }
    }
}

impl Inventory {
    /// Apply an op; returns how many items were lost (returning a held stack to a full inventory).
    pub fn apply(&mut self, content: &Content, op: InventoryOp, held: &mut Option<Stack>) -> u32 {
        match op {
            InventoryOp::Click(i) if i < SLOTS => self.click(content, i, held),
            InventoryOp::RightClick(i) if i < SLOTS => self.right_click(content, i, held),
            InventoryOp::QuickMove(i) if i < SLOTS => self.quick_move(content, i),
            InventoryOp::Select(i) if i < HOTBAR => self.selected = i,
            InventoryOp::ReturnHeld => return self.return_held(content, held),
            _ => {}
        }
        0
    }

    pub fn snapshot(&self, held: Option<Stack>, seq: u32) -> InventorySnapshot {
        InventorySnapshot {
            seq,
            slots: self.slots.to_vec(),
            selected: self.selected,
            held,
            remainders: self.remainders.iter().map(|(k, v)| (*k, *v)).collect(),
        }
    }

    /// Replace this inventory with a snapshot's contents; returns its held stack.
    pub fn restore(&mut self, s: &InventorySnapshot) -> Option<Stack> {
        for (slot, saved) in self.slots.iter_mut().zip(s.slots.iter().chain(std::iter::repeat(&None))) {
            *slot = *saved;
        }
        self.selected = s.selected.min(HOTBAR - 1);
        self.remainders = s.remainders.iter().copied().collect();
        s.held
    }

    /// Add items, filling existing stacks first and then empty slots (hotbar first). Returns how many
    /// didn't fit.
    pub fn add(&mut self, content: &Content, item: ItemId, mut count: u32) -> u32 {
        let max = content.item(item).stack;
        for slot in self.slots.iter_mut().flatten() {
            if count == 0 {
                break;
            }
            if slot.item == item && slot.count < max {
                let moved = count.min(max - slot.count);
                slot.count += moved;
                count -= moved;
            }
        }
        for slot in self.slots.iter_mut() {
            if count == 0 {
                break;
            }
            if slot.is_none() {
                let moved = count.min(max);
                *slot = Some(Stack { item, count: moved });
                count -= moved;
            }
        }
        count
    }

    pub fn count(&self, item: ItemId) -> u32 {
        self.slots.iter().flatten().filter(|s| s.item == item).map(|s| s.count).sum()
    }

    /// Remove `n` items, taking from the last slots first so the hotbar empties last. Fails (and
    /// changes nothing) if there aren't enough.
    pub fn take(&mut self, item: ItemId, mut n: u32) -> bool {
        if self.count(item) < n {
            return false;
        }
        for slot in self.slots.iter_mut().rev() {
            if n == 0 {
                break;
            }
            if let Some(s) = slot
                && s.item == item
            {
                let taken = n.min(s.count);
                s.count -= taken;
                n -= taken;
                if s.count == 0 {
                    *slot = None;
                }
            }
        }
        true
    }

    /// Credit `volume` m³ of an item. Returns the whole items that didn't fit.
    pub fn add_volume(&mut self, content: &Content, item: ItemId, volume: f32) -> u32 {
        let rem = self.remainders.entry(item).or_insert(0.0);
        *rem += volume.max(0.0);
        let whole = rem.floor();
        *rem -= whole;
        self.add(content, item, whole as u32)
    }

    /// Pay `volume` m³ of an item, from the remainder first and then whole items. Fails (and changes
    /// nothing) if there isn't enough.
    pub fn pay_volume(&mut self, item: ItemId, volume: f32) -> bool {
        let rem = self.remainders.get(&item).copied().unwrap_or(0.0);
        let need = volume - rem;
        if need <= 0.0 {
            self.remainders.insert(item, rem - volume);
            return true;
        }
        let items = need.ceil() as u32;
        if !self.take(item, items) {
            return false;
        }
        self.remainders.insert(item, rem + items as f32 - volume);
        true
    }

    pub fn selected_stack(&self) -> Option<Stack> {
        self.slots[self.selected]
    }

    /// Left click on a slot while possibly holding a stack on the cursor: pick up, put down, merge
    /// into a matching stack, or swap.
    pub fn click(&mut self, content: &Content, slot: usize, held: &mut Option<Stack>) {
        match (self.slots[slot], *held) {
            (None, None) => {}
            (Some(s), None) => {
                *held = Some(s);
                self.slots[slot] = None;
            }
            (None, Some(h)) => {
                self.slots[slot] = Some(h);
                *held = None;
            }
            (Some(s), Some(h)) if s.item == h.item => {
                let max = content.item(s.item).stack;
                let moved = h.count.min(max.saturating_sub(s.count));
                self.slots[slot] = Some(Stack { count: s.count + moved, ..s });
                *held = (h.count > moved).then_some(Stack { count: h.count - moved, ..h });
            }
            (Some(s), Some(h)) => {
                self.slots[slot] = Some(h);
                *held = Some(s);
            }
        }
    }

    /// Right click: with an empty cursor pick up half a stack; holding a stack, drop one item.
    pub fn right_click(&mut self, content: &Content, slot: usize, held: &mut Option<Stack>) {
        match (self.slots[slot], *held) {
            (Some(s), None) => {
                let half = s.count.div_ceil(2);
                *held = Some(Stack { count: half, ..s });
                self.slots[slot] = (s.count > half).then_some(Stack { count: s.count - half, ..s });
            }
            (None, Some(h)) => {
                self.slots[slot] = Some(Stack { count: 1, ..h });
                *held = (h.count > 1).then_some(Stack { count: h.count - 1, ..h });
            }
            (Some(s), Some(h)) if s.item == h.item && s.count < content.item(s.item).stack => {
                self.slots[slot] = Some(Stack { count: s.count + 1, ..s });
                *held = (h.count > 1).then_some(Stack { count: h.count - 1, ..h });
            }
            _ => {}
        }
    }

    /// Shift click: move a stack between the hotbar and the main inventory, merging where it can.
    pub fn quick_move(&mut self, content: &Content, slot: usize) {
        let Some(stack) = self.slots[slot] else { return };
        let targets: Vec<usize> =
            if slot < HOTBAR { (HOTBAR..SLOTS).collect() } else { (0..HOTBAR).collect() };
        let max = content.item(stack.item).stack;
        let mut left = stack.count;
        for &t in &targets {
            if let Some(s) = &mut self.slots[t]
                && s.item == stack.item
                && s.count < max
            {
                let moved = left.min(max - s.count);
                s.count += moved;
                left -= moved;
            }
        }
        for &t in &targets {
            if left == 0 {
                break;
            }
            if self.slots[t].is_none() {
                self.slots[t] = Some(Stack { count: left, ..stack });
                left = 0;
            }
        }
        self.slots[slot] = (left > 0).then_some(Stack { count: left, ..stack });
    }

    /// Put a stack held on the cursor back (e.g. when the inventory closes). Returns what didn't fit.
    pub fn return_held(&mut self, content: &Content, held: &mut Option<Stack>) -> u32 {
        match held.take() {
            Some(h) => self.add(content, h.item, h.count),
            None => 0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn content() -> Content {
        Content::load().expect("content")
    }

    #[test]
    fn add_merges_into_stacks_then_fills_empty_slots() {
        let c = content();
        let rock = c.item_id("rock").unwrap();
        let mut inv = Inventory::default();
        assert_eq!(inv.add(&c, rock, 100), 0);
        assert_eq!(inv.slots[0], Some(Stack { item: rock, count: 64 }));
        assert_eq!(inv.slots[1], Some(Stack { item: rock, count: 36 }));
        assert_eq!(inv.add(&c, rock, 30), 0);
        assert_eq!(inv.slots[1].unwrap().count, 64);
        assert_eq!(inv.slots[2].unwrap().count, 2);
        assert_eq!(inv.count(rock), 130);
    }

    #[test]
    fn overflow_is_reported() {
        let c = content();
        let rock = c.item_id("rock").unwrap();
        let mut inv = Inventory::default();
        assert_eq!(inv.add(&c, rock, 64 * SLOTS as u32 + 5), 5);
        assert_eq!(inv.count(rock), 64 * SLOTS as u32);
    }

    #[test]
    fn take_fails_cleanly_and_empties_the_back_first() {
        let c = content();
        let (rock, soil) = (c.item_id("rock").unwrap(), c.item_id("soil").unwrap());
        let mut inv = Inventory::default();
        inv.add(&c, rock, 70);
        inv.add(&c, soil, 3);
        assert!(!inv.take(rock, 71));
        assert_eq!(inv.count(rock), 70);
        assert!(inv.take(rock, 10));
        assert_eq!(inv.slots[0].unwrap().count, 60, "hotbar stack is used last");
        assert_eq!(inv.slots[1], None);
        assert_eq!(inv.count(soil), 3);
    }

    #[test]
    fn volumes_keep_their_fractions() {
        let c = content();
        let rock = c.item_id("rock").unwrap();
        let mut inv = Inventory::default();
        for _ in 0..4 {
            inv.add_volume(&c, rock, 0.6);
        }
        assert_eq!(inv.count(rock), 2);
        assert!((inv.remainders[&rock] - 0.4).abs() < 1e-5);
        // 0.3 comes out of the remainder alone.
        assert!(inv.pay_volume(rock, 0.3));
        assert_eq!(inv.count(rock), 2);
        // 1.5 needs the remaining 0.1 plus two items, leaving 0.6.
        assert!(inv.pay_volume(rock, 1.5));
        assert_eq!(inv.count(rock), 0);
        assert!((inv.remainders[&rock] - 0.6).abs() < 1e-5);
        assert!(!inv.pay_volume(rock, 1.0));
        assert!((inv.remainders[&rock] - 0.6).abs() < 1e-5, "failed payment changes nothing");
    }

    #[test]
    fn cursor_clicks_pick_put_merge_swap_and_split() {
        let c = content();
        let (rock, soil) = (c.item_id("rock").unwrap(), c.item_id("soil").unwrap());
        let mut inv = Inventory::default();
        inv.slots[0] = Some(Stack { item: rock, count: 10 });
        inv.slots[1] = Some(Stack { item: rock, count: 60 });
        inv.slots[2] = Some(Stack { item: soil, count: 5 });
        let mut held = None;
        inv.click(&c, 0, &mut held); // pick up 10 rock
        assert_eq!((inv.slots[0], held), (None, Some(Stack { item: rock, count: 10 })));
        inv.click(&c, 1, &mut held); // merge: 60 + 4 = 64, 6 left on the cursor
        assert_eq!(inv.slots[1].unwrap().count, 64);
        assert_eq!(held.unwrap().count, 6);
        inv.click(&c, 2, &mut held); // swap with the soil
        assert_eq!(inv.slots[2], Some(Stack { item: rock, count: 6 }));
        assert_eq!(held, Some(Stack { item: soil, count: 5 }));
        inv.right_click(&c, 3, &mut held); // drop one soil into an empty slot
        assert_eq!(inv.slots[3], Some(Stack { item: soil, count: 1 }));
        assert_eq!(held.unwrap().count, 4);
        inv.click(&c, 3, &mut held); // merge the rest
        assert_eq!((inv.slots[3].unwrap().count, held), (5, None));
        inv.right_click(&c, 3, &mut held); // pick up half (rounded up)
        assert_eq!((inv.slots[3].unwrap().count, held.unwrap().count), (2, 3));
        assert_eq!(inv.return_held(&c, &mut held), 0);
        assert_eq!(inv.count(soil), 5);
    }

    #[test]
    fn quick_move_crosses_between_hotbar_and_main() {
        let c = content();
        let rock = c.item_id("rock").unwrap();
        let mut inv = Inventory::default();
        inv.slots[3] = Some(Stack { item: rock, count: 20 });
        inv.slots[20] = Some(Stack { item: rock, count: 50 });
        inv.quick_move(&c, 3); // tops up slot 20 to 64, the remaining 6 go to the first empty main slot
        assert_eq!(inv.slots[3], None);
        assert_eq!(inv.slots[20].unwrap().count, 64);
        assert_eq!(inv.slots[HOTBAR], Some(Stack { item: rock, count: 6 }));
        inv.quick_move(&c, 20);
        assert_eq!(inv.slots[0], Some(Stack { item: rock, count: 64 }));
    }
}
