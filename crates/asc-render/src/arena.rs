//! Shared vertex/index storage for all chunk meshes.
//!
//! Creating and destroying two GPU buffers per chunk while streaming caused frame spikes (the driver
//! stalls on the churn). Instead every mesh lives in one big vertex buffer and one big index buffer,
//! sub-allocated with a first-fit free list. Uploads are `write_buffer` into the allocated range, and
//! drawing binds the buffers once per pass. When an arena fills up it doubles, copying on the GPU.

use std::collections::BTreeMap;

/// First-fit allocator over `[0, capacity)` in abstract units, merging neighbours on free.
#[derive(Debug)]
pub struct RangeAlloc {
    capacity: u32,
    /// start -> length of each free range, non-overlapping and non-adjacent.
    free: BTreeMap<u32, u32>,
}

impl RangeAlloc {
    pub fn new(capacity: u32) -> Self {
        let mut free = BTreeMap::new();
        if capacity > 0 {
            free.insert(0, capacity);
        }
        Self { capacity, free }
    }

    pub fn capacity(&self) -> u32 {
        self.capacity
    }

    pub fn alloc(&mut self, len: u32) -> Option<u32> {
        if len == 0 {
            return Some(0);
        }
        let (&start, &free_len) = self.free.iter().find(|(_, l)| **l >= len)?;
        self.free.remove(&start);
        if free_len > len {
            self.free.insert(start + len, free_len - len);
        }
        Some(start)
    }

    pub fn free(&mut self, start: u32, len: u32) {
        if len == 0 {
            return;
        }
        let (mut start, mut len) = (start, len);
        if let Some((&prev, &prev_len)) = self.free.range(..start).next_back()
            && prev + prev_len == start
        {
            self.free.remove(&prev);
            start = prev;
            len += prev_len;
        }
        if let Some(&next_len) = self.free.get(&(start + len)) {
            self.free.remove(&(start + len));
            len += next_len;
        }
        self.free.insert(start, len);
    }

    /// Extend the managed range to `capacity` (the new space is free).
    pub fn grow(&mut self, capacity: u32) {
        assert!(capacity >= self.capacity);
        let added = capacity - self.capacity;
        let old = self.capacity;
        self.capacity = capacity;
        self.free(old, added);
    }
}

/// A GPU buffer managed by a [`RangeAlloc`], in elements of `stride` bytes.
pub struct Arena {
    pub buffer: wgpu::Buffer,
    alloc: RangeAlloc,
    stride: u64,
    usage: wgpu::BufferUsages,
    label: &'static str,
}

impl Arena {
    pub fn new(
        device: &wgpu::Device,
        label: &'static str,
        usage: wgpu::BufferUsages,
        stride: u64,
        elements: u32,
    ) -> Self {
        let usage = usage | wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC;
        let buffer = Self::create(device, label, usage, stride * u64::from(elements));
        Self { buffer, alloc: RangeAlloc::new(elements), stride, usage, label }
    }

    fn create(device: &wgpu::Device, label: &str, usage: wgpu::BufferUsages, size: u64) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Store `data` and return its first element index, growing the buffer if needed.
    pub fn upload(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, data: &[u8]) -> u32 {
        let elements = (data.len() as u64 / self.stride) as u32;
        let start = match self.alloc.alloc(elements) {
            Some(start) => start,
            None => {
                self.grow(device, queue, elements);
                self.alloc.alloc(elements).expect("arena grew to fit")
            }
        };
        queue.write_buffer(&self.buffer, u64::from(start) * self.stride, data);
        start
    }

    pub fn free(&mut self, start: u32, elements: u32) {
        self.alloc.free(start, elements);
    }

    fn grow(&mut self, device: &wgpu::Device, queue: &wgpu::Queue, at_least: u32) {
        let old = self.alloc.capacity();
        let max_elements = (device.limits().max_buffer_size / self.stride).min(u64::from(u32::MAX)) as u32;
        let new = old.saturating_mul(2).max(old + at_least).min(max_elements);
        assert!(new >= old + at_least, "{} arena is at the device's buffer size limit", self.label);
        log::info!("{} arena grows to {:.0} MB", self.label, (u64::from(new) * self.stride) as f64 / 1e6);
        let buffer = Self::create(device, self.label, self.usage, u64::from(new) * self.stride);
        // Submit the copy right away. Writes queued earlier (into the old buffer) run before it, and
        // writes queued after this (into the new one) run before the next submit, i.e. after it.
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("arena grow") });
        encoder.copy_buffer_to_buffer(&self.buffer, 0, &buffer, 0, u64::from(old) * self.stride);
        queue.submit([encoder.finish()]);
        self.buffer = buffer;
        self.alloc.grow(new);
    }
}

#[cfg(test)]
mod tests {
    use super::RangeAlloc;

    #[test]
    fn allocates_frees_and_merges() {
        let mut a = RangeAlloc::new(100);
        let x = a.alloc(30).unwrap();
        let y = a.alloc(30).unwrap();
        let z = a.alloc(30).unwrap();
        assert_eq!((x, y, z), (0, 30, 60));
        assert_eq!(a.alloc(20), None);
        a.free(y, 30);
        assert_eq!(a.alloc(20), Some(30));
        a.free(30, 20);
        a.free(x, 30);
        a.free(z, 30);
        // Everything merged back into a single range.
        assert_eq!(a.alloc(100), Some(0));
    }

    #[test]
    fn grow_extends_the_last_free_range() {
        let mut a = RangeAlloc::new(10);
        assert_eq!(a.alloc(8), Some(0));
        a.grow(20);
        assert_eq!(a.alloc(12), Some(8));
    }
}
