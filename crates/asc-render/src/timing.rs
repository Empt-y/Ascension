//! GPU timings per stage of the frame, from timestamp queries.
//!
//! `mark` writes a timestamp after each stage, between passes or inside one; a stage's time is the
//! gap since the previous mark. The results come back a few frames later through a small ring of
//! readback buffers, and are averaged. Needs timestamps inside encoders and passes; without them
//! (or with `ASC_GPU_TIMING=0`) every call is a no-op.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

const MAX_MARKS: u32 = 24;
const RING: usize = 3;
/// Readback buffer states.
const FREE: u8 = 0;
const PENDING: u8 = 1;
const READY: u8 = 2;

struct Readback {
    buffer: wgpu::Buffer,
    state: Arc<AtomicU8>,
    labels: Vec<&'static str>,
}

pub struct GpuTimer {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    ring: Vec<Readback>,
    /// Nanoseconds per timestamp tick.
    period: f32,
    labels: Vec<&'static str>,
    /// The ring slot this frame copied into, to map after submitting.
    submitted: Option<usize>,
    /// Smoothed milliseconds per stage, in frame order.
    averages: Vec<(&'static str, f32)>,
}

impl GpuTimer {
    pub fn features() -> wgpu::Features {
        wgpu::Features::TIMESTAMP_QUERY
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS
            | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES
    }

    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(Self::features()) || std::env::var("ASC_GPU_TIMING").is_ok_and(|v| v == "0")
        {
            return None;
        }
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("gpu timer"),
            ty: wgpu::QueryType::Timestamp,
            count: MAX_MARKS,
        });
        let size = MAX_MARKS as u64 * 8;
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu timer resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let ring = (0..RING)
            .map(|_| Readback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("gpu timer readback"),
                    size,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }),
                state: Arc::new(AtomicU8::new(FREE)),
                labels: Vec::new(),
            })
            .collect();
        Some(Self {
            set,
            resolve,
            ring,
            period: queue.get_timestamp_period(),
            labels: Vec::new(),
            submitted: None,
            averages: Vec::new(),
        })
    }

    /// Start a frame: collect any results that have come back.
    pub fn begin(&mut self, device: &wgpu::Device) {
        self.labels.clear();
        let _ = device.poll(wgpu::PollType::Poll);
        for slot in &mut self.ring {
            if slot.state.load(Ordering::Acquire) != READY {
                continue;
            }
            if let Ok(data) = slot.buffer.slice(..).get_mapped_range() {
                let ticks: &[u64] = bytemuck::cast_slice(&data);
                let rebuilt = self.averages.len() != slot.labels.len()
                    || self.averages.iter().zip(&slot.labels).any(|(a, l)| a.0 != *l);
                if rebuilt {
                    self.averages = slot.labels.iter().map(|l| (*l, 0.0)).collect();
                }
                for i in 1..slot.labels.len() {
                    let ms = ticks[i].saturating_sub(ticks[i - 1]) as f32 * self.period * 1e-6;
                    let a = &mut self.averages[i].1;
                    *a = if rebuilt { ms } else { *a + (ms - *a) * 0.05 };
                }
            }
            slot.buffer.unmap();
            slot.state.store(FREE, Ordering::Release);
        }
    }

    fn next(&mut self, label: &'static str) -> Option<u32> {
        let i = self.labels.len() as u32;
        (i < MAX_MARKS).then(|| {
            self.labels.push(label);
            i
        })
    }

    /// Mark the end of stage `label`, between passes.
    pub fn mark(&mut self, encoder: &mut wgpu::CommandEncoder, label: &'static str) {
        if let Some(i) = self.next(label) {
            encoder.write_timestamp(&self.set, i);
        }
    }

    /// Mark the end of stage `label`, inside a pass.
    pub fn mark_in(&mut self, pass: &mut wgpu::RenderPass<'_>, label: &'static str) {
        if let Some(i) = self.next(label) {
            pass.write_timestamp(&self.set, i);
        }
    }

    /// Resolve this frame's marks into a free readback buffer (if there is one).
    pub fn finish(&mut self, encoder: &mut wgpu::CommandEncoder) {
        self.submitted = None;
        let n = self.labels.len() as u32;
        if n < 2 {
            return;
        }
        let Some(index) = self.ring.iter().position(|s| s.state.load(Ordering::Acquire) == FREE) else {
            return;
        };
        encoder.resolve_query_set(&self.set, 0..n, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(&self.resolve, 0, &self.ring[index].buffer, 0, n as u64 * 8);
        self.ring[index].labels = self.labels.clone();
        self.ring[index].state.store(PENDING, Ordering::Release);
        self.submitted = Some(index);
    }

    /// After submitting: start reading back this frame's results.
    pub fn after_submit(&mut self) {
        let Some(index) = self.submitted.take() else { return };
        let state = self.ring[index].state.clone();
        self.ring[index].buffer.slice(..).map_async(wgpu::MapMode::Read, move |r| {
            state.store(if r.is_ok() { READY } else { FREE }, Ordering::Release);
        });
    }

    /// Smoothed milliseconds per stage (the first mark only starts the clock).
    pub fn stages(&self) -> impl Iterator<Item = (&'static str, f32)> + '_ {
        self.averages.iter().skip(1).copied()
    }
}
