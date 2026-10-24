//! Skinned models (creatures, characters): drawn like parts, but bent by their skeletons.
//!
//! The game hands over this frame's instances, each with its joint palette (see `skeleton`).
//! Every palette goes into one storage buffer; each instance knows where its own starts, and
//! where last frame's did (kept here by the instance's key), so the vertex shader can place each
//! vertex where it was last frame too: TAA needs that motion, or moving limbs smear.

use std::collections::HashMap;
use std::sync::Arc;

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::camera::Frustum;
use crate::models::{SkinnedModelData, SkinnedVertex};
use crate::parts::Parts;
use crate::targets::CASCADES;

/// Skinned models aren't drawn beyond this (they're small; fog has them by then).
const DRAW_DISTANCE: f32 = 220.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SkinnedId(pub u16);

#[derive(Clone, Debug)]
pub struct SkinnedInstance {
    pub model: SkinnedId,
    /// The same from frame to frame for the same thing (its motion comes from last frame's).
    pub key: u64,
    pub position: Vec3,
    /// Rotation about +Y, as for parts.
    pub yaw: f32,
    pub scale: f32,
    /// Joint matrices (`Skeleton::palette`); attachments share their body's.
    pub palette: Arc<[Mat4]>,
    /// Cast a shadow but don't draw (the player's own body seen in first person).
    pub shadow_only: bool,
    /// Cast a shadow at all (not a tool held in front of the eyes).
    pub casts_shadow: bool,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    /// xyz = camera-relative origin, w = yaw.
    origin: [f32; 4],
    /// x = scale, w = fade.
    params: [f32; 4],
    /// Last frame's origin (relative to this frame's camera) and yaw.
    prev_origin: [f32; 4],
    /// The first joint of this frame's palette and of last frame's.
    palette: [u32; 2],
}

struct GpuPrimitive {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    material: wgpu::BindGroup,
}

struct GpuModel {
    primitives: Vec<GpuPrimitive>,
    radius: f32,
}

#[derive(Clone, Copy, Default)]
struct Range {
    start: u32,
    count: u32,
}

pub struct Skinned {
    models: Vec<GpuModel>,
    instances: Vec<SkinnedInstance>,
    /// Last frame's palette, position and yaw, by key.
    prev: HashMap<u64, (Arc<[Mat4]>, Vec3, f32)>,
    joints: wgpu::Buffer,
    joint_capacity: usize,
    bind_group: wgpu::BindGroup,
    instance_buffer: wgpu::Buffer,
    instance_capacity: usize,
    visible: Vec<Range>,
    shadow: [Vec<Range>; CASCADES],
    pub drawn: usize,
}

fn joint_buffer(device: &wgpu::Device, matrices: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("skinned joints"),
        size: (matrices.max(1) * std::mem::size_of::<Mat4>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn instance_buffer(device: &wgpu::Device, count: usize) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("skinned instances"),
        size: (count.max(1) * std::mem::size_of::<InstanceData>()) as u64,
        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    })
}

fn bind(device: &wgpu::Device, layout: &wgpu::BindGroupLayout, joints: &wgpu::Buffer) -> wgpu::BindGroup {
    device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("skinned joints"),
        layout,
        entries: &[wgpu::BindGroupEntry { binding: 0, resource: joints.as_entire_binding() }],
    })
}

impl Skinned {
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let joint_capacity = 4096;
        let joints = joint_buffer(device, joint_capacity);
        let bind_group = bind(device, layout, &joints);
        Self {
            models: Vec::new(),
            instances: Vec::new(),
            prev: HashMap::new(),
            joints,
            joint_capacity,
            bind_group,
            instance_buffer: instance_buffer(device, 256),
            instance_capacity: 256,
            visible: Vec::new(),
            shadow: Default::default(),
            drawn: 0,
        }
    }

    /// Upload a skinned model (its materials go through the parts' shared texture array).
    pub fn add_model(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        material_layout: &wgpu::BindGroupLayout,
        parts: &mut Parts,
        model: &SkinnedModelData,
    ) -> SkinnedId {
        let primitives = model
            .primitives
            .iter()
            .map(|p| GpuPrimitive {
                vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("skinned vertices"),
                    contents: bytemuck::cast_slice(&p.vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
                indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("skinned indices"),
                    contents: bytemuck::cast_slice(&p.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
                index_count: p.indices.len() as u32,
                material: parts.material(device, queue, material_layout, &p.material),
            })
            .collect();
        // Room for limbs flung out of the rest pose's bounds.
        let radius = model.min.length().max(model.max.length()) * 1.5;
        self.models.push(GpuModel { primitives, radius });
        SkinnedId((self.models.len() - 1) as u16)
    }

    pub fn set_instances(&mut self, instances: Vec<SkinnedInstance>) {
        self.instances = instances;
    }

    #[allow(clippy::too_many_arguments)]
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        camera: Vec3,
        view_proj: Mat4,
        cascades: [Option<(Mat4, f32)>; CASCADES],
    ) {
        let frustum = Frustum::from_view_proj(view_proj);
        let n = self.models.len();
        let mut buckets: Vec<Vec<InstanceData>> = vec![Vec::new(); n];
        let mut shadow_buckets: [Vec<Vec<InstanceData>>; CASCADES] = std::array::from_fn(|_| vec![Vec::new(); n]);
        let covers = |c: usize, rel: Vec3, radius: f32| {
            cascades[c].is_some_and(|(m, half)| {
                let p = m.transform_point3(rel);
                let r = radius / half;
                p.x.abs() <= 1.0 + r && p.y.abs() <= 1.0 + r
            })
        };
        let mut joints: Vec<Mat4> = Vec::new();
        // Palettes are shared (a body and its hair): upload each once.
        let mut placed: HashMap<*const Mat4, u32> = HashMap::new();
        let mut upload = |p: &Arc<[Mat4]>, joints: &mut Vec<Mat4>| -> u32 {
            *placed.entry(p.as_ptr()).or_insert_with(|| {
                let base = joints.len() as u32;
                joints.extend_from_slice(p);
                base
            })
        };
        let mut next_prev = HashMap::with_capacity(self.instances.len());
        for i in &self.instances {
            let m = i.model.0 as usize;
            let radius = self.models[m].radius * i.scale;
            let rel = i.position - camera;
            let in_view = !i.shadow_only && rel.length() < DRAW_DISTANCE + radius && frustum.intersects_box(rel - radius, rel + radius);
            let casts: Vec<usize> =
                (0..CASCADES).filter(|&c| i.casts_shadow && rel.length() < DRAW_DISTANCE && covers(c, rel, radius)).collect();
            next_prev.insert(i.key, (i.palette.clone(), i.position, i.yaw));
            if !in_view && casts.is_empty() {
                continue;
            }
            let base = upload(&i.palette, &mut joints);
            let (prev_palette, prev_position, prev_yaw) = match self.prev.get(&i.key) {
                Some((p, pos, yaw)) if p.len() == i.palette.len() => (p.clone(), *pos, *yaw),
                _ => (i.palette.clone(), i.position, i.yaw),
            };
            let prev_base = upload(&prev_palette, &mut joints);
            let prev_rel = prev_position - camera;
            let data = InstanceData {
                origin: [rel.x, rel.y, rel.z, i.yaw],
                params: [i.scale, 0.0, 0.0, 1.0],
                prev_origin: [prev_rel.x, prev_rel.y, prev_rel.z, prev_yaw],
                palette: [base, prev_base],
            };
            if in_view {
                buckets[m].push(data);
            }
            for c in casts {
                shadow_buckets[c][m].push(data);
            }
        }
        self.prev = next_prev;
        let mut scratch: Vec<InstanceData> = Vec::new();
        let flatten = |buckets: &[Vec<InstanceData>], scratch: &mut Vec<InstanceData>| -> Vec<Range> {
            buckets
                .iter()
                .map(|b| {
                    let r = Range { start: scratch.len() as u32, count: b.len() as u32 };
                    scratch.extend_from_slice(b);
                    r
                })
                .collect()
        };
        self.visible = flatten(&buckets, &mut scratch);
        for (shadow, buckets) in self.shadow.iter_mut().zip(&shadow_buckets) {
            *shadow = flatten(buckets, &mut scratch);
        }
        self.drawn = self.visible.iter().map(|r| r.count as usize).sum();
        if joints.len() > self.joint_capacity {
            self.joint_capacity = joints.len().next_power_of_two();
            self.joints = joint_buffer(device, self.joint_capacity);
            self.bind_group = bind(device, layout, &self.joints);
        }
        if !joints.is_empty() {
            queue.write_buffer(&self.joints, 0, bytemuck::cast_slice(&joints));
        }
        if scratch.len() > self.instance_capacity {
            self.instance_capacity = scratch.len().next_power_of_two();
            self.instance_buffer = instance_buffer(device, self.instance_capacity);
        }
        if !scratch.is_empty() {
            queue.write_buffer(&self.instance_buffer, 0, bytemuck::cast_slice(&scratch));
        }
    }

    fn draw_ranges(&self, pass: &mut wgpu::RenderPass<'_>, ranges: &[Range]) {
        if ranges.iter().all(|r| r.count == 0) {
            return;
        }
        pass.set_bind_group(2, &self.bind_group, &[]);
        pass.set_vertex_buffer(1, self.instance_buffer.slice(..));
        for (model, range) in self.models.iter().zip(ranges) {
            if range.count == 0 {
                continue;
            }
            for prim in &model.primitives {
                pass.set_bind_group(1, &prim.material, &[]);
                pass.set_vertex_buffer(0, prim.vertices.slice(..));
                pass.set_index_buffer(prim.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.index_count, 0, range.start..range.start + range.count);
            }
        }
    }

    /// In the scene pass (pipeline and group 0 already set).
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.draw_ranges(pass, &self.visible);
    }

    /// Shadow casters for one cascade (pipeline and group 0 already set).
    pub fn draw_shadows(&self, pass: &mut wgpu::RenderPass<'_>, cascade: usize) {
        self.draw_ranges(pass, &self.shadow[cascade]);
    }
}

pub const SKINNED_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
    0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Unorm8x4, 6 => Uint16x4, 7 => Float32x4
];
pub const SKINNED_INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![4 => Float32x4, 5 => Float32x4, 8 => Float32x4, 9 => Uint32x2];

pub fn vertex_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 2] {
    [
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<SkinnedVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &SKINNED_VERTEX_ATTRIBUTES,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceData>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &SKINNED_INSTANCE_ATTRIBUTES,
        }),
    ]
}
