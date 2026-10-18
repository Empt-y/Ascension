//! Instanced drawing of building parts (static models placed in the world).
//!
//! The game hands over the full list of placed parts whenever it changes; each frame the renderer
//! culls them (distance and frustum), groups them by model and writes one instance buffer laid out
//! as [visible…, shadow casters…, ghost]. Each model primitive is then one instanced draw.

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::camera::Frustum;
use crate::models::{ModelData, PartVertex};

/// Parts beyond this distance aren't drawn (fog has hidden them by then).
const DRAW_DISTANCE: f32 = 600.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelId(pub u16);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartInstance {
    pub model: ModelId,
    /// World position of the model's origin.
    pub position: Vec3,
    /// Quarter turns about +Y (0..3).
    pub rotation: u8,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ghost {
    pub instance: PartInstance,
    pub valid: bool,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniform {
    base: [f32; 4],
    extra: [f32; 4],
}

struct GpuPrimitive {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    material: wgpu::BindGroup,
}

struct GpuModel {
    primitives: Vec<GpuPrimitive>,
    /// Radius of a sphere around the origin containing the model (any rotation).
    radius: f32,
}

#[derive(Clone, Copy, Default)]
struct Range {
    start: u32,
    count: u32,
}

pub struct Parts {
    models: Vec<GpuModel>,
    instances: Vec<PartInstance>,
    ghost: Option<Ghost>,
    buffer: wgpu::Buffer,
    capacity: usize,
    scratch: Vec<[f32; 4]>,
    visible: Vec<Range>,
    shadow: Vec<Range>,
    ghost_slot: Option<(ModelId, u32)>,
    pub drawn: usize,
}

fn instance_data(i: &PartInstance, camera: Vec3, ghost: u32) -> [f32; 4] {
    let rel = i.position - camera;
    [rel.x, rel.y, rel.z, (u32::from(i.rotation % 4) + 4 * ghost) as f32]
}

impl Parts {
    pub fn new(device: &wgpu::Device) -> Self {
        let capacity = 1024;
        Self {
            models: Vec::new(),
            instances: Vec::new(),
            ghost: None,
            buffer: Self::buffer(device, capacity),
            capacity,
            scratch: Vec::new(),
            visible: Vec::new(),
            shadow: Vec::new(),
            ghost_slot: None,
            drawn: 0,
        }
    }

    fn buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("part instances"),
            size: (capacity * 16) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    pub fn add_model(
        &mut self,
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        model: &ModelData,
    ) -> ModelId {
        let primitives = model
            .primitives
            .iter()
            .map(|p| {
                let uniform = MaterialUniform {
                    base: [p.base_color[0], p.base_color[1], p.base_color[2], p.roughness.clamp(0.05, 1.0)],
                    extra: [p.metallic, 0.0, 0.0, 0.0],
                };
                let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("part material"),
                    contents: bytemuck::bytes_of(&uniform),
                    usage: wgpu::BufferUsages::UNIFORM,
                });
                GpuPrimitive {
                    vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("part vertices"),
                        contents: bytemuck::cast_slice(&p.vertices),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("part indices"),
                        contents: bytemuck::cast_slice(&p.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    }),
                    index_count: p.indices.len() as u32,
                    material: device.create_bind_group(&wgpu::BindGroupDescriptor {
                        label: Some("part material"),
                        layout,
                        entries: &[wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() }],
                    }),
                }
            })
            .collect();
        let radius = model.min.length().max(model.max.length());
        self.models.push(GpuModel { primitives, radius });
        ModelId((self.models.len() - 1) as u16)
    }

    pub fn set_instances(&mut self, instances: &[PartInstance]) {
        self.instances.clear();
        self.instances.extend(instances.iter().filter(|i| (i.model.0 as usize) < self.models.len()));
    }

    pub fn set_ghost(&mut self, ghost: Option<Ghost>) {
        self.ghost = ghost.filter(|g| (g.instance.model.0 as usize) < self.models.len());
    }

    pub fn count(&self) -> usize {
        self.instances.len()
    }

    /// Cull, sort and upload this frame's instances.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: Vec3,
        view_proj: Mat4,
        shadow_reach: f32,
    ) {
        let frustum = Frustum::from_view_proj(view_proj);
        let n = self.models.len();
        self.visible = vec![Range::default(); n];
        self.shadow = vec![Range::default(); n];
        self.scratch.clear();

        let fill = |ranges: &mut Vec<Range>,
                    scratch: &mut Vec<[f32; 4]>,
                    keep: &dyn Fn(&PartInstance, f32) -> bool| {
            let mut list: Vec<(ModelId, [f32; 4])> = Vec::new();
            for i in &self.instances {
                let radius = self.models[i.model.0 as usize].radius;
                if keep(i, radius) {
                    list.push((i.model, instance_data(i, camera, 0)));
                }
            }
            list.sort_unstable_by_key(|(m, _)| *m);
            for (model, data) in list {
                let r = &mut ranges[model.0 as usize];
                if r.count == 0 {
                    r.start = scratch.len() as u32;
                }
                r.count += 1;
                scratch.push(data);
            }
        };
        let visible = |i: &PartInstance, radius: f32| {
            let rel = i.position - camera;
            rel.length() < DRAW_DISTANCE + radius && frustum.intersects_box(rel - radius, rel + radius)
        };
        let casts = |i: &PartInstance, radius: f32| (i.position - camera).length() < shadow_reach + radius;
        fill(&mut self.visible, &mut self.scratch, &visible);
        fill(&mut self.shadow, &mut self.scratch, &casts);
        self.drawn = self.visible.iter().map(|r| r.count as usize).sum();
        self.ghost_slot = self.ghost.map(|g| {
            self.scratch.push(instance_data(&g.instance, camera, if g.valid { 1 } else { 2 }));
            (g.instance.model, self.scratch.len() as u32 - 1)
        });

        if self.scratch.len() > self.capacity {
            self.capacity = self.scratch.len().next_power_of_two();
            self.buffer = Self::buffer(device, self.capacity);
        }
        if !self.scratch.is_empty() {
            queue.write_buffer(&self.buffer, 0, bytemuck::cast_slice(&self.scratch));
        }
    }

    fn draw_ranges(&self, pass: &mut wgpu::RenderPass<'_>, ranges: &[Range], with_material: bool) {
        pass.set_vertex_buffer(1, self.buffer.slice(..));
        for (model, range) in self.models.iter().zip(ranges) {
            if range.count == 0 {
                continue;
            }
            for prim in &model.primitives {
                if with_material {
                    pass.set_bind_group(1, &prim.material, &[]);
                }
                pass.set_vertex_buffer(0, prim.vertices.slice(..));
                pass.set_index_buffer(prim.indices.slice(..), wgpu::IndexFormat::Uint32);
                pass.draw_indexed(0..prim.index_count, 0, range.start..range.start + range.count);
            }
        }
    }

    /// Opaque parts in the scene pass (pipeline and group 0 already set).
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.draw_ranges(pass, &self.visible, true);
    }

    /// Shadow casters in the shadow pass (pipeline and group 0 already set).
    pub fn draw_shadows(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.draw_ranges(pass, &self.shadow, false);
    }

    /// The placement ghost (ghost pipeline and group 0 already set).
    pub fn draw_ghost(&self, pass: &mut wgpu::RenderPass<'_>) {
        let Some((model, slot)) = self.ghost_slot else { return };
        pass.set_vertex_buffer(1, self.buffer.slice(..));
        for prim in &self.models[model.0 as usize].primitives {
            pass.set_bind_group(1, &prim.material, &[]);
            pass.set_vertex_buffer(0, prim.vertices.slice(..));
            pass.set_index_buffer(prim.indices.slice(..), wgpu::IndexFormat::Uint32);
            pass.draw_indexed(0..prim.index_count, 0, slot..slot + 1);
        }
    }
}

pub const PART_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
pub const PART_INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 1] = wgpu::vertex_attr_array![2 => Float32x4];

pub fn vertex_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 2] {
    [
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PartVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &PART_VERTEX_ATTRIBUTES,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: 16,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &PART_INSTANCE_ATTRIBUTES,
        }),
    ]
}
