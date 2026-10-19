//! Instanced drawing of static models placed in the world: building parts and nature scatter.
//!
//! The game hands over the full list of instances whenever it changes; each frame the renderer
//! culls them (distance and frustum), groups them by model and writes one instance buffer laid out
//! as [visible…, shadow casters…, ghost]. Each model primitive is then one instanced draw.
//!
//! Nature scatter arrives separately as batches (one per terrain column and kind), culled as a
//! whole by their bounds, with each instance fading out over the last stretch of its batch's draw
//! distance.
//!
//! Textured primitives (leaf cards) share one mipmapped texture array, one layer per distinct
//! texture name, so every model can use the same bind group layout.

use std::collections::HashMap;

use glam::{Mat4, Vec3};
use wgpu::util::DeviceExt;

use crate::camera::Frustum;
use crate::models::{ModelData, PartVertex, TextureData};
use crate::targets::CASCADES;

/// Instances beyond this distance aren't drawn (fog has hidden them by then).
const DRAW_DISTANCE: f32 = 600.0;
/// Every texture is resampled to this size to fit the array.
pub const TEXTURE_SIZE: u32 = 512;
const TEXTURE_LAYERS: u32 = 48;
const TEXTURE_MIPS: u32 = TEXTURE_SIZE.ilog2() + 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ModelId(pub u16);

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PartInstance {
    pub model: ModelId,
    /// World position of the model's origin.
    pub position: Vec3,
    /// Rotation about +Y in radians (anticlockwise seen from above, +X towards -Z).
    pub yaw: f32,
    /// Uniform scale.
    pub scale: f32,
    /// How much the model bends and flutters in the wind (0 for anything built).
    pub sway: f32,
}

impl PartInstance {
    /// A rigid, unscaled instance turned by whole quarter turns (building parts).
    pub fn rigid(model: ModelId, position: Vec3, quarter_turns: u8) -> Self {
        let yaw = f32::from(quarter_turns % 4) * std::f32::consts::FRAC_PI_2;
        Self { model, position, yaw, scale: 1.0, sway: 0.0 }
    }
}

/// Distant instances of a model switch to a cheaper one (an impostor) around this distance,
/// cross-fading over `LOD_BAND` metres.
const LOD_BAND: f32 = 12.0;

/// A group of scatter instances (one terrain column of one kind) drawn and culled together.
#[derive(Clone, Debug, PartialEq)]
pub struct ScatterBatch {
    pub instances: Vec<PartInstance>,
    /// Instances are drawn out to this distance...
    pub distance: f32,
    /// ...fading out over this last stretch of it.
    pub fade: f32,
    /// Instances cast into the shadow cascades no wider than this (half-width, metres; 0 for none).
    pub shadow_distance: f32,
}

struct StoredBatch {
    batch: ScatterBatch,
    min: Vec3,
    max: Vec3,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ghost {
    pub instance: PartInstance,
    pub valid: bool,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniform {
    /// rgb = base colour, a = roughness.
    base: [f32; 4],
    /// x = metallic, y = texture layer (-1 for none), z = alpha cutoff, w = 1 for foliage.
    extra: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct InstanceData {
    /// xyz = camera-relative origin, w = yaw.
    origin: [f32; 4],
    /// x = scale, y = ghost state (0 none, 1 valid, 2 invalid), z = sway, w = fade: 1 is solid,
    /// f in 0..1 keeps the pixels whose dither value is below f, and -t keeps the complementary
    /// ones (so a model fading out and its replacement fading in never overlap or leave holes).
    params: [f32; 4],
}

struct GpuPrimitive {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    material: wgpu::BindGroup,
}

struct GpuModel {
    primitives: Vec<GpuPrimitive>,
    /// Radius of a sphere around the origin containing the model (any rotation, unit scale).
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
    scratch: Vec<InstanceData>,
    scatter: HashMap<u64, StoredBatch>,
    /// Per model: the far model it turns into, and at what distance.
    lods: Vec<Option<(ModelId, f32)>>,
    /// Per-model buckets reused each frame while sorting instances.
    buckets: Vec<Vec<InstanceData>>,
    shadow_buckets: [Vec<Vec<InstanceData>>; CASCADES],
    visible: Vec<Range>,
    shadow: [Vec<Range>; CASCADES],
    ghost_slot: Option<(ModelId, u32)>,
    textures: wgpu::Texture,
    texture_view: wgpu::TextureView,
    sampler: wgpu::Sampler,
    layers: HashMap<String, u32>,
    pub drawn: usize,
}

fn instance_data(i: &PartInstance, camera: Vec3, ghost: u32, fade: f32) -> InstanceData {
    let rel = i.position - camera;
    InstanceData { origin: [rel.x, rel.y, rel.z, i.yaw], params: [i.scale, ghost as f32, i.sway, fade] }
}

impl Parts {
    pub fn new(device: &wgpu::Device) -> Self {
        let capacity = 1024;
        let textures = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("model textures"),
            size: wgpu::Extent3d {
                width: TEXTURE_SIZE,
                height: TEXTURE_SIZE,
                depth_or_array_layers: TEXTURE_LAYERS,
            },
            mip_level_count: TEXTURE_MIPS,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let texture_view = textures.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("model textures"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            anisotropy_clamp: 4,
            ..Default::default()
        });
        Self {
            models: Vec::new(),
            instances: Vec::new(),
            ghost: None,
            buffer: Self::buffer(device, capacity),
            capacity,
            scratch: Vec::new(),
            scatter: HashMap::new(),
            lods: Vec::new(),
            buckets: Vec::new(),
            shadow_buckets: Default::default(),
            visible: Vec::new(),
            shadow: Default::default(),
            ghost_slot: None,
            textures,
            texture_view,
            sampler,
            layers: HashMap::new(),
            drawn: 0,
        }
    }

    fn buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("part instances"),
            size: (capacity * std::mem::size_of::<InstanceData>()) as u64,
            usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// The array layer holding `texture`, uploading it (with its mip chain) the first time.
    fn texture_layer(&mut self, queue: &wgpu::Queue, texture: &TextureData, cutoff: f32) -> Option<u32> {
        if let Some(&layer) = self.layers.get(&texture.name) {
            return Some(layer);
        }
        let layer = self.layers.len() as u32;
        if layer >= TEXTURE_LAYERS {
            log::warn!("no room for texture {} (limit {TEXTURE_LAYERS})", texture.name);
            return None;
        }
        for (level, (size, pixels)) in mip_chain(texture, cutoff).into_iter().enumerate() {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.textures,
                    mip_level: level as u32,
                    origin: wgpu::Origin3d { x: 0, y: 0, z: layer },
                    aspect: wgpu::TextureAspect::All,
                },
                &pixels,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size * 4),
                    rows_per_image: Some(size),
                },
                wgpu::Extent3d { width: size, height: size, depth_or_array_layers: 1 },
            );
        }
        self.layers.insert(texture.name.clone(), layer);
        Some(layer)
    }

    pub fn add_model(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        layout: &wgpu::BindGroupLayout,
        model: &ModelData,
    ) -> ModelId {
        let mut primitives = Vec::new();
        for p in &model.primitives {
            let cutoff = p.alpha_cutoff.unwrap_or(0.5);
            let layer = p.texture.as_ref().and_then(|t| self.texture_layer(queue, t, cutoff));
            // Cut-out textured primitives are leaf cards: shaded as foliage.
            let foliage = layer.is_some() && p.alpha_cutoff.is_some();
            let uniform = MaterialUniform {
                base: [p.base_color[0], p.base_color[1], p.base_color[2], p.roughness.clamp(0.05, 1.0)],
                extra: [
                    p.metallic,
                    layer.map_or(-1.0, |l| l as f32),
                    if foliage { cutoff } else { 0.0 },
                    f32::from(u8::from(foliage)),
                ],
            };
            let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("part material"),
                contents: bytemuck::bytes_of(&uniform),
                usage: wgpu::BufferUsages::UNIFORM,
            });
            primitives.push(GpuPrimitive {
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
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: buffer.as_entire_binding() },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&self.texture_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(&self.sampler),
                        },
                    ],
                }),
            });
        }
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

    /// Add, replace (`Some`) or remove (`None`) the scatter batch under `key`. Returns the bounds the
    /// change touched and how far its instances cast shadows, if it touched anything.
    pub fn set_scatter(&mut self, key: u64, batch: Option<ScatterBatch>) -> Option<(Vec3, Vec3, f32)> {
        let old = self.scatter.get(&key).map(|b| (b.min, b.max, b.batch.shadow_distance));
        match batch {
            Some(mut batch) => {
                batch.instances.retain(|i| (i.model.0 as usize) < self.models.len());
                let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
                for i in &batch.instances {
                    let r = self.models[i.model.0 as usize].radius * i.scale;
                    min = min.min(i.position - r);
                    max = max.max(i.position + r);
                }
                if batch.instances.is_empty() {
                    self.scatter.remove(&key);
                } else {
                    self.scatter.insert(key, StoredBatch { batch, min, max });
                }
            }
            None => {
                self.scatter.remove(&key);
            }
        }
        let new = self.scatter.get(&key).map(|b| (b.min, b.max, b.batch.shadow_distance));
        match (old, new) {
            (Some(a), Some(b)) => Some((a.0.min(b.0), a.1.max(b.1), a.2.max(b.2))),
            (a, b) => a.or(b),
        }
    }

    /// Draw `near` as `far` beyond `distance` (scatter only).
    pub fn set_lod(&mut self, near: ModelId, far: ModelId, distance: f32) {
        let i = near.0 as usize;
        if self.lods.len() <= i {
            self.lods.resize(i + 1, None);
        }
        self.lods[i] = Some((far, distance));
    }

    /// Scatter instances held (not just drawn).
    pub fn scatter_count(&self) -> usize {
        self.scatter.values().map(|b| b.batch.instances.len()).sum()
    }

    /// Cull, sort and upload this frame's instances.
    pub fn prepare(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        camera: Vec3,
        view_proj: Mat4,
        cascades: [Option<(Mat4, f32)>; CASCADES],
    ) {
        let frustum = Frustum::from_view_proj(view_proj);
        let n = self.models.len();
        self.buckets.resize_with(n, Vec::new);
        for b in &mut self.buckets {
            b.clear();
        }
        for cascade in &mut self.shadow_buckets {
            cascade.resize_with(n, Vec::new);
            for b in cascade.iter_mut() {
                b.clear();
            }
        }
        // A caster goes into a cascade if it lands anywhere in the area that cascade covers, never
        // by distance from the camera: a distance cut-off moves with the camera, so shadows popped
        // in and out as you walked. `cascades[c]` is its (camera-relative) light matrix and
        // half-width, for the cascades redrawn this frame.
        let covers = |c: usize, rel: Vec3, radius: f32| {
            cascades[c].is_some_and(|(m, half)| {
                let p = m.transform_point3(rel);
                let r = radius / half;
                p.x.abs() <= 1.0 + r && p.y.abs() <= 1.0 + r
            })
        };

        // Placed parts: culled one by one.
        for i in &self.instances {
            let m = i.model.0 as usize;
            let radius = self.models[m].radius * i.scale;
            let rel = i.position - camera;
            if rel.length() < DRAW_DISTANCE + radius && frustum.intersects_box(rel - radius, rel + radius) {
                self.buckets[m].push(instance_data(i, camera, 0, 1.0));
            }
            for c in 0..CASCADES {
                if covers(c, rel, radius) {
                    self.shadow_buckets[c][m].push(instance_data(i, camera, 0, 1.0));
                }
            }
        }
        // Scatter: whole batches culled by their bounds, then instances by distance.
        for stored in self.scatter.values() {
            let b = &stored.batch;
            let nearest = camera.clamp(stored.min, stored.max).distance(camera);
            let in_view =
                nearest < b.distance && frustum.intersects_box(stored.min - camera, stored.max - camera);
            // Which cascades this batch casts into: those no wider than its kind's shadow distance,
            // that its bounds reach.
            let center = (stored.min + stored.max) * 0.5 - camera;
            let extent = (stored.max - stored.min).length() * 0.5;
            let mut casts = [false; CASCADES];
            for (c, cast) in casts.iter_mut().enumerate() {
                *cast = cascades[c].is_some_and(|(_, half)| half <= b.shadow_distance)
                    && covers(c, center, extent);
            }
            if !in_view && !casts.iter().any(|c| *c) {
                continue;
            }
            for i in &b.instances {
                let m = i.model.0 as usize;
                let dist = (i.position - camera).length();
                let lod = self.lods.get(m).copied().flatten();
                if in_view && dist < b.distance {
                    let fade = ((b.distance - dist) / b.fade.max(1e-3)).clamp(0.0, 1.0);
                    match lod {
                        Some((far, at)) => {
                            // t: 0 fully near model .. 1 fully far model.
                            let t = ((dist - at) / LOD_BAND + 0.5).clamp(0.0, 1.0);
                            if t < 1.0 {
                                let keep = if t > 0.0 { 1.0 - t } else { fade };
                                self.buckets[m].push(instance_data(i, camera, 0, keep));
                            }
                            if t > 0.0 {
                                let keep = if t < 1.0 { -t } else { fade };
                                let far_i = PartInstance { model: far, ..*i };
                                self.buckets[far.0 as usize].push(instance_data(&far_i, camera, 0, keep));
                            }
                        }
                        None => self.buckets[m].push(instance_data(i, camera, 0, fade)),
                    }
                }
                let rel = i.position - camera;
                let radius = self.models[m].radius * i.scale;
                for c in 0..CASCADES {
                    if casts[c] && covers(c, rel, radius) {
                        // Full models in the nearest cascade, impostors in the wider ones: the
                        // switch happens at a cascade boundary, where cascades blend, so it fades.
                        let model = match lod {
                            Some((far, _)) if c > 0 => far,
                            _ => i.model,
                        };
                        let shadow_i = PartInstance { model, ..*i };
                        self.shadow_buckets[c][model.0 as usize]
                            .push(instance_data(&shadow_i, camera, 0, 1.0));
                    }
                }
            }
        }

        self.scratch.clear();
        let flatten = |buckets: &[Vec<InstanceData>], scratch: &mut Vec<InstanceData>| -> Vec<Range> {
            buckets
                .iter()
                .map(|bucket| {
                    let range = Range { start: scratch.len() as u32, count: bucket.len() as u32 };
                    scratch.extend_from_slice(bucket);
                    range
                })
                .collect()
        };
        self.visible = flatten(&self.buckets, &mut self.scratch);
        for c in 0..CASCADES {
            self.shadow[c] = flatten(&self.shadow_buckets[c], &mut self.scratch);
        }
        self.drawn = self.visible.iter().map(|r| r.count as usize).sum();
        self.ghost_slot = self.ghost.map(|g| {
            self.scratch.push(instance_data(&g.instance, camera, if g.valid { 1 } else { 2 }, 1.0));
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

    fn draw_ranges(&self, pass: &mut wgpu::RenderPass<'_>, ranges: &[Range]) {
        pass.set_vertex_buffer(1, self.buffer.slice(..));
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

    /// Parts in the scene pass (pipeline and group 0 already set).
    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        self.draw_ranges(pass, &self.visible);
    }

    /// Shadow casters for one cascade (pipeline and group 0 already set). Materials are bound so
    /// leaf cards cut their shadows out too.
    pub fn draw_shadows(&self, pass: &mut wgpu::RenderPass<'_>, cascade: usize) {
        self.draw_ranges(pass, &self.shadow[cascade]);
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

/// `texture` resampled to `TEXTURE_SIZE` with a full mip chain. Each level's alpha is rescaled so
/// the same share of texels passes `cutoff` as at the top level; otherwise averaging thins leaf
/// cards out until distant trees go bare.
fn mip_chain(texture: &TextureData, cutoff: f32) -> Vec<(u32, Vec<u8>)> {
    let size = TEXTURE_SIZE as usize;
    let (w, h) = (texture.width as usize, texture.height as usize);
    let mut level: Vec<[f32; 4]> = (0..size * size)
        .map(|i| {
            let (x, y) = (i % size * w / size, i / size * h / size);
            let p = &texture.rgba[(y * w + x) * 4..][..4];
            [p[0] as f32, p[1] as f32, p[2] as f32, p[3] as f32 / 255.0]
        })
        .collect();
    let coverage = |l: &[[f32; 4]], scale: f32| {
        l.iter().filter(|p| (p[3] * scale).min(1.0) > cutoff).count() as f32 / l.len() as f32
    };
    let target = coverage(&level, 1.0);
    let mut out = Vec::new();
    let mut s = size;
    loop {
        // Find the alpha scale that keeps the coverage (bisection; coverage grows with scale).
        let (mut lo, mut hi) = (0.0f32, 8.0f32);
        for _ in 0..16 {
            let mid = 0.5 * (lo + hi);
            if coverage(&level, mid) < target { lo = mid } else { hi = mid }
        }
        let scale = if out.is_empty() { 1.0 } else { hi };
        let bytes = level
            .iter()
            .flat_map(|p| {
                [p[0] as u8, p[1] as u8, p[2] as u8, ((p[3] * scale).min(1.0) * 255.0).round() as u8]
            })
            .collect();
        out.push((s as u32, bytes));
        if s == 1 {
            return out;
        }
        let half = s / 2;
        level = (0..half * half)
            .map(|i| {
                let (x, y) = (i % half * 2, i / half * 2);
                let texels = [
                    level[y * s + x],
                    level[y * s + x + 1],
                    level[(y + 1) * s + x],
                    level[(y + 1) * s + x + 1],
                ];
                // Colour weighted by alpha so transparent texels don't darken edges.
                let a: f32 = texels.iter().map(|t| t[3]).sum();
                let mut c = [0.0f32; 4];
                for t in texels {
                    let wgt = if a > 1e-4 { t[3] / a } else { 0.25 };
                    for k in 0..3 {
                        c[k] += t[k] * wgt;
                    }
                }
                c[3] = a / 4.0;
                c
            })
            .collect();
        s = half;
    }
}

pub const PART_VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3, 2 => Float32x2, 3 => Unorm8x4];
pub const PART_INSTANCE_ATTRIBUTES: [wgpu::VertexAttribute; 2] =
    wgpu::vertex_attr_array![4 => Float32x4, 5 => Float32x4];

pub fn vertex_layouts() -> [Option<wgpu::VertexBufferLayout<'static>>; 2] {
    [
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<PartVertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &PART_VERTEX_ATTRIBUTES,
        }),
        Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<InstanceData>() as u64,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &PART_INSTANCE_ATTRIBUTES,
        }),
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mips_keep_their_alpha_coverage() {
        // A sparse pattern: thin lines that box filtering alone would fade below the cutoff.
        let n = 64usize;
        let rgba: Vec<u8> = (0..n * n)
            .flat_map(|i| {
                let on = (i % n).is_multiple_of(8);
                [200, 220, 180, if on { 255 } else { 0 }]
            })
            .collect();
        let tex = TextureData { name: "lines".into(), width: n as u32, height: n as u32, rgba };
        let chain = mip_chain(&tex, 0.5);
        assert_eq!(chain.len() as u32, TEXTURE_MIPS);
        let cover = |(s, px): &(u32, Vec<u8>)| {
            px.as_chunks::<4>().0.iter().filter(|p| p[3] as f32 / 255.0 > 0.5).count() as f32 / (s * s) as f32
        };
        let top = cover(&chain[0]);
        for level in &chain[1..4] {
            assert!((cover(level) - top).abs() < 0.08, "level {}: {} vs {top}", level.0, cover(level));
        }
    }
}
