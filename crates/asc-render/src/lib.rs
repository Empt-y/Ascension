//! Ascension's renderer: terrain, sky, fog, shadows and post-processing on top of wgpu.
//!
//! Everything is drawn camera-relative so precision holds far from the world origin. Depth is
//! reversed-Z (1 near, 0 at infinity). The scene renders into an HDR target, then bloom and an ACES
//! tonemap write the sRGB swapchain, and egui draws on top.

mod camera;
mod gpu;
mod pipelines;
mod shaders;
mod targets;

use std::sync::Arc;
use std::time::Instant;

use ahash::AHashMap;
use anyhow::Result;
use asc_world::{ChunkCoord, ChunkMesh, chunk_origin};
use glam::{Mat4, Vec3, Vec4};
use wgpu::util::DeviceExt;
use winit::window::Window;

pub use camera::{Camera, Frustum};
pub use gpu::Gpu;
use pipelines::{Layouts, Pipelines};
pub use shaders::ShaderLib;
use targets::{BLOOM_LEVELS, SHADOW_SIZE, Targets};

/// Half-width of the area around the camera covered by the sun's shadow map (metres).
const SHADOW_RADIUS: f32 = 110.0;
const SHADOW_DEPTH: f32 = 400.0;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    shadow_view_proj: [[f32; 4]; 4],
    camera: [f32; 4],
    sun: [f32; 4],
    fog: [f32; 4],
    screen: [f32; 4],
    brush: [f32; 4],
}

/// Build every pipeline against an sRGB target without a window. Used by tests.
pub fn build_pipelines_headless(device: &wgpu::Device, shaders: &ShaderLib) -> Result<()> {
    let layouts = Layouts::new(device);
    Pipelines::build(device, shaders, &layouts, wgpu::TextureFormat::Bgra8UnormSrgb).map(|_| ())
}

/// What the game hands the renderer each frame.
pub struct FrameParams {
    pub camera: Camera,
    /// Unit vector towards the sun.
    pub sun_dir: Vec3,
    /// Terrain fades fully into the sky at this horizontal distance.
    pub view_distance: f32,
    /// World-space centre and radius of the dig brush preview.
    pub brush: Option<(Vec3, f32)>,
}

/// egui output for this frame.
pub struct UiFrame<'a> {
    pub primitives: &'a [egui::ClippedPrimitive],
    pub textures: &'a egui::TexturesDelta,
    pub pixels_per_point: f32,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct FrameStats {
    pub chunks_drawn: usize,
    pub shadow_chunks: usize,
    pub triangles: usize,
}

struct GpuChunk {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    /// Chunk-local bounds of the mesh.
    min: Vec3,
    max: Vec3,
}

pub struct Renderer {
    gpu: Gpu,
    shaders: ShaderLib,
    layouts: Layouts,
    pipelines: Pipelines,
    targets: Targets,
    globals: wgpu::Buffer,
    offsets: wgpu::Buffer,
    offsets_capacity: usize,
    shadow_map: wgpu::TextureView,
    shadow_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,
    placeholder: wgpu::TextureView,
    scene_bg: wgpu::BindGroup,
    shadow_bg: wgpu::BindGroup,
    post_bg: wgpu::BindGroup,
    bloom_bgs: BloomBindGroups,
    chunks: AHashMap<ChunkCoord, GpuChunk>,
    // Per-frame scratch, kept to avoid allocating every frame.
    draw_list: Vec<(f32, ChunkCoord)>,
    shadow_list: Vec<ChunkCoord>,
    offset_data: Vec<[f32; 4]>,
    egui: egui_wgpu::Renderer,
    start: Instant,
}

struct BloomBindGroups {
    prefilter: wgpu::BindGroup,
    /// `down[i]` reads level i and writes level i + 1.
    down: Vec<wgpu::BindGroup>,
    /// `up[i]` reads level i + 1 and adds into level i.
    up: Vec<wgpu::BindGroup>,
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        let gpu = Gpu::new(window)?;
        let device = &gpu.device;
        let shaders = ShaderLib::new();
        let layouts = Layouts::new(device);
        let pipelines = Pipelines::build(device, &shaders, &layouts, gpu.config.format)?;
        let targets = Targets::new(device, gpu.size());

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let offsets_capacity = 4096;
        let offsets = Self::offsets_buffer(device, offsets_capacity);
        let shadow_map = targets::shadow_map(device);
        let shadow_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let linear_sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("linear clamp"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let placeholder = targets::placeholder_texture(device);
        let egui = egui_wgpu::Renderer::new(device, gpu.ui_format(), egui_wgpu::RendererOptions::default());

        let (scene_bg, shadow_bg) =
            Self::scene_bind_groups(device, &layouts, &globals, &offsets, &shadow_map, &shadow_sampler);
        let (post_bg, bloom_bgs) =
            Self::texture_bind_groups(device, &layouts, &targets, &placeholder, &linear_sampler);
        Ok(Self {
            gpu,
            shaders,
            layouts,
            pipelines,
            targets,
            globals,
            offsets,
            offsets_capacity,
            shadow_map,
            shadow_sampler,
            linear_sampler,
            placeholder,
            scene_bg,
            shadow_bg,
            post_bg,
            bloom_bgs,
            chunks: AHashMap::new(),
            draw_list: Vec::new(),
            shadow_list: Vec::new(),
            offset_data: Vec::new(),
            egui,
            start: Instant::now(),
        })
    }

    fn offsets_buffer(device: &wgpu::Device, capacity: usize) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk offsets"),
            size: (capacity * 16) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    fn scene_bind_groups(
        device: &wgpu::Device,
        layouts: &Layouts,
        globals: &wgpu::Buffer,
        offsets: &wgpu::Buffer,
        shadow_map: &wgpu::TextureView,
        shadow_sampler: &wgpu::Sampler,
    ) -> (wgpu::BindGroup, wgpu::BindGroup) {
        let scene = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("scene"),
            layout: &layouts.scene,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: offsets.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::TextureView(shadow_map) },
                wgpu::BindGroupEntry { binding: 3, resource: wgpu::BindingResource::Sampler(shadow_sampler) },
            ],
        });
        let shadow = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow"),
            layout: &layouts.shadow,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: globals.as_entire_binding() },
                wgpu::BindGroupEntry { binding: 1, resource: offsets.as_entire_binding() },
            ],
        });
        (scene, shadow)
    }

    fn texture_bind_groups(
        device: &wgpu::Device,
        layouts: &Layouts,
        targets: &Targets,
        placeholder: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
    ) -> (wgpu::BindGroup, BloomBindGroups) {
        let pair = |a: &wgpu::TextureView, b: &wgpu::TextureView| {
            device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("textures"),
                layout: &layouts.textures,
                entries: &[
                    wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(a) },
                    wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(b) },
                    wgpu::BindGroupEntry { binding: 2, resource: wgpu::BindingResource::Sampler(sampler) },
                ],
            })
        };
        let post = pair(&targets.hdr, &targets.bloom[0]);
        let bloom = BloomBindGroups {
            prefilter: pair(&targets.hdr, placeholder),
            down: (0..BLOOM_LEVELS - 1).map(|i| pair(&targets.bloom[i], placeholder)).collect(),
            up: (0..BLOOM_LEVELS - 1).map(|i| pair(&targets.bloom[i + 1], placeholder)).collect(),
        };
        (post, bloom)
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.gpu.device
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        self.gpu.set_vsync(vsync);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.gpu.resize(width, height);
        self.targets = Targets::new(&self.gpu.device, self.gpu.size());
        (self.post_bg, self.bloom_bgs) = Self::texture_bind_groups(
            &self.gpu.device,
            &self.layouts,
            &self.targets,
            &self.placeholder,
            &self.linear_sampler,
        );
    }

    /// Recompile every shader from disk. On failure the old pipelines stay in use.
    pub fn reload_shaders(&mut self) -> Result<()> {
        let pipelines =
            Pipelines::build(&self.gpu.device, &self.shaders, &self.layouts, self.gpu.config.format)?;
        self.pipelines = pipelines;
        Ok(())
    }

    pub fn hot_reload_available(&self) -> bool {
        self.shaders.hot_reload_available()
    }

    pub fn upload_chunk(&mut self, coord: ChunkCoord, mesh: &ChunkMesh) {
        if mesh.is_empty() {
            self.chunks.remove(&coord);
            return;
        }
        let device = &self.gpu.device;
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for v in &mesh.vertices {
            let p = Vec3::from(v.position);
            min = min.min(p);
            max = max.max(p);
        }
        let vertices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk vertices"),
            contents: bytemuck::cast_slice(&mesh.vertices),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk indices"),
            contents: bytemuck::cast_slice(&mesh.indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        self.chunks
            .insert(coord, GpuChunk { vertices, indices, index_count: mesh.indices.len() as u32, min, max });
    }

    pub fn remove_chunk(&mut self, coord: ChunkCoord) {
        self.chunks.remove(&coord);
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Light-space matrix for camera-relative positions, snapped to shadow texels so it doesn't shimmer.
    fn shadow_matrix(camera_pos: Vec3, sun: Vec3) -> Mat4 {
        let up = if sun.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
        let light_view = Mat4::look_to_rh(Vec3::ZERO, -sun, up);
        let cam_ls = light_view.transform_point3(camera_pos);
        let texel = 2.0 * SHADOW_RADIUS / SHADOW_SIZE as f32;
        let snapped = (cam_ls.truncate() / texel).floor() * texel;
        let shift = Vec3::new(cam_ls.x - snapped.x, cam_ls.y - snapped.y, -(SHADOW_DEPTH * 0.5));
        let proj = Mat4::orthographic_rh(
            -SHADOW_RADIUS,
            SHADOW_RADIUS,
            -SHADOW_RADIUS,
            SHADOW_RADIUS,
            0.0,
            SHADOW_DEPTH,
        );
        // Rotation of the relative position into light space, then centre on the snapped camera.
        proj * Mat4::from_translation(shift) * light_view
    }

    pub fn render(&mut self, frame: &FrameParams, ui: Option<UiFrame<'_>>) -> FrameStats {
        let surface = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                let (w, h) = self.gpu.size();
                self.resize(w, h);
                return FrameStats::default();
            }
            _ => return FrameStats::default(),
        };
        let (width, height) = self.gpu.size();
        let aspect = width as f32 / height as f32;
        let cam = &frame.camera;
        let view_proj = cam.view_proj(aspect);
        let shadow_view_proj = Self::shadow_matrix(cam.position, frame.sun_dir);
        let stats = self.build_draw_lists(cam.position, view_proj, frame.view_distance);

        let daylight = smoothstep(-0.08, 0.25, frame.sun_dir.y);
        let exposure = 1.0 + (1.0 - daylight) * 1.5;
        let globals = Globals {
            view_proj: view_proj.to_cols_array_2d(),
            inv_view_proj: view_proj.inverse().to_cols_array_2d(),
            shadow_view_proj: shadow_view_proj.to_cols_array_2d(),
            camera: cam.position.extend(self.start.elapsed().as_secs_f32()).into(),
            sun: frame.sun_dir.extend(daylight).into(),
            fog: [0.0022, 0.012, frame.view_distance, exposure],
            screen: [width as f32, height as f32, 1.0 / width as f32, 1.0 / height as f32],
            brush: match frame.brush {
                Some((center, radius)) => (center - cam.position).extend(radius).into(),
                None => Vec4::ZERO.into(),
            },
        };
        let queue = &self.gpu.queue;
        queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
        queue.write_buffer(&self.offsets, 0, bytemuck::cast_slice(&self.offset_data));

        let device = &self.gpu.device;
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        self.shadow_pass(&mut encoder);
        self.scene_pass(&mut encoder);
        self.bloom_passes(&mut encoder);

        let view = surface.texture.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut pass = color_pass(&mut encoder, "post", &view, Some(wgpu::Color::BLACK));
            pass.set_pipeline(&self.pipelines.post);
            pass.set_bind_group(0, &self.scene_bg, &[]);
            pass.set_bind_group(1, &self.post_bg, &[]);
            pass.draw(0..3, 0..1);
        }

        let mut extra = Vec::new();
        if let Some(ui) = &ui {
            let screen = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [width, height],
                pixels_per_point: ui.pixels_per_point,
            };
            for (id, deltas) in &ui.textures.set {
                for delta in deltas {
                    self.egui.update_texture(device, queue, *id, delta);
                }
            }
            extra = self.egui.update_buffers(device, queue, &mut encoder, ui.primitives, &screen);
            let ui_view = surface.texture.create_view(&wgpu::TextureViewDescriptor {
                format: Some(self.gpu.ui_format()),
                ..Default::default()
            });
            let mut pass = color_pass(&mut encoder, "egui", &ui_view, None).forget_lifetime();
            self.egui.render(&mut pass, ui.primitives, &screen);
        }
        queue.submit(extra.into_iter().chain([encoder.finish()]));
        if let Some(ui) = &ui {
            for id in &ui.textures.free {
                self.egui.free_texture(id);
            }
        }
        queue.present(surface);
        stats
    }

    /// Pick visible and shadow-casting chunks and lay out their camera-relative offsets: visible chunks
    /// first (front to back), then shadow casters. Instance index = position in this array.
    fn build_draw_lists(&mut self, camera_pos: Vec3, view_proj: Mat4, view_distance: f32) -> FrameStats {
        let frustum = Frustum::from_view_proj(view_proj);
        self.draw_list.clear();
        self.shadow_list.clear();
        let mut stats = FrameStats::default();
        let shadow_reach = SHADOW_RADIUS * 1.5 + 32.0;
        for (coord, chunk) in &self.chunks {
            let origin = chunk_origin(*coord) - camera_pos;
            let (min, max) = (origin + chunk.min, origin + chunk.max);
            let nearest = Vec3::ZERO.clamp(min, max);
            let dist = nearest.length();
            if dist < shadow_reach {
                self.shadow_list.push(*coord);
            }
            if nearest.truncate().length() < view_distance && frustum.intersects_box(min, max) {
                self.draw_list.push((dist, *coord));
                stats.triangles += chunk.index_count as usize / 3;
            }
        }
        self.draw_list.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        stats.chunks_drawn = self.draw_list.len();
        stats.shadow_chunks = self.shadow_list.len();

        self.offset_data.clear();
        let draws = self.draw_list.iter().map(|(_, c)| *c);
        for coord in draws.chain(self.shadow_list.iter().copied()) {
            self.offset_data.push((chunk_origin(coord) - camera_pos).extend(0.0).into());
        }
        if self.offset_data.len() > self.offsets_capacity {
            self.offsets_capacity = self.offset_data.len().next_power_of_two();
            self.offsets = Self::offsets_buffer(&self.gpu.device, self.offsets_capacity);
            (self.scene_bg, self.shadow_bg) = Self::scene_bind_groups(
                &self.gpu.device,
                &self.layouts,
                &self.globals,
                &self.offsets,
                &self.shadow_map,
                &self.shadow_sampler,
            );
        }
        stats
    }

    fn draw_chunk(&self, pass: &mut wgpu::RenderPass<'_>, coord: ChunkCoord, instance: u32) {
        let chunk = &self.chunks[&coord];
        pass.set_vertex_buffer(0, chunk.vertices.slice(..));
        pass.set_index_buffer(chunk.indices.slice(..), wgpu::IndexFormat::Uint32);
        pass.draw_indexed(0..chunk.index_count, 0, instance..instance + 1);
    }

    fn shadow_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("shadow"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.shadow_map,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_pipeline(&self.pipelines.shadow);
        pass.set_bind_group(0, &self.shadow_bg, &[]);
        let base = self.draw_list.len() as u32;
        for (i, coord) in self.shadow_list.iter().enumerate() {
            self.draw_chunk(&mut pass, *coord, base + i as u32);
        }
    }

    fn scene_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &self.targets.hdr,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.targets.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &self.scene_bg, &[]);
        pass.set_pipeline(&self.pipelines.terrain);
        for (i, (_, coord)) in self.draw_list.iter().enumerate() {
            self.draw_chunk(&mut pass, *coord, i as u32);
        }
        pass.set_pipeline(&self.pipelines.sky);
        pass.draw(0..3, 0..1);
    }

    fn bloom_passes(&self, encoder: &mut wgpu::CommandEncoder) {
        let fullscreen = |encoder: &mut wgpu::CommandEncoder,
                          target: &wgpu::TextureView,
                          pipeline: &wgpu::RenderPipeline,
                          bg: &wgpu::BindGroup,
                          clear: bool| {
            let mut pass = color_pass(encoder, "bloom", target, clear.then_some(wgpu::Color::BLACK));
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &self.scene_bg, &[]);
            pass.set_bind_group(1, bg, &[]);
            pass.draw(0..3, 0..1);
        };
        let p = &self.pipelines;
        let levels = &self.targets.bloom;
        fullscreen(encoder, &levels[0], &p.bloom_prefilter, &self.bloom_bgs.prefilter, true);
        for i in 0..BLOOM_LEVELS - 1 {
            fullscreen(encoder, &levels[i + 1], &p.bloom_down, &self.bloom_bgs.down[i], true);
        }
        for i in (0..BLOOM_LEVELS - 1).rev() {
            fullscreen(encoder, &levels[i], &p.bloom_up, &self.bloom_bgs.up[i], false);
        }
    }
}

fn color_pass<'e>(
    encoder: &'e mut wgpu::CommandEncoder,
    label: &str,
    view: &wgpu::TextureView,
    clear: Option<wgpu::Color>,
) -> wgpu::RenderPass<'e> {
    encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
        label: Some(label),
        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
            view,
            depth_slice: None,
            resolve_target: None,
            ops: wgpu::Operations {
                load: clear.map_or(wgpu::LoadOp::Load, wgpu::LoadOp::Clear),
                store: wgpu::StoreOp::Store,
            },
        })],
        depth_stencil_attachment: None,
        timestamp_writes: None,
        occlusion_query_set: None,
        multiview_mask: None,
    })
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
