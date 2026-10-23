//! Ascension's renderer: terrain, sky, fog, shadows and post-processing on top of wgpu.
//!
//! Everything is drawn camera-relative so precision holds far from the world origin. Depth is
//! reversed-Z (1 near, 0 at infinity). The scene renders into an HDR target, then bloom and an ACES
//! tonemap write the sRGB swapchain, and egui draws on top.

mod arena;
mod camera;
pub mod clipmap;
mod far;
mod gpu;
pub mod models;
mod parts;
mod pipelines;
mod shaders;
mod targets;
mod timing;

use std::sync::Arc;
use std::time::Instant;

use ahash::AHashMap;
use anyhow::Result;
use asc_world::{ChunkCoord, ChunkMesh, chunk_origin};
use glam::{Mat4, Vec3, Vec4};
use winit::window::Window;

use arena::Arena;
pub use camera::{Camera, Frustum};
use far::FarTerrain;
pub use gpu::Gpu;
use parts::Parts;
pub use parts::{Ghost, ModelId, PartInstance, ScatterBatch};
use pipelines::{Layouts, Pipelines};
pub use shaders::ShaderLib;
use targets::{BLOOM_LEVELS, CASCADES, SHADOW_LAYERS, SHADOW_SIZE, Targets};

/// With far terrain on, everything has faded into the sky by this distance (metres).
pub const FAR_RANGE: f32 = 14_000.0;
const DEFAULT_FOG_DENSITY: f32 = 0.0003;
/// Samples per pixel for the scene pass (4x MSAA smooths silhouettes and geometric edges).
pub const DEFAULT_MSAA: u32 = 4;

/// Half-width of the area around the camera covered by each sun shadow cascade (metres), nearest
/// first: centimetre-sharp shadows close up, still shadowed out past 100 m.
const CASCADE_RADII: [f32; CASCADES] = [16.0, 50.0, 120.0];
/// Depth range of every cascade (matches SHADOW_DEPTH in common.wgsl).
const SHADOW_DEPTH: f32 = 500.0;
/// The sun direction used for shadows only moves in steps of this angle (radians, half a degree).
/// Rotating the shadow maps a little every frame re-rasterises every leaf and edge onto a new
/// grid, and the result sparkles. Instead each step draws a fresh set of cascades and the old set
/// dissolves into it over `SHADOW_FADE` seconds, so shadows drift smoothly with the sun.
const SHADOW_SUN_STEP: f32 = 0.0087;
const SHADOW_FADE: f32 = 1.2;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    inv_view_proj: [[f32; 4]; 4],
    shadow_view_proj: [[[f32; 4]; 4]; SHADOW_LAYERS],
    camera: [f32; 4],
    sun: [f32; 4],
    fog: [f32; 4],
    screen: [f32; 4],
    brush: [f32; 4],
    far: [f32; 4],
    /// x = the layer being rendered (shadow passes), y = 1 if shadows are on, z = wind strength,
    /// w = how far the current cascade set has faded in over the previous one (0..1).
    shadow: [f32; 4],
    /// xyz = each cascade's half-width, w = the current set (0 or 1).
    shadow_radii: [f32; 4],
    /// TAA: this frame's unjittered NDC and depth to last frame's clip space.
    taa_reproject: [[f32; 4]; 4],
    /// x = 1 if the history is usable, y = weight of the new frame, zw = this frame's jitter (NDC).
    taa: [f32; 4],
    /// Last frame's x = time, y = wind strength (for motion vectors of swaying plants).
    motion: [f32; 4],
    /// The window size: xy pixels, zw 1 / pixels (`screen` is the render size).
    output: [f32; 4],
    /// x = render scale, y = RCAS sharpness.
    upscale: [f32; 4],
    /// x = 1 when the camera is underwater, y = point light count, z = volcanic ash (0..1).
    env: [f32; 4],
    light_pos: [[f32; 4]; MAX_LIGHTS],
    light_color: [[f32; 4]; MAX_LIGHTS],
}

/// Point lights passed to the shaders each frame (the nearest of those set).
pub const MAX_LIGHTS: usize = 16;

/// A point light: world position, reach in metres, and colour times intensity.
#[derive(Clone, Copy, Debug)]
pub struct PointLight {
    pub position: Vec3,
    pub radius: f32,
    pub color: Vec3,
}

/// Render scales of the upscaling presets: native, quality, balanced, performance.
pub const RENDER_SCALES: [(f32, &str); 4] =
    [(1.0, "native"), (0.67, "quality"), (0.59, "balanced"), (0.5, "performance")];
/// RCAS sharpness at native resolution and at the lowest render scale (upscaled images are softer).
const SHARPNESS: (f32, f32) = (0.35, 0.6);

/// Weight of each new frame in the TAA history (the rest is the history).
const TAA_FEEDBACK: f32 = 0.08;
/// How much of a pixel the jitter spans: a little under the whole pixel keeps the image crisper.
const TAA_JITTER: f32 = 0.8;
/// The history is thrown away when the camera jumps further than this between frames.
const TAA_RESET_DISTANCE: f32 = 8.0;

/// TAA debug counters' readback (ASC_DEBUG_TAA).
struct TaaDebug {
    readback: wgpu::Buffer,
    state: std::sync::Arc<std::sync::atomic::AtomicU8>,
    since: Instant,
    frames: u32,
    frames_in_copy: u32,
    /// Fireflies, hot inputs, NaN/infinite pixels, and the frames they were counted over.
    last: Option<[u32; 4]>,
    /// Hot pixels by source: terrain 1, foliage 2, other models 3, far terrain 4, sky 5, water 6.
    sources: [u32; 8],
}

impl TaaDebug {
    fn new(device: &wgpu::Device) -> Self {
        log::info!("TAA debug: counting and tinting fireflies (magenta), sun-bright inputs (cyan), NaN (yellow)");
        Self {
            readback: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("taa debug readback"),
                size: 48,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            state: std::sync::Arc::new(std::sync::atomic::AtomicU8::new(0)),
            since: Instant::now(),
            frames: 0,
            frames_in_copy: 0,
            last: None,
            sources: [0; 8],
        }
    }
}

/// Element `i` of the Halton sequence in base `b`, in [0, 1).
fn halton(mut i: u32, b: u32) -> f32 {
    let (mut f, mut r) = (1.0, 0.0);
    while i > 0 {
        f /= b as f32;
        r += f * (i % b) as f32;
        i /= b;
    }
    r
}

/// Build every pipeline against an sRGB target without a window. Used by tests.
pub fn build_pipelines_headless(device: &wgpu::Device, shaders: &ShaderLib) -> Result<()> {
    let layouts = Layouts::new(device);
    Pipelines::build(device, shaders, &layouts, wgpu::TextureFormat::Bgra8UnormSrgb, DEFAULT_MSAA).map(|_| ())
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
    /// The camera is underwater.
    pub underwater: bool,
    /// Point lights near the camera (the nearest `MAX_LIGHTS` are used).
    pub lights: Vec<PointLight>,
    /// 0..1 volcanic ash in the air round the camera: darker sky, thicker fog, dimmer sun.
    pub ash: f32,
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
    pub parts_drawn: usize,
    pub far_levels: usize,
    pub far_triangles: usize,
    pub shadow_chunks: usize,
    pub triangles: usize,
    /// CPU time blocked acquiring the swapchain image, submitting, and presenting.
    pub acquire_ms: f32,
    pub submit_ms: f32,
    pub present_ms: f32,
}

/// Where a chunk's mesh lives in the shared arenas.
struct GpuChunk {
    first_vertex: u32,
    vertex_count: u32,
    first_index: u32,
    index_count: u32,
    /// Chunk-local bounds of the mesh.
    min: Vec3,
    max: Vec3,
}

pub struct Renderer {
    gpu: Gpu,
    /// Per-stage GPU timings, where supported (a cell: the passes mark it from `&self`).
    timer: std::cell::RefCell<Option<timing::GpuTimer>>,
    shaders: ShaderLib,
    layouts: Layouts,
    pipelines: Pipelines,
    targets: Targets,
    globals: wgpu::Buffer,
    offsets: wgpu::Buffer,
    offsets_capacity: usize,
    shadow_map: wgpu::TextureView,
    shadow_layers: Vec<wgpu::TextureView>,
    /// A copy of the globals per cascade, differing only in which cascade is being rendered.
    cascade_globals: Vec<wgpu::Buffer>,
    shadow_sampler: wgpu::Sampler,
    linear_sampler: wgpu::Sampler,
    placeholder: wgpu::TextureView,
    scene_bg: wgpu::BindGroup,
    shadow_bgs: Vec<wgpu::BindGroup>,
    post_bg: wgpu::BindGroup,
    bloom_bgs: BloomBindGroups,
    taa_bgs: TaaBindGroups,
    /// Temporal anti-aliasing on, which history target this frame writes, and whether the
    /// history holds a usable image.
    taa_on: bool,
    /// TAA debug counters (ASC_DEBUG_TAA): the GPU buffer, a readback copy, whether that copy is
    /// mapped and ready (0 idle, 1 pending, 2 ready), and the last counts read.
    taa_debug: Option<TaaDebug>,
    taa_counts: wgpu::Buffer,
    /// Sun disc brightness (ASC_SUN_DISC=0 hides it, for isolating tests).
    sun_disc: f32,
    /// Scene resolution as a fraction of the window's (upscaled by TAA; 1 without TAA).
    render_scale: f32,
    taa_index: usize,
    taa_valid: bool,
    /// Last frame's unjittered view-projection and camera position, for reprojection.
    prev_view_proj: Mat4,
    prev_camera: Vec3,
    /// Last frame's `camera.w` time and wind strength.
    prev_time: f32,
    prev_wind: f32,
    chunks: AHashMap<ChunkCoord, GpuChunk>,
    vertices: Arena,
    indices: Arena,
    far: FarTerrain,
    parts: Parts,
    pub fog_density: f32,
    msaa: u32,
    // Per-frame scratch, kept to avoid allocating every frame.
    draw_list: Vec<(f32, ChunkCoord)>,
    /// Simulated water meshes, per chunk (sharing the terrain arenas), and the ones visible this
    /// frame (their offsets follow the terrain's).
    fluid_chunks: AHashMap<ChunkCoord, GpuChunk>,
    /// The mesh arenas hit the device's buffer size limit (warned once).
    arena_full_warned: bool,
    fluid_draw: Vec<ChunkCoord>,
    shadow_list: Vec<ChunkCoord>,
    /// Per cascade, the indices into `shadow_list` of the chunks that cast into it.
    shadow_sets: [Vec<u32>; CASCADES],
    shadows: bool,
    wind: f32,
    /// Each layer's matrix and the camera position it was rendered from. Layers are only redrawn when
    /// needed (see `render`); in between they're looked up through these.
    cascade_state: [(Mat4, Vec3); SHADOW_LAYERS],
    /// Which layers are redrawn this frame.
    cascade_update: [bool; SHADOW_LAYERS],
    /// The current cascade set, and when it started fading in.
    shadow_set: usize,
    shadow_fade_start: Instant,
    frame: u64,
    /// The sun direction the shadow maps are rendered from (see `SHADOW_SUN_STEP`).
    shadow_sun: Vec3,
    /// Cascades with a changed shadow caster in reach since they were last drawn.
    shadow_dirty: [bool; CASCADES],
    /// The camera position at the last frame, to judge which cascades a change reaches.
    last_camera: Vec3,
    offset_data: Vec<[f32; 4]>,
    egui: egui_wgpu::Renderer,
    start: Instant,
}

/// Per history target `i` (this frame writes `taa[i]` and reads `taa[1 - i]`): the TAA pass, and
/// post and bloom reading the anti-aliased result.
struct TaaBindGroups {
    pass: [wgpu::BindGroup; 2],
    post: [wgpu::BindGroup; 2],
    prefilter: [wgpu::BindGroup; 2],
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
        let msaa = DEFAULT_MSAA;
        let pipelines = Pipelines::build(device, &shaders, &layouts, gpu.config.format, msaa)?;
        let targets = Targets::new(device, gpu.size(), gpu.size(), msaa);

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let offsets_capacity = 4096;
        let offsets = Self::offsets_buffer(device, offsets_capacity);
        let (shadow_map, shadow_layers) = targets::shadow_maps(device);
        let cascade_globals: Vec<wgpu::Buffer> = (0..SHADOW_LAYERS)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("cascade globals"),
                    size: std::mem::size_of::<Globals>() as u64,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect();
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
        let vertex_size = std::mem::size_of::<asc_world::Vertex>() as u64;
        let vertices = Arena::new(device, "chunk vertices", wgpu::BufferUsages::VERTEX, vertex_size, 2 << 20);
        let indices = Arena::new(device, "chunk indices", wgpu::BufferUsages::INDEX, 4, 12 << 20);
        let far = FarTerrain::new(device, &layouts.far);
        let parts = Parts::new(device);
        let egui = egui_wgpu::Renderer::new(device, gpu.ui_format(), egui_wgpu::RendererOptions::default());

        let (scene_bg, shadow_bgs) = Self::scene_bind_groups(
            device,
            &layouts,
            &globals,
            &cascade_globals,
            &offsets,
            &shadow_map,
            &shadow_sampler,
        );
        let taa_counts = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("taa debug counts"),
            size: 48,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let taa_debug = std::env::var("ASC_DEBUG_TAA").is_ok_and(|v| v == "1").then(|| TaaDebug::new(device));
        let (post_bg, bloom_bgs, taa_bgs) = Self::texture_bind_groups(
            device,
            &layouts,
            &targets,
            &placeholder,
            &linear_sampler,
            msaa,
            &taa_counts,
        );
        let timer = std::cell::RefCell::new(timing::GpuTimer::new(&gpu.device, &gpu.queue));
        Ok(Self {
            timer,
            gpu,
            shaders,
            layouts,
            pipelines,
            targets,
            globals,
            offsets,
            offsets_capacity,
            shadow_map,
            shadow_layers,
            cascade_globals,
            shadow_sampler,
            linear_sampler,
            placeholder,
            scene_bg,
            shadow_bgs,
            post_bg,
            bloom_bgs,
            taa_bgs,
            taa_on: true,
            taa_counts,
            taa_debug,
            sun_disc: std::env::var("ASC_SUN_DISC").ok().and_then(|v| v.parse().ok()).unwrap_or(1.0),
            render_scale: 1.0,
            taa_index: 0,
            taa_valid: false,
            prev_view_proj: Mat4::IDENTITY,
            prev_camera: Vec3::ZERO,
            prev_time: 0.0,
            prev_wind: 1.0,
            chunks: AHashMap::new(),
            vertices,
            indices,
            far,
            parts,
            fog_density: DEFAULT_FOG_DENSITY,
            msaa,
            draw_list: Vec::new(),
            fluid_chunks: AHashMap::new(),
            arena_full_warned: false,
            fluid_draw: Vec::new(),
            shadow_list: Vec::new(),
            shadow_sets: Default::default(),
            shadows: true,
            wind: 1.0,
            cascade_state: [(Mat4::IDENTITY, Vec3::ZERO); SHADOW_LAYERS],
            cascade_update: [true; SHADOW_LAYERS],
            shadow_set: 0,
            shadow_fade_start: Instant::now(),
            frame: 0,
            shadow_sun: Vec3::Y,
            shadow_dirty: [true; CASCADES],
            last_camera: Vec3::ZERO,
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

    #[allow(clippy::too_many_arguments)]
    fn scene_bind_groups(
        device: &wgpu::Device,
        layouts: &Layouts,
        globals: &wgpu::Buffer,
        cascade_globals: &[wgpu::Buffer],
        offsets: &wgpu::Buffer,
        shadow_map: &wgpu::TextureView,
        shadow_sampler: &wgpu::Sampler,
    ) -> (wgpu::BindGroup, Vec<wgpu::BindGroup>) {
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
        let shadow = cascade_globals
            .iter()
            .map(|g| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("shadow"),
                    layout: &layouts.shadow,
                    entries: &[
                        wgpu::BindGroupEntry { binding: 0, resource: g.as_entire_binding() },
                        wgpu::BindGroupEntry { binding: 1, resource: offsets.as_entire_binding() },
                    ],
                })
            })
            .collect();
        (scene, shadow)
    }

    fn texture_bind_groups(
        device: &wgpu::Device,
        layouts: &Layouts,
        targets: &Targets,
        placeholder: &wgpu::TextureView,
        sampler: &wgpu::Sampler,
        msaa: u32,
        debug_buffer: &wgpu::Buffer,
    ) -> (wgpu::BindGroup, BloomBindGroups, TaaBindGroups) {
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
        let taa_layout = if msaa > 1 { &layouts.taa_ms } else { &layouts.taa_ss };
        let taa = TaaBindGroups {
            pass: [0, 1].map(|i| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("taa"),
                    layout: taa_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::TextureView(&targets.hdr),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&targets.taa[1 - i]),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(sampler),
                        },
                        wgpu::BindGroupEntry {
                            binding: 3,
                            resource: wgpu::BindingResource::TextureView(&targets.depth),
                        },
                        wgpu::BindGroupEntry {
                            binding: 4,
                            resource: wgpu::BindingResource::TextureView(&targets.motion),
                        },
                        wgpu::BindGroupEntry { binding: 5, resource: debug_buffer.as_entire_binding() },
                    ],
                })
            }),
            post: [0, 1].map(|i| pair(&targets.taa[i], &targets.bloom[0])),
            prefilter: [0, 1].map(|i| pair(&targets.taa[i], placeholder)),
        };
        (post, bloom, taa)
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
        self.targets = Targets::new(&self.gpu.device, self.gpu.size(), self.render_size(), self.msaa);
        (self.post_bg, self.bloom_bgs, self.taa_bgs) = Self::texture_bind_groups(
            &self.gpu.device,
            &self.layouts,
            &self.targets,
            &self.placeholder,
            &self.linear_sampler,
            self.msaa,
            &self.taa_counts,
        );
        self.taa_valid = false;
    }

    /// Temporal anti-aliasing on or off. (Off also turns upscaling off.)
    pub fn set_taa(&mut self, on: bool) {
        self.taa_on = on;
        self.taa_valid = false;
        let (w, h) = self.gpu.size();
        self.resize(w, h);
    }

    /// Render the scene at `scale` of the window's resolution and upscale it (through TAA).
    pub fn set_render_scale(&mut self, scale: f32) {
        self.render_scale = scale.clamp(0.25, 1.0);
        let (w, h) = self.gpu.size();
        self.resize(w, h);
    }

    /// The scale in effect: upscaling needs TAA.
    pub fn render_scale(&self) -> f32 {
        if self.taa_on { self.render_scale } else { 1.0 }
    }

    /// The scene's resolution.
    pub fn render_size(&self) -> (u32, u32) {
        let (w, h) = self.gpu.size();
        let s = self.render_scale();
        (((w as f32 * s).round() as u32).max(1), ((h as f32 * s).round() as u32).max(1))
    }

    /// Recompile every shader from disk. On failure the old pipelines stay in use.
    pub fn reload_shaders(&mut self) -> Result<()> {
        let pipelines = Pipelines::build(
            &self.gpu.device,
            &self.shaders,
            &self.layouts,
            self.gpu.config.format,
            self.msaa,
        )?;
        self.pipelines = pipelines;
        Ok(())
    }

    pub fn msaa(&self) -> u32 {
        self.msaa
    }

    /// Change the scene's sample count (1 = off). Rebuilds the targets and pipelines.
    pub fn set_msaa(&mut self, samples: u32) -> Result<()> {
        let samples = if samples > 1 { 4 } else { 1 };
        let pipelines = Pipelines::build(
            &self.gpu.device,
            &self.shaders,
            &self.layouts,
            self.gpu.config.format,
            samples,
        )?;
        self.pipelines = pipelines;
        self.msaa = samples;
        let (w, h) = self.gpu.size();
        self.resize(w, h);
        Ok(())
    }

    pub fn hot_reload_available(&self) -> bool {
        self.shaders.hot_reload_available()
    }

    /// Something that casts shadows changed within `min..max` (world space); redraw the cascades it
    /// can reach, at most `limit` metres from the camera.
    fn shadow_changed(&mut self, min: Vec3, max: Vec3, limit: f32) {
        let nearest = self.last_camera.clamp(min, max).distance(self.last_camera);
        for (c, dirty) in self.shadow_dirty.iter_mut().enumerate() {
            if nearest < (CASCADE_RADII[c] * 1.25 + 32.0).min(limit) {
                *dirty = true;
            }
        }
    }

    /// Put a mesh into the shared arenas: its first vertex and index, or `None` (logged once) if
    /// they're at the device's buffer size limit, when the mesh is left out rather than crashing.
    fn store_mesh(&mut self, mesh: &ChunkMesh) -> Option<(u32, u32)> {
        let (device, queue) = (&self.gpu.device, &self.gpu.queue);
        let vertex_count = mesh.vertices.len() as u32;
        let first_vertex = self.vertices.upload(device, queue, bytemuck::cast_slice(&mesh.vertices));
        let first_index = first_vertex.and_then(|_| self.indices.upload(device, queue, bytemuck::cast_slice(&mesh.indices)));
        match (first_vertex, first_index) {
            (Some(v), Some(i)) => Some((v, i)),
            (v, _) => {
                if let Some(v) = v {
                    self.vertices.free(v, vertex_count);
                }
                if !self.arena_full_warned {
                    self.arena_full_warned = true;
                    log::warn!("chunk mesh storage is at the GPU's buffer size limit: leaving meshes out");
                }
                None
            }
        }
    }

    pub fn upload_chunk(&mut self, coord: ChunkCoord, mesh: &ChunkMesh) {
        self.remove_chunk(coord);
        if mesh.is_empty() {
            return;
        }
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for v in &mesh.vertices {
            let p = Vec3::from(v.position);
            min = min.min(p);
            max = max.max(p);
        }
        let Some((first_vertex, first_index)) = self.store_mesh(mesh) else { return };
        self.chunks.insert(
            coord,
            GpuChunk {
                first_vertex,
                vertex_count: mesh.vertices.len() as u32,
                first_index,
                index_count: mesh.indices.len() as u32,
                min,
                max,
            },
        );
        let origin = chunk_origin(coord);
        self.shadow_changed(origin + min, origin + max, f32::MAX);
    }

    /// Replace the simulated water mesh of a chunk (an empty mesh removes it).
    pub fn upload_fluid(&mut self, coord: ChunkCoord, mesh: &ChunkMesh) {
        self.remove_fluid(coord);
        if mesh.is_empty() {
            return;
        }
        let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
        for v in &mesh.vertices {
            let p = Vec3::from(v.position);
            min = min.min(p);
            max = max.max(p);
        }
        let Some((first_vertex, first_index)) = self.store_mesh(mesh) else { return };
        self.fluid_chunks.insert(
            coord,
            GpuChunk {
                first_vertex,
                vertex_count: mesh.vertices.len() as u32,
                first_index,
                index_count: mesh.indices.len() as u32,
                min,
                max,
            },
        );
    }

    pub fn remove_fluid(&mut self, coord: ChunkCoord) {
        if let Some(chunk) = self.fluid_chunks.remove(&coord) {
            self.vertices.free(chunk.first_vertex, chunk.vertex_count);
            self.indices.free(chunk.first_index, chunk.index_count);
        }
    }

    /// Chunk columns with simulated water (the sea plane leaves them to the fluid meshes).
    pub fn set_fluid_columns(&mut self, columns: &[[i32; 2]]) {
        self.far.set_fluid_columns(&self.gpu.queue, columns);
    }

    pub fn remove_chunk(&mut self, coord: ChunkCoord) {
        if let Some(chunk) = self.chunks.remove(&coord) {
            let origin = chunk_origin(coord);
            self.shadow_changed(origin + chunk.min, origin + chunk.max, f32::MAX);
            self.vertices.free(chunk.first_vertex, chunk.vertex_count);
            self.indices.free(chunk.first_index, chunk.index_count);
        }
    }

    /// New samples for one far-terrain level; it switches to the window centred on `to`.
    pub fn far_upload(
        &mut self,
        level: usize,
        to: glam::IVec2,
        rects: &[(clipmap::GridRect, clipmap::Samples)],
    ) {
        self.far.upload(&self.gpu.queue, level, to, rects);
    }

    /// Upload a model for drawing as building parts.
    pub fn add_model(&mut self, model: &models::ModelData) -> ModelId {
        self.parts.add_model(&self.gpu.device, &self.gpu.queue, &self.layouts.material, model)
    }

    /// Every placed part. Call when the set changes.
    pub fn set_parts(&mut self, instances: &[PartInstance]) {
        self.shadow_dirty = [true; CASCADES];
        self.parts.set_instances(instances);
    }

    pub fn parts_count(&self) -> usize {
        self.parts.count()
    }

    /// Add, replace or remove a batch of nature scatter (see [`ScatterBatch`]).
    pub fn set_scatter(&mut self, key: u64, batch: Option<ScatterBatch>) {
        if let Some((min, max, shadow_distance)) = self.parts.set_scatter(key, batch)
            && shadow_distance > 0.0
        {
            self.shadow_changed(min, max, shadow_distance);
        }
    }

    /// Draw scatter instances of `near` as `far` beyond `distance`.
    pub fn set_lods(&mut self, near: ModelId, levels: Vec<(ModelId, f32)>) {
        self.parts.set_lods(near, levels);
    }

    pub fn scatter_count(&self) -> usize {
        self.parts.scatter_count()
    }

    pub fn set_ghost(&mut self, ghost: Option<Ghost>) {
        self.parts.set_ghost(ghost);
    }

    pub fn set_far_enabled(&mut self, enabled: bool) {
        self.far.enabled = enabled;
    }

    pub fn far_enabled(&self) -> bool {
        self.far.enabled
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Turn the sun's shadows on or off (off skips the shadow passes entirely).
    pub fn set_shadows(&mut self, on: bool) {
        self.shadows = on;
        self.shadow_dirty = [true; CASCADES];
    }

    /// How strongly plants move in the wind: 1 normal, 0 still (reduce motion).
    pub fn set_wind(&mut self, strength: f32) {
        self.wind = strength.max(0.0);
    }

    /// Light-space matrix of one cascade for camera-relative positions, snapped to its texels so
    /// shadows don't crawl or shimmer as the camera moves.
    fn shadow_matrix(camera_pos: Vec3, sun: Vec3, radius: f32) -> Mat4 {
        let up = if sun.y.abs() > 0.99 { Vec3::Z } else { Vec3::Y };
        let light_view = Mat4::look_to_rh(Vec3::ZERO, -sun, up);
        let cam_ls = light_view.transform_point3(camera_pos);
        let texel = 2.0 * radius / SHADOW_SIZE as f32;
        let snapped = (cam_ls.truncate() / texel).floor() * texel;
        let shift = Vec3::new(cam_ls.x - snapped.x, cam_ls.y - snapped.y, -(SHADOW_DEPTH * 0.5));
        let proj = Mat4::orthographic_rh(-radius, radius, -radius, radius, 0.0, SHADOW_DEPTH);
        // Rotation of the relative position into light space, then centre on the snapped camera.
        proj * Mat4::from_translation(shift) * light_view
    }

    /// The last TAA debug counts (ASC_DEBUG_TAA): fireflies, sun-bright inputs, NaN/infinite
    /// pixels, and over how many frames.
    pub fn taa_debug_counts(&self) -> Option<[u32; 4]> {
        self.taa_debug.as_ref().and_then(|d| d.last)
    }

    /// Hot pixels by the shader that made them (see `TaaDebug::sources`).
    pub fn taa_debug_sources(&self) -> Option<[u32; 8]> {
        self.taa_debug.as_ref().map(|d| d.sources)
    }

    /// Smoothed GPU milliseconds per stage of the frame, e.g. "shadows 1.20 | terrain 2.10", or
    /// `None` without timestamp support.
    pub fn gpu_stages(&self) -> Option<String> {
        let timer = self.timer.borrow();
        let timer = timer.as_ref()?;
        let parts: Vec<String> = timer.stages().map(|(l, ms)| format!("{l} {ms:.2}")).collect();
        let total: f32 = timer.stages().map(|(_, ms)| ms).sum();
        Some(format!("{} | total {total:.2}", parts.join(" | ")))
    }

    fn mark(&self, encoder: &mut wgpu::CommandEncoder, label: &'static str) {
        if let Some(t) = self.timer.borrow_mut().as_mut() {
            t.mark(encoder, label);
        }
    }

    fn mark_in(&self, pass: &mut wgpu::RenderPass<'_>, label: &'static str) {
        if let Some(t) = self.timer.borrow_mut().as_mut() {
            t.mark_in(pass, label);
        }
    }

    pub fn render(&mut self, frame: &FrameParams, ui: Option<UiFrame<'_>>) -> FrameStats {
        let t_acquire = Instant::now();
        let surface = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t) | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                let (w, h) = self.gpu.size();
                self.resize(w, h);
                return FrameStats::default();
            }
            _ => return FrameStats::default(),
        };
        let (out_width, out_height) = self.gpu.size();
        let aspect = out_width as f32 / out_height as f32;
        // The scene's own resolution (below the window's when upscaling).
        let (width, height) = self.render_size();
        let scale = self.render_scale();
        // The camera snapped to the same 1/1024 m grid as terrain vertices, so chunk offsets and
        // vertex positions add exactly and neighbouring chunks meet without cracks. (A millimetre
        // step is far too small to see.)
        let mut snapped = frame.camera;
        snapped.position = (snapped.position * asc_world::POSITION_GRID).round() / asc_world::POSITION_GRID;
        let cam = &snapped;
        let view_proj = cam.view_proj(aspect);
        // TAA: shift the whole projection by a different sub-pixel offset each frame (Halton 2, 3),
        // so over a few frames every pixel is sampled across its whole area. Upscaling needs more
        // phases (each window pixel needs a sample near it) and the full render pixel's span.
        let jumped = cam.position.distance(self.prev_camera) > TAA_RESET_DISTANCE;
        let taa = self.taa_on;
        let jitter = if taa {
            let phases = (8.0 / (scale * scale)).ceil() as u64;
            let span = if scale < 1.0 { 1.0 } else { TAA_JITTER };
            let i = (self.frame % phases) as u32 + 1;
            glam::Vec2::new(
                (halton(i, 2) - 0.5) * 2.0 * span / width as f32,
                (halton(i, 3) - 0.5) * 2.0 * span / height as f32,
            )
        } else {
            glam::Vec2::ZERO
        };
        let jittered = Mat4::from_cols(
            glam::Vec4::X,
            glam::Vec4::Y,
            glam::Vec4::Z,
            glam::Vec4::new(jitter.x, jitter.y, 0.0, 1.0),
        ) * view_proj;
        // This frame's NDC (unjittered) and depth to last frame's clip space: back to a
        // camera-relative position, shifted by how far the camera moved, then last frame's
        // projection.
        let taa_reproject = self.prev_view_proj
            * Mat4::from_translation(cam.position - self.prev_camera)
            * view_proj.inverse();
        let history_ok = taa && self.taa_valid && !jumped;
        // Shadows hold still (no wind, a stepped sun), so a cascade is only redrawn when the camera
        // has moved a fair way across it or something that casts shadows changed; in between it's
        // looked up through the matrix it was drawn with. When the sun steps, a fresh set of cascades
        // is drawn and the previous set fades out over it.
        let first = self.frame == 0;
        self.frame += 1;
        let sun_stepped = !first && self.shadow_sun.angle_between(frame.sun_dir) > SHADOW_SUN_STEP;
        if first || sun_stepped {
            self.shadow_sun = frame.sun_dir;
        }
        if sun_stepped {
            self.shadow_set ^= 1;
            self.shadow_fade_start = Instant::now();
        }
        self.cascade_update = [false; SHADOW_LAYERS];
        for (c, radius) in CASCADE_RADII.iter().enumerate() {
            let layer = self.shadow_set * CASCADES + c;
            // Following the camera can't sparkle (the sun holds still and the maps snap to their own
            // texels), so redraw often: the near cascade whenever the camera has moved at all, the
            // others after a couple of percent of their width. Lagging further showed as detail
            // popping in around you.
            let follow = if c == 0 { 0.02 } else { radius * 0.02 };
            let moved = (cam.position - self.cascade_state[layer].1).length() > follow;
            self.cascade_update[layer] = first || sun_stepped || moved || self.shadow_dirty[c];
        }
        self.shadow_dirty = [false; CASCADES];
        self.last_camera = cam.position;
        let fade =
            if first { 1.0 } else { (self.shadow_fade_start.elapsed().as_secs_f32() / SHADOW_FADE).min(1.0) };
        for layer in 0..SHADOW_LAYERS {
            if self.cascade_update[layer] {
                self.cascade_state[layer] = (
                    Self::shadow_matrix(cam.position, self.shadow_sun, CASCADE_RADII[layer % CASCADES]),
                    cam.position,
                );
            }
        }
        // A layer drawn from an earlier camera position: shift today's camera-relative positions back.
        let shadow_view_proj =
            self.cascade_state.map(|(m, from)| m * Mat4::from_translation(cam.position - from));
        let current: [Mat4; CASCADES] =
            std::array::from_fn(|c| shadow_view_proj[self.shadow_set * CASCADES + c]);
        let mut stats = self.build_draw_lists(cam.position, view_proj, frame.view_distance, current);
        stats.acquire_ms = t_acquire.elapsed().as_secs_f32() * 1e3;

        let daylight = smoothstep(-0.08, 0.25, frame.sun_dir.y);
        // Open the exposure up only once it's actually getting dark, not while a low sun is still up.
        let night = 1.0 - smoothstep(-0.15, 0.03, frame.sun_dir.y);
        // (Below 1 by day: full sun pushed light colours up into the flat top of the tonemap curve,
        // which read as washed out.)
        let exposure = 0.8 + 1.2 * night;
        let time = self.start.elapsed().as_secs_f32();
        let mut lights: Vec<PointLight> = frame.lights.clone();
        lights.sort_by(|a, b| {
            let da = a.position.distance(cam.position) - a.radius;
            let db = b.position.distance(cam.position) - b.radius;
            da.total_cmp(&db)
        });
        lights.truncate(MAX_LIGHTS);
        let globals = Globals {
            view_proj: jittered.to_cols_array_2d(),
            inv_view_proj: jittered.inverse().to_cols_array_2d(),
            shadow_view_proj: shadow_view_proj.map(|m| m.to_cols_array_2d()),
            camera: cam.position.extend(time).into(),
            sun: frame.sun_dir.extend(daylight).into(),
            fog: [self.fog_density, 0.012, frame.view_distance, exposure],
            screen: [width as f32, height as f32, 1.0 / width as f32, 1.0 / height as f32],
            brush: match frame.brush {
                Some((center, radius)) => (center - cam.position).extend(radius).into(),
                None => Vec4::ZERO.into(),
            },
            far: [if self.far.enabled { FAR_RANGE } else { frame.view_distance }, 0.0, 0.0, 0.0],
            shadow: [0.0, f32::from(u8::from(self.shadows)), self.wind, fade],
            shadow_radii: [CASCADE_RADII[0], CASCADE_RADII[1], CASCADE_RADII[2], self.shadow_set as f32],
            taa_reproject: taa_reproject.to_cols_array_2d(),
            taa: [f32::from(u8::from(history_ok)), TAA_FEEDBACK, jitter.x, jitter.y],
            motion: [
                if self.taa_valid { self.prev_time } else { time },
                if self.taa_valid { self.prev_wind } else { self.wind },
                0.0,
                0.0,
            ],
            output: [out_width as f32, out_height as f32, 1.0 / out_width as f32, 1.0 / out_height as f32],
            upscale: [
                scale,
                SHARPNESS.0 + (SHARPNESS.1 - SHARPNESS.0) * ((1.0 - scale) / 0.5).clamp(0.0, 1.0),
                f32::from(u8::from(self.taa_debug.is_some())),
                self.sun_disc,
            ],
            env: [f32::from(u8::from(frame.underwater)), lights.len() as f32, frame.ash, 0.0],
            light_pos: std::array::from_fn(|i| {
                lights.get(i).map_or([0.0; 4], |l| (l.position - cam.position).extend(l.radius).into())
            }),
            light_color: std::array::from_fn(|i| lights.get(i).map_or([0.0; 4], |l| l.color.extend(0.0).into())),
        };
        let queue = &self.gpu.queue;
        queue.write_buffer(&self.globals, 0, bytemuck::bytes_of(&globals));
        if self.shadows {
            for (layer, buffer) in self.cascade_globals.iter().enumerate() {
                if self.cascade_update[layer] {
                    let g = Globals { shadow: [layer as f32, 1.0, self.wind, fade], ..globals };
                    queue.write_buffer(buffer, 0, bytemuck::bytes_of(&g));
                }
            }
        }
        queue.write_buffer(&self.offsets, 0, bytemuck::cast_slice(&self.offset_data));
        self.far.write_levels(queue, glam::Vec2::new(cam.position.x, cam.position.z), frame.view_distance);
        let mut cascades = [None; CASCADES];
        for (c, cascade) in cascades.iter_mut().enumerate() {
            let layer = self.shadow_set * CASCADES + c;
            if self.shadows && self.cascade_update[layer] {
                *cascade = Some((shadow_view_proj[layer], CASCADE_RADII[c]));
            }
        }
        let pixels_per_radian = height as f32 / (2.0 * (cam.fov_y * 0.5).tan());
        self.parts.prepare(&self.gpu.device, queue, cam.position, view_proj, pixels_per_radian, cascades);
        stats.parts_drawn = self.parts.drawn;
        stats.far_levels = if self.far.enabled { self.far.ready_levels() } else { 0 };
        stats.far_triangles = self.far.triangles();

        let device = &self.gpu.device;
        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        if let Some(t) = self.timer.get_mut().as_mut() {
            t.begin(device);
        }
        self.mark(&mut encoder, "start");
        self.shadow_pass(&mut encoder);
        self.mark(&mut encoder, "shadows");
        self.scene_pass(&mut encoder);
        self.mark(&mut encoder, "resolve");
        if taa {
            let mut pass = color_pass(&mut encoder, "taa", &self.targets.taa[self.taa_index], None);
            pass.set_pipeline(&self.pipelines.taa);
            pass.set_bind_group(0, &self.scene_bg, &[]);
            pass.set_bind_group(1, &self.taa_bgs.pass[self.taa_index], &[]);
            pass.draw(0..3, 0..1);
        }
        self.mark(&mut encoder, "taa");
        self.bloom_passes(&mut encoder, taa);
        self.mark(&mut encoder, "bloom");

        let view = surface.texture.create_view(&wgpu::TextureViewDescriptor::default());
        {
            let mut pass = color_pass(&mut encoder, "post", &view, Some(wgpu::Color::BLACK));
            pass.set_pipeline(&self.pipelines.post);
            pass.set_bind_group(0, &self.scene_bg, &[]);
            pass.set_bind_group(1, if taa { &self.taa_bgs.post[self.taa_index] } else { &self.post_bg }, &[]);
            pass.draw(0..3, 0..1);
        }
        self.mark(&mut encoder, "post");

        let mut extra = Vec::new();
        if let Some(ui) = &ui {
            let screen = egui_wgpu::ScreenDescriptor {
                size_in_pixels: [out_width, out_height],
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
        self.mark(&mut encoder, "ui");
        if let Some(t) = self.timer.get_mut().as_mut() {
            t.finish(&mut encoder);
        }
        let debug_copied = match &mut self.taa_debug {
            Some(d) if d.state.load(std::sync::atomic::Ordering::Acquire) == 0 && d.since.elapsed().as_secs_f32() >= 1.0 => {
                encoder.copy_buffer_to_buffer(&self.taa_counts, 0, &d.readback, 0, 48);
                d.state.store(1, std::sync::atomic::Ordering::Release);
                true
            }
            _ => false,
        };
        let t_submit = Instant::now();
        queue.submit(extra.into_iter().chain([encoder.finish()]));
        if let Some(t) = self.timer.get_mut().as_mut() {
            t.after_submit();
        }
        if let Some(d) = &mut self.taa_debug {
            if debug_copied {
                // Start the next window from zero (runs before the next frame's commands).
                queue.write_buffer(&self.taa_counts, 0, &[0u8; 48]);
                d.frames_in_copy = std::mem::take(&mut d.frames);
                let state = d.state.clone();
                d.readback.slice(..).map_async(wgpu::MapMode::Read, move |r| {
                    state.store(if r.is_ok() { 2 } else { 0 }, std::sync::atomic::Ordering::Release);
                });
                d.since = Instant::now();
            }
            d.frames += 1;
            if d.state.load(std::sync::atomic::Ordering::Acquire) == 2 {
                if let Ok(data) = d.readback.slice(..).get_mapped_range() {
                    let v: &[u32] = bytemuck::cast_slice(&data);
                    d.last = Some([v[0], v[1], v[2], d.frames_in_copy]);
                    d.sources = [v[4], v[5], v[6], v[7], v[8], v[9], v[10], v[11]];
                }
                d.readback.unmap();
                d.state.store(0, std::sync::atomic::Ordering::Release);
            }
        }
        stats.submit_ms = t_submit.elapsed().as_secs_f32() * 1e3;
        if let Some(ui) = &ui {
            for id in &ui.textures.free {
                self.egui.free_texture(id);
            }
        }
        let t_present = Instant::now();
        queue.present(surface);
        stats.present_ms = t_present.elapsed().as_secs_f32() * 1e3;
        // Next frame reprojects into this one.
        self.prev_view_proj = view_proj;
        self.prev_camera = cam.position;
        self.prev_time = time;
        self.prev_wind = self.wind;
        self.taa_valid = taa;
        if taa {
            self.taa_index ^= 1;
        }
        stats
    }

    /// Pick visible and shadow-casting chunks and lay out their camera-relative offsets: visible chunks
    /// first (front to back), then shadow casters. Instance index = position in this array.
    /// Visible chunks (front to back), and the chunks casting into each shadow cascade: those whose
    /// bounds land in the area the cascade covers (`cascades`: camera-relative light matrices).
    fn build_draw_lists(
        &mut self,
        camera_pos: Vec3,
        view_proj: Mat4,
        view_distance: f32,
        cascades: [Mat4; CASCADES],
    ) -> FrameStats {
        let frustum = Frustum::from_view_proj(view_proj);
        self.draw_list.clear();
        self.shadow_list.clear();
        for set in &mut self.shadow_sets {
            set.clear();
        }
        let mut stats = FrameStats::default();
        for (coord, chunk) in &self.chunks {
            let origin = chunk_origin(*coord) - camera_pos;
            let (min, max) = (origin + chunk.min, origin + chunk.max);
            let nearest = Vec3::ZERO.clamp(min, max);
            let dist = nearest.length();
            if self.shadows {
                let (center, extent) = ((min + max) * 0.5, (max - min).length() * 0.5);
                let mut listed = false;
                for (c, m) in cascades.iter().enumerate() {
                    let p = m.transform_point3(center);
                    let r = extent / CASCADE_RADII[c];
                    if p.x.abs() <= 1.0 + r && p.y.abs() <= 1.0 + r {
                        if !listed {
                            self.shadow_list.push(*coord);
                            listed = true;
                        }
                        self.shadow_sets[c].push(self.shadow_list.len() as u32 - 1);
                    }
                }
            }
            // Horizontal distance (as the streamer and the far terrain's hand-over measure it): from
            // high up, the ground below is still in view.
            if glam::Vec2::new(nearest.x, nearest.z).length() < view_distance && frustum.intersects_box(min, max) {
                self.draw_list.push((dist, *coord));
                stats.triangles += chunk.index_count as usize / 3;
            }
        }
        self.draw_list.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
        stats.chunks_drawn = self.draw_list.len();
        stats.shadow_chunks = self.shadow_list.len();

        self.fluid_draw.clear();
        for (coord, chunk) in &self.fluid_chunks {
            let origin = chunk_origin(*coord) - camera_pos;
            let (min, max) = (origin + chunk.min, origin + chunk.max);
            let nearest = Vec3::ZERO.clamp(min, max);
            if glam::Vec2::new(nearest.x, nearest.z).length() < view_distance && frustum.intersects_box(min, max) {
                self.fluid_draw.push(*coord);
            }
        }
        self.offset_data.clear();
        let draws = self.draw_list.iter().map(|(_, c)| *c);
        for coord in draws.chain(self.shadow_list.iter().copied()).chain(self.fluid_draw.iter().copied()) {
            self.offset_data.push((chunk_origin(coord) - camera_pos).extend(0.0).into());
        }
        if self.offset_data.len() > self.offsets_capacity {
            self.offsets_capacity = self.offset_data.len().next_power_of_two();
            self.offsets = Self::offsets_buffer(&self.gpu.device, self.offsets_capacity);
            (self.scene_bg, self.shadow_bgs) = Self::scene_bind_groups(
                &self.gpu.device,
                &self.layouts,
                &self.globals,
                &self.cascade_globals,
                &self.offsets,
                &self.shadow_map,
                &self.shadow_sampler,
            );
        }
        stats
    }

    fn bind_arenas(&self, pass: &mut wgpu::RenderPass<'_>) {
        pass.set_vertex_buffer(0, self.vertices.buffer.slice(..));
        pass.set_index_buffer(self.indices.buffer.slice(..), wgpu::IndexFormat::Uint32);
    }

    fn draw_chunk(&self, pass: &mut wgpu::RenderPass<'_>, coord: ChunkCoord, instance: u32) {
        let c = &self.chunks[&coord];
        pass.draw_indexed(
            c.first_index..c.first_index + c.index_count,
            c.first_vertex as i32,
            instance..instance + 1,
        );
    }

    /// One depth pass per cascade, each drawing only the casters that reach it.
    fn shadow_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        if !self.shadows {
            return;
        }
        for (index, layer) in self.shadow_layers.iter().enumerate() {
            if !self.cascade_update[index] {
                continue;
            }
            let c = index % CASCADES;
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: layer,
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
            pass.set_bind_group(0, &self.shadow_bgs[index], &[]);
            self.bind_arenas(&mut pass);
            let base = self.draw_list.len() as u32;
            for &i in &self.shadow_sets[c] {
                self.draw_chunk(&mut pass, self.shadow_list[i as usize], base + i);
            }
            pass.set_pipeline(&self.pipelines.parts_shadow);
            self.parts.draw_shadows(&mut pass, c);
        }
    }

    fn scene_pass(&self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("scene"),
            color_attachments: &[
                Some(match &self.targets.hdr_msaa {
                    // Multisampled: render into the MSAA target and resolve into the HDR texture.
                    Some(msaa) => wgpu::RenderPassColorAttachment {
                        view: msaa,
                        depth_slice: None,
                        resolve_target: Some(&self.targets.hdr),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Discard,
                        },
                    },
                    None => wgpu::RenderPassColorAttachment {
                        view: &self.targets.hdr,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                            store: wgpu::StoreOp::Store,
                        },
                    },
                }),
                Some(match &self.targets.motion_msaa {
                    Some(msaa) => wgpu::RenderPassColorAttachment {
                        view: msaa,
                        depth_slice: None,
                        resolve_target: Some(&self.targets.motion),
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Discard,
                        },
                    },
                    None => wgpu::RenderPassColorAttachment {
                        view: &self.targets.motion,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                            store: wgpu::StoreOp::Store,
                        },
                    },
                }),
            ],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.targets.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(0.0),
                    // Kept: TAA reprojects from it.
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
            multiview_mask: None,
        });
        pass.set_bind_group(0, &self.scene_bg, &[]);
        self.mark_in(&mut pass, "scene start");
        pass.set_pipeline(&self.pipelines.terrain);
        self.bind_arenas(&mut pass);
        for (i, (_, coord)) in self.draw_list.iter().enumerate() {
            self.draw_chunk(&mut pass, *coord, i as u32);
        }
        self.mark_in(&mut pass, "terrain");
        pass.set_pipeline(&self.pipelines.parts);
        self.parts.draw(&mut pass);
        self.mark_in(&mut pass, "parts");
        // Far terrain after the chunks, so the depth test lets chunks win wherever they exist.
        pass.set_pipeline(&self.pipelines.far);
        self.far.draw(&mut pass);
        self.mark_in(&mut pass, "far");
        pass.set_pipeline(&self.pipelines.sky);
        pass.draw(0..3, 0..1);
        self.mark_in(&mut pass, "sky");
        // Water last: it blends over terrain and sky and doesn't write depth.
        pass.set_pipeline(&self.pipelines.water);
        self.far.draw_water(&mut pass);
        // Then the simulated water, through the terrain arenas (bind group 0 again: the sea used 1).
        if !self.fluid_draw.is_empty() {
            pass.set_bind_group(0, &self.scene_bg, &[]);
            pass.set_pipeline(&self.pipelines.fluid);
            self.bind_arenas(&mut pass);
            let base = (self.draw_list.len() + self.shadow_list.len()) as u32;
            for (i, coord) in self.fluid_draw.iter().enumerate() {
                let c = &self.fluid_chunks[coord];
                pass.draw_indexed(
                    c.first_index..c.first_index + c.index_count,
                    c.first_vertex as i32,
                    base + i as u32..base + i as u32 + 1,
                );
            }
        }
        self.mark_in(&mut pass, "water");
        // The placement ghost blends over everything, including water.
        pass.set_bind_group(0, &self.scene_bg, &[]);
        pass.set_pipeline(&self.pipelines.ghost);
        self.parts.draw_ghost(&mut pass);
    }

    fn bloom_passes(&self, encoder: &mut wgpu::CommandEncoder, taa: bool) {
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
        let source = if taa { &self.taa_bgs.prefilter[self.taa_index] } else { &self.bloom_bgs.prefilter };
        fullscreen(encoder, &levels[0], &p.bloom_prefilter, source, true);
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
