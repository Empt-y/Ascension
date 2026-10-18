//! Bind group layouts (fixed for the renderer's lifetime) and pipelines (rebuilt on shader reload).

use anyhow::{Result, bail};

use crate::shaders::ShaderLib;
use crate::targets::{DEPTH_FORMAT, HDR_FORMAT};

pub struct Layouts {
    /// Globals, chunk offsets, shadow map and its comparison sampler.
    pub scene: wgpu::BindGroupLayout,
    /// Globals and chunk offsets only, for rendering into the shadow map.
    pub shadow: wgpu::BindGroupLayout,
    /// Two filterable textures and a linear sampler, for post-processing passes.
    pub textures: wgpu::BindGroupLayout,
    /// Far terrain: the clipmap height array and the per-level table.
    pub far: wgpu::BindGroupLayout,
}

fn uniform_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn storage_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: true },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn texture_entry(binding: u32, sample_type: wgpu::TextureSampleType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Texture {
            sample_type,
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn sampler_entry(binding: u32, kind: wgpu::SamplerBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Sampler(kind),
        count: None,
    }
}

impl Layouts {
    pub fn new(device: &wgpu::Device) -> Self {
        let float = wgpu::TextureSampleType::Float { filterable: true };
        Self {
            scene: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("scene"),
                entries: &[
                    uniform_entry(0),
                    storage_entry(1),
                    texture_entry(2, wgpu::TextureSampleType::Depth),
                    sampler_entry(3, wgpu::SamplerBindingType::Comparison),
                ],
            }),
            shadow: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow"),
                entries: &[uniform_entry(0), storage_entry(1)],
            }),
            far: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("far"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: true },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            }),
            textures: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("textures"),
                entries: &[
                    texture_entry(0, float),
                    texture_entry(1, float),
                    sampler_entry(2, wgpu::SamplerBindingType::Filtering),
                ],
            }),
        }
    }
}

pub struct Pipelines {
    pub terrain: wgpu::RenderPipeline,
    pub shadow: wgpu::RenderPipeline,
    pub sky: wgpu::RenderPipeline,
    pub bloom_prefilter: wgpu::RenderPipeline,
    pub bloom_down: wgpu::RenderPipeline,
    pub bloom_up: wgpu::RenderPipeline,
    pub post: wgpu::RenderPipeline,
    pub far: wgpu::RenderPipeline,
    pub water: wgpu::RenderPipeline,
}

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 4] =
    wgpu::vertex_attr_array![0 => Float32x3, 1 => Snorm8x4, 2 => Unorm8x4, 3 => Unorm8x4];

struct Builder<'a> {
    device: &'a wgpu::Device,
    samples: u32,
}

impl Builder<'_> {
    fn layout(&self, groups: &[&wgpu::BindGroupLayout]) -> wgpu::PipelineLayout {
        let groups: Vec<Option<&wgpu::BindGroupLayout>> = groups.iter().copied().map(Some).collect();
        self.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &groups,
            immediate_size: 0,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn pipeline(
        &self,
        label: &str,
        layout: &wgpu::PipelineLayout,
        module: &wgpu::ShaderModule,
        (vs, fs): (&str, Option<&str>),
        vertex_buffers: &[Option<wgpu::VertexBufferLayout<'_>>],
        primitive: wgpu::PrimitiveState,
        depth_stencil: Option<wgpu::DepthStencilState>,
        target: Option<wgpu::ColorTargetState>,
    ) -> wgpu::RenderPipeline {
        let targets = [target];
        self.device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some(label),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module,
                entry_point: Some(vs),
                compilation_options: Default::default(),
                buffers: vertex_buffers,
            },
            primitive,
            depth_stencil,
            multisample: wgpu::MultisampleState { count: self.samples, ..Default::default() },
            fragment: fs.map(|fs| wgpu::FragmentState {
                module,
                entry_point: Some(fs),
                compilation_options: Default::default(),
                targets: &targets,
            }),
            multiview_mask: None,
            cache: None,
        })
    }
}

fn replace(format: wgpu::TextureFormat) -> Option<wgpu::ColorTargetState> {
    Some(wgpu::ColorTargetState { format, blend: None, write_mask: wgpu::ColorWrites::ALL })
}

impl Pipelines {
    pub fn build(
        device: &wgpu::Device,
        shaders: &ShaderLib,
        layouts: &Layouts,
        surface_format: wgpu::TextureFormat,
        msaa: u32,
    ) -> Result<Self> {
        let terrain_src = shaders.module(device, "terrain")?;
        let sky_src = shaders.module(device, "sky")?;
        let bloom_src = shaders.module(device, "bloom")?;
        let post_src = shaders.module(device, "post")?;
        let far_src = shaders.module(device, "far")?;
        let water_src = shaders.module(device, "water")?;

        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let b = Builder { device, samples: 1 };
        // Everything drawn in the HDR scene pass shares its sample count.
        let scene_builder = Builder { device, samples: msaa };
        let scene = b.layout(&[&layouts.scene]);
        let shadow_layout = b.layout(&[&layouts.shadow]);
        let textures = b.layout(&[&layouts.scene, &layouts.textures]);
        let far_layout = b.layout(&[&layouts.scene, &layouts.far]);
        let vertex = [Some(wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<asc_world::Vertex>() as u64,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VERTEX_ATTRIBUTES,
        })];
        let back_culled = wgpu::PrimitiveState { cull_mode: Some(wgpu::Face::Back), ..Default::default() };
        let fullscreen = wgpu::PrimitiveState::default();

        let terrain = scene_builder.pipeline(
            "terrain",
            &scene,
            &terrain_src,
            ("vs_main", Some("fs_main")),
            &vertex,
            back_culled,
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            replace(HDR_FORMAT),
        );
        let shadow = b.pipeline(
            "shadow",
            &shadow_layout,
            &terrain_src,
            ("vs_shadow", None),
            &vertex,
            // Both faces cast so thin overhangs and cave roofs still shadow.
            wgpu::PrimitiveState::default(),
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                bias: wgpu::DepthBiasState { constant: 2, slope_scale: 2.0, clamp: 0.0 },
            }),
            None,
        );
        let sky = scene_builder.pipeline(
            "sky",
            &scene,
            &sky_src,
            ("vs_main", Some("fs_main")),
            &[],
            fullscreen,
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Equal),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            replace(HDR_FORMAT),
        );
        let bloom = |label, fs, blend| {
            b.pipeline(
                label,
                &textures,
                &bloom_src,
                ("vs_main", Some(fs)),
                &[],
                fullscreen,
                None,
                Some(wgpu::ColorTargetState {
                    format: HDR_FORMAT,
                    blend,
                    write_mask: wgpu::ColorWrites::ALL,
                }),
            )
        };
        let additive = wgpu::BlendState {
            color: wgpu::BlendComponent {
                src_factor: wgpu::BlendFactor::One,
                dst_factor: wgpu::BlendFactor::One,
                operation: wgpu::BlendOperation::Add,
            },
            alpha: wgpu::BlendComponent::REPLACE,
        };
        let bloom_prefilter = bloom("bloom prefilter", "fs_prefilter", None);
        let bloom_down = bloom("bloom down", "fs_down", None);
        let bloom_up = bloom("bloom up", "fs_up", Some(additive));
        let post = b.pipeline(
            "post",
            &textures,
            &post_src,
            ("vs_main", Some("fs_main")),
            &[],
            fullscreen,
            None,
            replace(surface_format),
        );
        let far = scene_builder.pipeline(
            "far terrain",
            &far_layout,
            &far_src,
            ("vs_main", Some("fs_main")),
            &[],
            // Seen from above almost always; skipping culling avoids caring about grid winding.
            wgpu::PrimitiveState::default(),
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            replace(HDR_FORMAT),
        );
        let water = scene_builder.pipeline(
            "water",
            &far_layout,
            &water_src,
            ("vs_main", Some("fs_main")),
            &[],
            wgpu::PrimitiveState { topology: wgpu::PrimitiveTopology::TriangleStrip, ..Default::default() },
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(false),
                depth_compare: Some(wgpu::CompareFunction::Greater),
                stencil: Default::default(),
                bias: Default::default(),
            }),
            Some(wgpu::ColorTargetState {
                format: HDR_FORMAT,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            }),
        );
        if let Some(err) = pollster::block_on(scope.pop()) {
            bail!("building pipelines: {err}");
        }
        Ok(Self { terrain, shadow, sky, bloom_prefilter, bloom_down, bloom_up, post, far, water })
    }
}
