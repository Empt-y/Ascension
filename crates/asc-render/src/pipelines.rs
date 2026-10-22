//! Bind group layouts (fixed for the renderer's lifetime) and pipelines (rebuilt on shader reload).

use anyhow::{Result, bail};

use crate::shaders::ShaderLib;
use crate::targets::{DEPTH_FORMAT, HDR_FORMAT, MOTION_FORMAT};

pub struct Layouts {
    /// Globals, chunk offsets, shadow map and its comparison sampler.
    pub scene: wgpu::BindGroupLayout,
    /// Globals and chunk offsets only, for rendering into the shadow map.
    pub shadow: wgpu::BindGroupLayout,
    /// Two filterable textures and a linear sampler, for post-processing passes.
    pub textures: wgpu::BindGroupLayout,
    /// Far terrain: the clipmap height array and the per-level table.
    pub far: wgpu::BindGroupLayout,
    /// One model material: its uniform, plus the shared model texture array and sampler.
    pub material: wgpu::BindGroupLayout,
    /// TAA: this frame, the history, a linear sampler and the scene depth (multisampled or not).
    pub taa_ms: wgpu::BindGroupLayout,
    pub taa_ss: wgpu::BindGroupLayout,
}

fn taa_layout(device: &wgpu::Device, multisampled: bool) -> wgpu::BindGroupLayout {
    let float = wgpu::TextureSampleType::Float { filterable: true };
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("taa"),
        entries: &[
            texture_entry(0, float),
            texture_entry(1, float),
            sampler_entry(2, wgpu::SamplerBindingType::Filtering),
            texture_entry(4, float),
            // Debug counters (ASC_DEBUG_TAA).
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: false },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Texture {
                    sample_type: wgpu::TextureSampleType::Depth,
                    view_dimension: wgpu::TextureViewDimension::D2,
                    multisampled,
                },
                count: None,
            },
        ],
    })
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        // Vertex too: foliage looks its shadows up per vertex.
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Depth,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
                        visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
                        count: None,
                    },
                ],
            }),
            shadow: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("shadow"),
                entries: &[uniform_entry(0), storage_entry(1)],
            }),
            material: device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("part material"),
                entries: &[
                    uniform_entry(0),
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: float,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
                        },
                        count: None,
                    },
                    sampler_entry(2, wgpu::SamplerBindingType::Filtering),
                ],
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Uint,
                            view_dimension: wgpu::TextureViewDimension::D2Array,
                            multisampled: false,
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
            taa_ms: taa_layout(device, true),
            taa_ss: taa_layout(device, false),
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
    pub parts: wgpu::RenderPipeline,
    pub parts_shadow: wgpu::RenderPipeline,
    pub ghost: wgpu::RenderPipeline,
    pub taa: wgpu::RenderPipeline,
}

const VERTEX_ATTRIBUTES: [wgpu::VertexAttribute; 6] = wgpu::vertex_attr_array![
    0 => Float32x3, 1 => Snorm8x4, 2 => Unorm8x4, 3 => Unorm8x4, 4 => Unorm8x4, 5 => Unorm8x4
];

struct Builder<'a> {
    device: &'a wgpu::Device,
    samples: u32,
    /// Turn shader alpha into MSAA coverage (only meaningful with more than one sample).
    alpha_to_coverage: bool,
    /// Scene pass pipelines also have the motion target: `Some(writes)`, empty for those whose
    /// pixels only move with the camera (TAA reprojects those from depth).
    motion: Option<wgpu::ColorWrites>,
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
        let mut targets: Vec<Option<wgpu::ColorTargetState>> = target.into_iter().map(Some).collect();
        if let Some(write_mask) = self.motion {
            targets.push(Some(wgpu::ColorTargetState { format: MOTION_FORMAT, blend: None, write_mask }));
        }
        // parts.wgsl's `ALPHA_TO_COVERAGE` override defaults to false; other shaders don't declare it.
        let constants = [("ALPHA_TO_COVERAGE", 1.0)];
        let compilation_options = wgpu::PipelineCompilationOptions {
            constants: if self.alpha_to_coverage { &constants } else { &[] },
            ..Default::default()
        };
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
            multisample: wgpu::MultisampleState {
                count: self.samples,
                alpha_to_coverage_enabled: self.alpha_to_coverage,
                ..Default::default()
            },
            fragment: fs.map(|fs| wgpu::FragmentState {
                module,
                entry_point: Some(fs),
                compilation_options: compilation_options.clone(),
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
        let parts_src = shaders.module(device, "parts")?;
        let depth_type = if msaa > 1 { "texture_depth_multisampled_2d" } else { "texture_depth_2d" };
        let taa_src = shaders.module_with(device, "taa", &[("DEPTH_TEXTURE", depth_type)])?;

        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let b = Builder { device, samples: 1, alpha_to_coverage: false, motion: None };
        // Everything drawn in the HDR scene pass shares its sample count and targets.
        let scene_builder = Builder {
            device,
            samples: msaa,
            alpha_to_coverage: false,
            motion: Some(wgpu::ColorWrites::empty()),
        };
        // Models (leaf cards) turn their cut-out alpha into coverage for smooth edges under MSAA,
        // and write their wind motion.
        let models_builder = Builder {
            device,
            samples: msaa,
            alpha_to_coverage: msaa > 1,
            motion: Some(wgpu::ColorWrites::ALL),
        };
        let scene = b.layout(&[&layouts.scene]);
        let shadow_layout = b.layout(&[&layouts.shadow]);
        let textures = b.layout(&[&layouts.scene, &layouts.textures]);
        let far_layout = b.layout(&[&layouts.scene, &layouts.far]);
        let parts_layout = b.layout(&[&layouts.scene, &layouts.material]);
        let parts_shadow_layout = b.layout(&[&layouts.shadow, &layouts.material]);
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
        let taa_layout =
            b.layout(&[&layouts.scene, if msaa > 1 { &layouts.taa_ms } else { &layouts.taa_ss }]);
        let taa = b.pipeline(
            "taa",
            &taa_layout,
            &taa_src,
            ("vs_main", Some("fs_main")),
            &[],
            fullscreen,
            None,
            replace(HDR_FORMAT),
        );
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
        let part_buffers = crate::parts::vertex_layouts();
        let parts = models_builder.pipeline(
            "parts",
            &parts_layout,
            &parts_src,
            ("vs_main", Some("fs_main")),
            &part_buffers,
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
        let parts_shadow = b.pipeline(
            "parts shadow",
            &parts_shadow_layout,
            &parts_src,
            ("vs_shadow", Some("fs_shadow")),
            &part_buffers,
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
        let ghost = scene_builder.pipeline(
            "ghost",
            &parts_layout,
            &parts_src,
            ("vs_main", Some("fs_ghost")),
            &part_buffers,
            wgpu::PrimitiveState::default(),
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
        Ok(Self {
            terrain,
            shadow,
            sky,
            bloom_prefilter,
            bloom_down,
            bloom_up,
            post,
            far,
            water,
            parts,
            parts_shadow,
            ghost,
            taa,
        })
    }
}
