//! Size-dependent render targets: HDR colour, depth and the bloom chain. Rebuilt on resize.

pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const BLOOM_LEVELS: usize = 6;

pub struct Targets {
    /// Single-sample HDR colour: the scene resolves into it and post-processing reads it.
    pub hdr: wgpu::TextureView,
    /// Multisampled HDR colour the scene renders into, when MSAA is on.
    pub hdr_msaa: Option<wgpu::TextureView>,
    pub depth: wgpu::TextureView,
    /// Half resolution and down; `bloom[0]` holds the final blurred result.
    pub bloom: Vec<wgpu::TextureView>,
    /// Temporal anti-aliasing history, ping-ponged: each frame reads one and writes the other.
    pub taa: [wgpu::TextureView; 2],
}

fn target(
    device: &wgpu::Device,
    label: &str,
    (width, height): (u32, u32),
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::TextureView {
    target_ms(device, label, (width, height), format, usage, 1)
}

fn target_ms(
    device: &wgpu::Device,
    label: &str,
    (width, height): (u32, u32),
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
    sample_count: u32,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

impl Targets {
    pub fn new(device: &wgpu::Device, (width, height): (u32, u32), samples: u32) -> Self {
        use wgpu::TextureUsages as U;
        let hdr =
            target(device, "hdr", (width, height), HDR_FORMAT, U::RENDER_ATTACHMENT | U::TEXTURE_BINDING);
        let hdr_msaa = (samples > 1).then(|| {
            target_ms(device, "hdr msaa", (width, height), HDR_FORMAT, U::RENDER_ATTACHMENT, samples)
        });
        // Sampled too: TAA reprojects from it.
        let depth = target_ms(
            device,
            "depth",
            (width, height),
            DEPTH_FORMAT,
            U::RENDER_ATTACHMENT | U::TEXTURE_BINDING,
            samples,
        );
        let taa = [0, 1].map(|_| {
            target(
                device,
                "taa history",
                (width, height),
                HDR_FORMAT,
                U::RENDER_ATTACHMENT | U::TEXTURE_BINDING,
            )
        });
        let bloom = (0..BLOOM_LEVELS)
            .map(|i| {
                let size = ((width >> (i + 1)).max(1), (height >> (i + 1)).max(1));
                target(device, "bloom", size, HDR_FORMAT, U::RENDER_ATTACHMENT | U::TEXTURE_BINDING)
            })
            .collect();
        Self { hdr, hdr_msaa, depth, bloom, taa }
    }
}

pub fn placeholder_texture(device: &wgpu::Device) -> wgpu::TextureView {
    target(device, "placeholder", (1, 1), HDR_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING)
}

pub const SHADOW_SIZE: u32 = 2048;
/// Sun shadow cascades, nearest first.
pub const CASCADES: usize = 3;
/// Two sets of cascades: the current one, and the previous one it is fading in over.
pub const SHADOW_LAYERS: usize = 2 * CASCADES;

/// The cascaded shadow maps (two sets): an array view for sampling and one view per layer to render
/// into. Layer = set * CASCADES + cascade.
pub fn shadow_maps(device: &wgpu::Device) -> (wgpu::TextureView, Vec<wgpu::TextureView>) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow cascades"),
        size: wgpu::Extent3d {
            width: SHADOW_SIZE,
            height: SHADOW_SIZE,
            depth_or_array_layers: SHADOW_LAYERS as u32,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let array = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let layers = (0..SHADOW_LAYERS as u32)
        .map(|layer| {
            texture.create_view(&wgpu::TextureViewDescriptor {
                dimension: Some(wgpu::TextureViewDimension::D2),
                base_array_layer: layer,
                array_layer_count: Some(1),
                ..Default::default()
            })
        })
        .collect();
    (array, layers)
}
