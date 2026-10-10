//! Size-dependent render targets: HDR colour, depth and the bloom chain. Rebuilt on resize.

pub const HDR_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
pub const BLOOM_LEVELS: usize = 6;

pub struct Targets {
    pub hdr: wgpu::TextureView,
    pub depth: wgpu::TextureView,
    /// Half resolution and down; `bloom[0]` holds the final blurred result.
    pub bloom: Vec<wgpu::TextureView>,
}

fn target(
    device: &wgpu::Device,
    label: &str,
    (width, height): (u32, u32),
    format: wgpu::TextureFormat,
    usage: wgpu::TextureUsages,
) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d { width: width.max(1), height: height.max(1), depth_or_array_layers: 1 },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage,
            view_formats: &[],
        })
        .create_view(&wgpu::TextureViewDescriptor::default())
}

impl Targets {
    pub fn new(device: &wgpu::Device, (width, height): (u32, u32)) -> Self {
        use wgpu::TextureUsages as U;
        let hdr =
            target(device, "hdr", (width, height), HDR_FORMAT, U::RENDER_ATTACHMENT | U::TEXTURE_BINDING);
        let depth = target(device, "depth", (width, height), DEPTH_FORMAT, U::RENDER_ATTACHMENT);
        let bloom = (0..BLOOM_LEVELS)
            .map(|i| {
                let size = ((width >> (i + 1)).max(1), (height >> (i + 1)).max(1));
                target(device, "bloom", size, HDR_FORMAT, U::RENDER_ATTACHMENT | U::TEXTURE_BINDING)
            })
            .collect();
        Self { hdr, depth, bloom }
    }
}

pub fn placeholder_texture(device: &wgpu::Device) -> wgpu::TextureView {
    target(device, "placeholder", (1, 1), HDR_FORMAT, wgpu::TextureUsages::TEXTURE_BINDING)
}

pub const SHADOW_SIZE: u32 = 2048;

pub fn shadow_map(device: &wgpu::Device) -> wgpu::TextureView {
    use wgpu::TextureUsages as U;
    target(
        device,
        "shadow map",
        (SHADOW_SIZE, SHADOW_SIZE),
        DEPTH_FORMAT,
        U::RENDER_ATTACHMENT | U::TEXTURE_BINDING,
    )
}
