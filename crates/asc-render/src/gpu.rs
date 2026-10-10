//! Device, queue and window surface setup.

use std::sync::Arc;

use anyhow::{Context, Result};
use winit::window::Window;

pub struct Gpu {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    pub adapter_info: wgpu::AdapterInfo,
    vsync: bool,
    mailbox_supported: bool,
}

impl Gpu {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
        let surface = instance.create_surface(window.clone()).context("creating window surface")?;

        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: Some(&surface),
            apply_limit_buckets: false,
        }))
        .context("no suitable GPU adapter")?;
        let adapter_info = adapter.get_info();
        log::info!("GPU: {} ({:?})", adapter_info.name, adapter_info.backend);

        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("ascension"),
            required_features: wgpu::Features::empty(),
            required_limits: wgpu::Limits::default(),
            ..Default::default()
        }))
        .context("requesting GPU device")?;

        let size = window.inner_size();
        let caps = surface.get_capabilities(&adapter);
        // Tonemapping writes display-referred values, so an sRGB swapchain does the encoding for us.
        let format =
            caps.formats.iter().copied().find(wgpu::TextureFormat::is_srgb).unwrap_or(caps.formats[0]);
        let mailbox_supported = caps.present_modes.contains(&wgpu::PresentMode::Mailbox);
        let mut config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .context("surface is not supported by the adapter")?;
        config.format = format;
        config.desired_maximum_frame_latency = 2;

        let mut gpu = Self { device, queue, surface, config, adapter_info, vsync: false, mailbox_supported };
        gpu.apply_present_mode();
        Ok(gpu)
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
    }

    pub fn vsync(&self) -> bool {
        self.vsync
    }

    pub fn set_vsync(&mut self, vsync: bool) {
        self.vsync = vsync;
        self.apply_present_mode();
    }

    fn apply_present_mode(&mut self) {
        self.config.present_mode = if self.vsync {
            wgpu::PresentMode::Fifo
        } else if self.mailbox_supported {
            wgpu::PresentMode::Mailbox
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        self.surface.configure(&self.device, &self.config);
    }

    pub fn size(&self) -> (u32, u32) {
        (self.config.width, self.config.height)
    }
}
