//! Ascension's renderer: terrain, sky, fog and post-processing on top of wgpu.

mod gpu;

use std::sync::Arc;

use anyhow::Result;
use winit::window::Window;

pub use gpu::Gpu;

pub struct Renderer {
    gpu: Gpu,
}

impl Renderer {
    pub fn new(window: Arc<Window>) -> Result<Self> {
        Ok(Self { gpu: Gpu::new(window)? })
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        self.gpu.resize(width, height);
    }

    pub fn gpu(&self) -> &Gpu {
        &self.gpu
    }

    pub fn render(&mut self) {
        let frame = match self.gpu.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(frame) => frame,
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => frame,
            wgpu::CurrentSurfaceTexture::Outdated | wgpu::CurrentSurfaceTexture::Lost => {
                let (w, h) = self.gpu.size();
                self.gpu.resize(w, h);
                return;
            }
            _ => return,
        };
        let view = frame.texture.create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder =
            self.gpu.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });
        {
            let _pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("clear"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color { r: 0.25, g: 0.45, b: 0.75, a: 1.0 }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
        }
        self.gpu.queue.submit([encoder.finish()]);
        self.gpu.queue.present(frame);
    }
}
