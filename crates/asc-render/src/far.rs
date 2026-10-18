//! GPU resources for the far-terrain clipmap: the toroidal height texture array, per-level info and
//! the shared grid index buffer. See [`crate::clipmap`] for the maths and `far.wgsl` for drawing.

use glam::IVec2;
use wgpu::util::DeviceExt;

use crate::clipmap::{self, GridRect, LEVELS, MESH_HALF, TEX};

#[repr(C)]
#[derive(Clone, Copy, Default, bytemuck::Pod, bytemuck::Zeroable)]
struct LevelInfo {
    center: [i32; 2],
    spacing: f32,
    finer: i32,
    drawn: u32,
    pad: [u32; 3],
}

pub struct FarTerrain {
    texture: wgpu::Texture,
    levels: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    pub bind_group: wgpu::BindGroup,
    centers: [Option<IVec2>; LEVELS],
    pub enabled: bool,
}

impl FarTerrain {
    pub fn new(device: &wgpu::Device, layout: &wgpu::BindGroupLayout) -> Self {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("far heights"),
            size: wgpu::Extent3d {
                width: TEX as u32,
                height: TEX as u32,
                depth_or_array_layers: LEVELS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let levels = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("far levels"),
            size: (std::mem::size_of::<LevelInfo>() * LEVELS) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        // One (2*MESH_HALF)^2-quad grid, shared by every level; the vertex shader derives positions
        // from the vertex index.
        let grid = (2 * MESH_HALF + 1) as u32;
        let mut index_data = Vec::with_capacity(((grid - 1) * (grid - 1) * 6) as usize);
        for j in 0..grid - 1 {
            for i in 0..grid - 1 {
                let a = j * grid + i;
                let (b, c, d) = (a + 1, a + grid, a + grid + 1);
                // Alternate the diagonal so the grid has no directional bias.
                if (i + j) % 2 == 0 {
                    index_data.extend_from_slice(&[a, c, b, b, c, d]);
                } else {
                    index_data.extend_from_slice(&[a, c, d, a, d, b]);
                }
            }
        }
        let indices = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("far grid indices"),
            contents: bytemuck::cast_slice(&index_data),
            usage: wgpu::BufferUsages::INDEX,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("far"),
            layout,
            entries: &[
                wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(&view) },
                wgpu::BindGroupEntry { binding: 1, resource: levels.as_entire_binding() },
            ],
        });
        Self {
            texture,
            levels,
            indices,
            index_count: index_data.len() as u32,
            bind_group,
            centers: [None; LEVELS],
            enabled: true,
        }
    }

    /// Write freshly computed samples for `level` and switch it to the window centred on `to`.
    pub fn upload(&mut self, queue: &wgpu::Queue, level: usize, to: IVec2, rects: &[(GridRect, Vec<f32>)]) {
        for (rect, data) in rects {
            for (tex, size, grid) in clipmap::wrapped_pieces(*rect) {
                let piece = clipmap::piece_data(*rect, data, grid, size);
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d { x: tex.x, y: tex.y, z: level as u32 },
                        aspect: wgpu::TextureAspect::All,
                    },
                    bytemuck::cast_slice(&piece),
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(size.x * 4),
                        rows_per_image: Some(size.y),
                    },
                    wgpu::Extent3d { width: size.x, height: size.y, depth_or_array_layers: 1 },
                );
            }
        }
        self.centers[level] = Some(to);
    }

    pub fn ready_levels(&self) -> usize {
        self.centers.iter().filter(|c| c.is_some()).count()
    }

    /// Per-frame level table: centres, and which finer level each one must leave room for.
    pub fn write_levels(&self, queue: &wgpu::Queue) {
        let mut info = [LevelInfo::default(); LEVELS];
        let mut finer = -1;
        for (level, slot) in info.iter_mut().enumerate() {
            let center = self.centers[level].unwrap_or_default();
            *slot = LevelInfo {
                center: center.into(),
                spacing: clipmap::spacing(level),
                finer,
                drawn: u32::from(self.centers[level].is_some()),
                pad: [0; 3],
            };
            if self.centers[level].is_some() {
                finer = level as i32;
            }
        }
        queue.write_buffer(&self.levels, 0, bytemuck::cast_slice(&info));
    }

    pub fn draw(&self, pass: &mut wgpu::RenderPass<'_>) {
        if !self.enabled {
            return;
        }
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.set_index_buffer(self.indices.slice(..), wgpu::IndexFormat::Uint32);
        for level in 0..LEVELS {
            if self.centers[level].is_some() {
                pass.draw_indexed(0..self.index_count, 0, level as u32..level as u32 + 1);
            }
        }
    }

    /// Water samples terrain depth from the height textures, so it shares this bind group.
    pub fn draw_water(&self, pass: &mut wgpu::RenderPass<'_>) {
        if !self.enabled || self.ready_levels() == 0 {
            return;
        }
        pass.set_bind_group(1, &self.bind_group, &[]);
        pass.draw(0..4, 0..1);
    }

    pub fn triangles(&self) -> usize {
        if self.enabled { self.ready_levels() * self.index_count as usize / 3 } else { 0 }
    }
}
