//! GPU resources for the far-terrain clipmap: the toroidal height texture array, per-level info and
//! the shared grid index buffer. See [`crate::clipmap`] for the maths and `far.wgsl` for drawing.

use glam::{IVec2, Vec2};
use wgpu::util::DeviceExt;

use crate::clipmap::{self, GridRect, LEVELS, MESH_HALF, Samples, TEX};

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
    /// Surface material id per sample (R8Uint), alongside the heights.
    materials: wgpu::Texture,
    levels: wgpu::Buffer,
    indices: wgpu::Buffer,
    /// Index offset where each ring of quads starts; the last entry is the total.
    ring_starts: Vec<u32>,
    /// Per level, the first ring drawn this frame (the rings inside it are hidden).
    first_ring: [usize; LEVELS],
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
        let materials = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("far materials"),
            size: wgpu::Extent3d {
                width: TEX as u32,
                height: TEX as u32,
                depth_or_array_layers: LEVELS as u32,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let material_view = materials.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2Array),
            ..Default::default()
        });
        let levels = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("far levels"),
            size: (std::mem::size_of::<LevelInfo>() * LEVELS) as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let (index_data, ring_starts) = ring_indices();
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
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&material_view),
                },
            ],
        });
        Self {
            texture,
            materials,
            levels,
            indices,
            ring_starts,
            first_ring: [0; LEVELS],
            bind_group,
            centers: [None; LEVELS],
            enabled: true,
        }
    }

    /// Write freshly computed samples for `level` and switch it to the window centred on `to`.
    pub fn upload(&mut self, queue: &wgpu::Queue, level: usize, to: IVec2, rects: &[(GridRect, Samples)]) {
        for (rect, data) in rects {
            for (tex, size, grid) in clipmap::wrapped_pieces(*rect) {
                let heights = clipmap::piece_data(*rect, &data.heights, grid, size);
                let materials = clipmap::piece_data(*rect, &data.materials, grid, size);
                let at = |texture| wgpu::TexelCopyTextureInfo {
                    texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d { x: tex.x, y: tex.y, z: level as u32 },
                    aspect: wgpu::TextureAspect::All,
                };
                let extent = wgpu::Extent3d { width: size.x, height: size.y, depth_or_array_layers: 1 };
                let layout = |bytes: u32| wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(size.x * bytes),
                    rows_per_image: Some(size.y),
                };
                queue.write_texture(at(&self.texture), bytemuck::cast_slice(&heights), layout(4), extent);
                queue.write_texture(at(&self.materials), &materials, layout(1), extent);
            }
        }
        self.centers[level] = Some(to);
    }

    pub fn ready_levels(&self) -> usize {
        self.centers.iter().filter(|c| c.is_some()).count()
    }

    /// Per-frame level table: centres, and which finer level each one must leave room for. Also
    /// works out each level's hole: the middle rings hidden under a finer level or under the near
    /// chunks (`chunk_radius` round the camera), which then aren't drawn at all.
    pub fn write_levels(&mut self, queue: &wgpu::Queue, camera_xz: Vec2, chunk_radius: f32) {
        for level in 0..LEVELS {
            let Some(center) = self.centers[level] else { continue };
            let finer = (0..level).rev().find_map(|f| self.centers[f].map(|c| (f, c)));
            self.first_ring[level] = hidden_rings(level, center, camera_xz, finer, chunk_radius);
        }
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
        let end = *self.ring_starts.last().unwrap();
        for level in 0..LEVELS {
            let first = self.ring_starts[self.first_ring[level]];
            if self.centers[level].is_some() && first < end {
                pass.draw_indexed(first..end, 0, level as u32..level as u32 + 1);
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
        if !self.enabled {
            return 0;
        }
        let end = *self.ring_starts.last().unwrap();
        (0..LEVELS)
            .filter(|&l| self.centers[l].is_some())
            .map(|l| (end - self.ring_starts[self.first_ring[l]]) as usize / 3)
            .sum()
    }
}

/// Matches `CHUNK_OVERLAP` in far.wgsl: far terrain is discarded inside the chunk radius less this.
const CHUNK_OVERLAP: f32 = 24.0;

/// A quad's ring: its Chebyshev distance from the grid centre in quads (0 for the middle four).
fn ring_of(i: i32, j: i32) -> usize {
    let q = |v: i32| {
        let o = v - MESH_HALF;
        if o >= 0 { o } else { -o - 1 }
    };
    q(i).max(q(j)) as usize
}

/// One (2*MESH_HALF)^2-quad grid shared by every level, its quads sorted by ring from the centre
/// out, so a level skips its hidden middle by starting part-way into the buffer. The vertex shader
/// derives positions from the vertex index. Returns the indices and each ring's start offset
/// (`MESH_HALF + 1` entries, the last being the total).
fn ring_indices() -> (Vec<u32>, Vec<u32>) {
    let grid = 2 * MESH_HALF + 1;
    let mut rings: Vec<Vec<u32>> = vec![Vec::new(); MESH_HALF as usize];
    for j in 0..grid - 1 {
        for i in 0..grid - 1 {
            let a = (j * grid + i) as u32;
            let (b, c, d) = (a + 1, a + grid as u32, a + grid as u32 + 1);
            // Alternate the diagonal so the grid has no directional bias.
            let quad = if (i + j) % 2 == 0 { [a, c, b, b, c, d] } else { [a, c, d, a, d, b] };
            rings[ring_of(i, j)].extend_from_slice(&quad);
        }
    }
    let mut starts = vec![0u32];
    let mut indices = Vec::with_capacity((4 * MESH_HALF * MESH_HALF * 6) as usize);
    for ring in rings {
        indices.extend(ring);
        starts.push(indices.len() as u32);
    }
    (indices, starts)
}

/// How many middle rings of a level are hidden, either under `finer` (the nearest finer level drawn,
/// and its centre) or inside the circle the near chunks cover. Conservative: a ring only counts if
/// all of it is hidden, and the shader's discards still trim the exact boundary.
fn hidden_rings(
    level: usize,
    center: IVec2,
    camera_xz: Vec2,
    finer: Option<(usize, IVec2)>,
    chunk_radius: f32,
) -> usize {
    let s = clipmap::spacing(level);
    let c = center.as_vec2() * s;
    // Skipping n rings leaves out the square of half-width n quads round the centre.
    let mut hidden = 0.0f32;
    if let Some((f, fc)) = finer {
        let fs = clipmap::spacing(f);
        // The finer level covers this square (the shader discards a half-quad inside it).
        let half = (MESH_HALF as f32 - 0.5) * fs;
        let offset = (fc.as_vec2() * fs - c).abs().max_element();
        hidden = hidden.max(((half - offset) / s).floor());
    }
    let r = chunk_radius - CHUNK_OVERLAP;
    if r > 0.0 {
        // The square is inside the chunk circle if its far corner is: the largest h with
        // (h + a)^2 + (h + b)^2 <= r^2 for the camera's offset (a, b) from the centre.
        let off = (camera_xz - c).abs();
        let (a, b) = (off.x, off.y);
        let disc = 2.0 * r * r - (a - b) * (a - b);
        if disc > 0.0 {
            let h = (-(a + b) + disc.sqrt()) / 2.0;
            hidden = hidden.max((h / s).floor());
        }
    }
    (hidden.max(0.0) as usize).min(MESH_HALF as usize)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rings_cover_the_grid_in_order() {
        let (indices, starts) = ring_indices();
        let quads = (4 * MESH_HALF * MESH_HALF) as usize;
        assert_eq!(indices.len(), quads * 6);
        assert_eq!(starts.len(), MESH_HALF as usize + 1);
        assert_eq!(*starts.last().unwrap() as usize, indices.len());
        // Ring r is a square annulus of (2r + 2)^2 - (2r)^2 quads.
        for r in 0..MESH_HALF as usize {
            let n = (starts[r + 1] - starts[r]) as usize / 6;
            assert_eq!(n, (2 * r + 2).pow(2) - (2 * r).pow(2), "ring {r}");
        }
    }

    #[test]
    fn chunk_hole_stays_inside_the_chunk_radius() {
        let s = clipmap::spacing(0);
        for cam in [Vec2::ZERO, Vec2::new(5.9, -5.9), Vec2::new(-6.0, 3.0), Vec2::new(103.0, 2.0)] {
            let center = clipmap::keep_or_recenter(0, cam, Some(clipmap::center_for(0, Vec2::ZERO)));
            let n = hidden_rings(0, center, cam, None, 320.0);
            let h = n as f32 * s;
            let c = center.as_vec2() * s;
            for corner in [Vec2::new(h, h), Vec2::new(-h, h), Vec2::new(h, -h), Vec2::new(-h, -h)] {
                assert!((c + corner - cam).length() <= 296.0 + 1e-3, "cam {cam}: {n} rings");
            }
            assert!(n > 90, "level 0 should skip most of its middle, skipped {n}");
        }
    }

    #[test]
    fn finer_hole_stays_inside_the_finer_level() {
        // Level 1 centred at grid (10, -4) = (40, -16) m; level 0 two of its own quads off centre.
        let center = IVec2::new(10, -4);
        let fine = center * 2 + IVec2::new(2, -2);
        let n = hidden_rings(1, center, Vec2::new(40.0, -16.0), Some((0, fine)), 0.0);
        let s = clipmap::spacing(1);
        let fine_half = (MESH_HALF as f32 - 0.5) * clipmap::spacing(0);
        let offset = (fine.as_vec2() * clipmap::spacing(0) - center.as_vec2() * s).abs().max_element();
        assert!(n as f32 * s + offset <= fine_half, "{n} rings");
        assert!(n >= 60, "{n}");
    }
}
