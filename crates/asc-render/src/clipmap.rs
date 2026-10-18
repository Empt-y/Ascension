//! Geometry clipmap bookkeeping for the far terrain (pure maths, no GPU types).
//!
//! Level `L` samples the height field on a grid of spacing `2^(L+1)` metres. Its window is
//! `WINDOW` samples per side centred on an *even* grid index near the camera, so each level's edge
//! lies on grid lines of the next coarser level and the levels nest without cracks. Heights live in
//! a `TEX`×`TEX` texture per level addressed toroidally (`grid index mod TEX`), so when a window moves
//! only the samples that entered it need computing.

use glam::{IVec2, Vec2};

pub const LEVELS: usize = 7;
/// Texture size per level.
pub const TEX: i32 = 256;
/// Samples either side of the centre held in the texture: the mesh uses `MESH_HALF`, plus one for
/// normals and morphing.
pub const SAMPLE_HALF: i32 = 127;
/// Quads either side of the centre in each level's mesh. Even, so the edge is on the coarser grid.
pub const MESH_HALF: i32 = 126;
/// Fraction of a level's half-extent where its heights blend into the coarser level's.
pub const MORPH_START: f32 = 0.8;

const _: () = assert!(2 * SAMPLE_HALF < TEX && MESH_HALF % 2 == 0 && MESH_HALF < SAMPLE_HALF);

pub fn spacing(level: usize) -> f32 {
    (2u32 << level) as f32
}

/// Grid index of a level's window centre for a camera at `xz`: the even index at or below it.
pub fn center_for(level: usize, xz: Vec2) -> IVec2 {
    let s = spacing(level);
    (xz / (2.0 * s)).floor().as_ivec2() * 2
}

/// The window centre a level should use: stay on `current` until the camera has moved more than 1.5
/// snap steps away from it, so a camera resting on a snap boundary doesn't flip the level back and
/// forth every frame. The levels still nest: an offset of up to 3 spacings per level is far inside
/// the coarser level's margin.
pub fn keep_or_recenter(level: usize, xz: Vec2, current: Option<IVec2>) -> IVec2 {
    let s = spacing(level);
    match current {
        Some(c) if (xz - c.as_vec2() * s).abs().max_element() <= 3.0 * s => c,
        _ => center_for(level, xz),
    }
}

/// Half-extent in metres of the area a level's mesh covers.
pub fn mesh_half_extent(level: usize) -> f32 {
    MESH_HALF as f32 * spacing(level)
}

/// An inclusive rectangle of grid indices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GridRect {
    pub min: IVec2,
    pub max: IVec2,
}

impl GridRect {
    pub fn window(center: IVec2) -> Self {
        Self { min: center - SAMPLE_HALF, max: center + SAMPLE_HALF }
    }

    pub fn size(&self) -> IVec2 {
        self.max - self.min + 1
    }

    pub fn len(&self) -> usize {
        let s = self.size();
        (s.x.max(0) * s.y.max(0)) as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn contains(&self, p: IVec2) -> bool {
        p.cmpge(self.min).all() && p.cmple(self.max).all()
    }
}

/// The samples to compute when a window moves from `old` (or from nothing) to `new`: a band of whole
/// columns plus a band of rows, never overlapping.
pub fn dirty(old: Option<IVec2>, new: IVec2) -> Vec<GridRect> {
    let next = GridRect::window(new);
    let Some(old) = old else { return vec![next] };
    let delta = new - old;
    if delta.x.abs() > 2 * SAMPLE_HALF || delta.y.abs() > 2 * SAMPLE_HALF {
        return vec![next];
    }
    let prev = GridRect::window(old);
    let mut rects = Vec::new();
    // Columns that entered on the x side, full height of the new window.
    let (cx0, cx1) = if delta.x > 0 {
        (prev.max.x + 1, next.max.x)
    } else if delta.x < 0 {
        (next.min.x, prev.min.x - 1)
    } else {
        (1, 0)
    };
    if cx0 <= cx1 {
        rects.push(GridRect { min: IVec2::new(cx0, next.min.y), max: IVec2::new(cx1, next.max.y) });
    }
    // Rows that entered on the z side, over the remaining columns.
    let (rz0, rz1) = if delta.y > 0 {
        (prev.max.y + 1, next.max.y)
    } else if delta.y < 0 {
        (next.min.y, prev.min.y - 1)
    } else {
        (1, 0)
    };
    if rz0 <= rz1 {
        let (x0, x1) = match (delta.x > 0, delta.x < 0) {
            (true, _) => (next.min.x, prev.max.x),
            (_, true) => (prev.min.x, next.max.x),
            _ => (next.min.x, next.max.x),
        };
        rects.push(GridRect { min: IVec2::new(x0, rz0), max: IVec2::new(x1, rz1) });
    }
    rects
}

/// Toroidal texel coordinate of a grid index.
pub fn texel(g: i32) -> u32 {
    g.rem_euclid(TEX) as u32
}

/// Pieces of a grid rectangle that are contiguous in the toroidal texture: (texel origin, size, grid
/// origin). At most four, where the rectangle wraps past the texture edge.
pub fn wrapped_pieces(rect: GridRect) -> Vec<(glam::UVec2, glam::UVec2, IVec2)> {
    let split = |lo: i32, hi: i32| -> Vec<(i32, i32)> {
        let mut out = Vec::new();
        let mut g = lo;
        while g <= hi {
            let room = TEX - g.rem_euclid(TEX);
            let end = (g + room - 1).min(hi);
            out.push((g, end));
            g = end + 1;
        }
        out
    };
    let mut pieces = Vec::new();
    for (z0, z1) in split(rect.min.y, rect.max.y) {
        for (x0, x1) in split(rect.min.x, rect.max.x) {
            let tex = glam::UVec2::new(texel(x0), texel(z0));
            let size = glam::UVec2::new((x1 - x0 + 1) as u32, (z1 - z0 + 1) as u32);
            pieces.push((tex, size, IVec2::new(x0, z0)));
        }
    }
    pieces
}

/// The values of one wrapped piece, row by row, cut out of `data` (row-major over `rect`).
/// A rectangle's samples, row-major: heights and the surface material id at each.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Samples {
    pub heights: Vec<f32>,
    pub materials: Vec<u8>,
}

pub fn piece_data<T: Copy>(rect: GridRect, data: &[T], grid: IVec2, size: glam::UVec2) -> Vec<T> {
    let width = rect.size().x as usize;
    let mut out = Vec::with_capacity((size.x * size.y) as usize);
    for row in 0..size.y as i32 {
        let z = (grid.y + row - rect.min.y) as usize;
        let x = (grid.x - rect.min.x) as usize;
        out.extend_from_slice(&data[z * width + x..z * width + x + size.x as usize]);
    }
    out
}

/// How far a vertex `offset` quads from its level's centre has blended into the coarser level:
/// 0 in the interior, rising to exactly 1 at the mesh edge.
pub fn morph(offset: IVec2) -> f32 {
    let t = offset.x.abs().max(offset.y.abs()) as f32 / MESH_HALF as f32;
    ((t - MORPH_START) / (1.0 - MORPH_START)).clamp(0.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn cells(r: &GridRect) -> impl Iterator<Item = IVec2> + '_ {
        (r.min.y..=r.max.y).flat_map(move |z| (r.min.x..=r.max.x).map(move |x| IVec2::new(x, z)))
    }

    #[test]
    fn levels_nest_and_edges_fall_on_the_coarser_grid() {
        let mut rng = 12345u64;
        let mut next = || {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 40_000.0
        };
        for _ in 0..500 {
            let cam = Vec2::new(next(), next());
            for level in 0..LEVELS - 1 {
                let (s, c) = (spacing(level), center_for(level, cam));
                let coarse_c = center_for(level + 1, cam);
                // This level's mesh edges are on multiples of the coarser spacing.
                for edge in [c.x - MESH_HALF, c.x + MESH_HALF, c.y - MESH_HALF, c.y + MESH_HALF] {
                    assert_eq!(edge.rem_euclid(2), 0, "edge {edge} not on the coarse grid");
                }
                // The whole mesh lies inside the coarser level's mesh.
                let lo = (c - MESH_HALF).as_vec2() * s;
                let hi = (c + MESH_HALF).as_vec2() * s;
                let cs = spacing(level + 1);
                let clo = (coarse_c - MESH_HALF).as_vec2() * cs;
                let chi = (coarse_c + MESH_HALF).as_vec2() * cs;
                assert!(lo.cmpge(clo).all() && hi.cmple(chi).all(), "level {level} escapes its parent");
                // And the camera is near the middle of it.
                assert!((cam - c.as_vec2() * s).abs().max_element() <= 2.0 * s);
            }
        }
    }

    #[test]
    fn resting_on_a_boundary_does_not_flip_the_window() {
        let level = 0;
        let edge = Vec2::new(-60.0, -90.0);
        let c = keep_or_recenter(level, edge, None);
        for i in 0..100 {
            let jitter = Vec2::splat(if i % 2 == 0 { 4e-6 } else { -4e-6 });
            assert_eq!(keep_or_recenter(level, edge + jitter, Some(c)), c);
        }
        // Moving well away does re-centre, onto the normal snapped centre.
        let far = edge + Vec2::new(20.0, 0.0);
        assert_eq!(keep_or_recenter(level, far, Some(c)), center_for(level, far));
    }

    #[test]
    fn hysteresis_keeps_levels_nested() {
        for level in 0..LEVELS - 1 {
            let (s, cs) = (spacing(level), spacing(level + 1));
            // Worst case: each level's centre 3 of its spacings from the camera, in opposite directions.
            let lo = -3.0 * s - MESH_HALF as f32 * s;
            let coarse_lo = 3.0 * cs - MESH_HALF as f32 * cs;
            assert!(lo >= coarse_lo, "level {level} could escape its parent");
        }
    }

    #[test]
    fn dirty_regions_are_exactly_the_new_samples() {
        let moves = [
            (IVec2::new(0, 0), IVec2::new(2, 0)),
            (IVec2::new(10, -4), IVec2::new(4, 6)),
            (IVec2::new(-8, 8), IVec2::new(-8, -2)),
            (IVec2::new(0, 0), IVec2::new(300, 2)),
            (IVec2::new(0, 0), IVec2::new(0, 0)),
            (IVec2::new(0, 0), IVec2::new(-254, 254)),
        ];
        for (old, new) in moves {
            let rects = dirty(Some(old), new);
            let mut seen = HashSet::new();
            for r in &rects {
                for c in cells(r) {
                    assert!(seen.insert(c), "{c} listed twice for {old} -> {new}");
                }
            }
            let (prev, next) = (GridRect::window(old), GridRect::window(new));
            let expected: HashSet<IVec2> = cells(&next).filter(|c| !prev.contains(*c)).collect();
            assert_eq!(seen, expected, "move {old} -> {new}");
        }
        assert_eq!(dirty(None, IVec2::ZERO), vec![GridRect::window(IVec2::ZERO)]);
    }

    #[test]
    fn a_window_never_collides_with_itself_in_the_texture() {
        let w = GridRect::window(IVec2::new(-37 * 2, 1001 * 2));
        let mut texels = HashSet::new();
        for c in cells(&w) {
            assert!(texels.insert((texel(c.x), texel(c.y))));
        }
    }

    #[test]
    fn wrapped_pieces_cover_the_rect_in_place() {
        let rect = GridRect { min: IVec2::new(200, -30), max: IVec2::new(300, 10) };
        let pieces = wrapped_pieces(rect);
        assert_eq!(pieces.len(), 4);
        let mut covered = 0;
        for (tex, size, grid) in pieces {
            assert!(tex.x + size.x <= TEX as u32 && tex.y + size.y <= TEX as u32);
            assert_eq!((texel(grid.x), texel(grid.y)), (tex.x, tex.y));
            covered += size.x * size.y;
        }
        assert_eq!(covered as usize, rect.len());
    }

    #[test]
    fn piece_data_lands_each_value_on_its_own_texel() {
        let rect = GridRect { min: IVec2::new(250, -3), max: IVec2::new(262, 4) };
        // Encode each sample's grid position in its value.
        let data: Vec<f32> = cells(&rect).map(|c| (c.x * 1000 + c.y) as f32).collect();
        let mut tex = std::collections::HashMap::new();
        for (t, size, grid) in wrapped_pieces(rect) {
            let piece = piece_data(rect, &data, grid, size);
            for z in 0..size.y {
                for x in 0..size.x {
                    tex.insert((t.x + x, t.y + z), piece[(z * size.x + x) as usize]);
                }
            }
        }
        for c in cells(&rect) {
            assert_eq!(tex[&(texel(c.x), texel(c.y))], (c.x * 1000 + c.y) as f32);
        }
    }

    #[test]
    fn morph_is_zero_inside_and_one_at_the_edge() {
        assert_eq!(morph(IVec2::ZERO), 0.0);
        assert_eq!(morph(IVec2::new(90, -50)), 0.0);
        assert_eq!(morph(IVec2::new(MESH_HALF, 3)), 1.0);
        assert_eq!(morph(IVec2::new(-7, -MESH_HALF)), 1.0);
        let mid = morph(IVec2::new(115, 0));
        assert!(mid > 0.0 && mid < 1.0);
    }
}
