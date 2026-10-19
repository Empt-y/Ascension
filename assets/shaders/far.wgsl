// Far terrain: a geometry clipmap of the height field, drawn after the near chunks.
//
// Each instance is one level: a (2*MESH_HALF+1)^2 vertex grid around the level's centre, heights read
// from that level's layer of a toroidal texture array. The outer band of every level morphs into the
// next coarser level so edges meet exactly; a level discards where a finer level is drawn, and
// everything near the camera is left to the full-detail chunks.

struct LevelInfo {
    center: vec2<i32>,
    spacing: f32,
    // Nearest finer level that is drawn, or -1.
    finer: i32,
    // 1 if this level's texture is ready.
    drawn: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(1) @binding(0) var heights: texture_2d_array<f32>;
@group(1) @binding(1) var<storage, read> levels: array<LevelInfo, 7>;
// Surface material id per sample (asc_world::Material), from worldgen's own rules.
@group(1) @binding(2) var materials: texture_2d_array<u32>;

const LEVEL_COUNT: u32 = 7u;
const MESH_HALF: i32 = 126;
const GRID: u32 = 253u;
const TEX: i32 = 256;
const MORPH_START: f32 = 0.8;
// The far surface sits this far below the true height inside the chunk radius, so chunks always win.
const SINK: f32 = 0.4;
// Inside this distance short of the chunk radius only chunks are drawn.
const CHUNK_OVERLAP: f32 = 24.0;

fn wrap(g: i32) -> i32 {
    return ((g % TEX) + TEX) % TEX;
}

fn height_at(level: u32, g: vec2<i32>) -> f32 {
    return textureLoad(heights, vec2(wrap(g.x), wrap(g.y)), i32(level), 0).r;
}

fn height_bilinear(level: u32, xz: vec2<f32>) -> f32 {
    let g = xz / levels[level].spacing;
    let g0 = vec2<i32>(floor(g));
    let f = g - floor(g);
    let h00 = height_at(level, g0);
    let h10 = height_at(level, g0 + vec2(1, 0));
    let h01 = height_at(level, g0 + vec2(0, 1));
    let h11 = height_at(level, g0 + vec2(1, 1));
    return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
}

// dh/dx, dh/dz at a grid point.
fn gradient_at(level: u32, g: vec2<i32>) -> vec2<f32> {
    let s = levels[level].spacing;
    let dx = height_at(level, g + vec2(1, 0)) - height_at(level, g - vec2(1, 0));
    let dz = height_at(level, g + vec2(0, 1)) - height_at(level, g - vec2(0, 1));
    return vec2(dx, dz) / (2.0 * s);
}

struct VsOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) world_xz: vec2<f32>,
    @location(3) @interpolate(flat) level: u32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) level: u32) -> VsOut {
    let info = levels[level];
    let offset = vec2(i32(vi % GRID), i32(vi / GRID)) - vec2(MESH_HALF);
    let g = info.center + offset;
    let xz = vec2<f32>(g) * info.spacing;
    var h = height_at(level, g);
    var grad = gradient_at(level, g);

    // Blend the outer band into the coarser level; at the very edge this lands exactly on its surface.
    let t = f32(max(abs(offset.x), abs(offset.y))) / f32(MESH_HALF);
    let m = clamp((t - MORPH_START) / (1.0 - MORPH_START), 0.0, 1.0);
    if m > 0.0 && level + 1u < LEVEL_COUNT && levels[level + 1u].drawn == 1u {
        let coarse = level + 1u;
        h = mix(h, height_bilinear(coarse, xz), m);
        let cg = vec2<i32>(round(xz / levels[coarse].spacing));
        grad = mix(grad, gradient_at(coarse, cg), m);
    }

    let cam = globals.camera.xyz;
    let dist = length(xz - cam.xz);
    h -= SINK * (1.0 - smoothstep(globals.fog.z, globals.fog.z + 64.0, dist));

    var out: VsOut;
    out.rel = vec3(xz.x - cam.x, h - cam.y, xz.y - cam.z);
    out.clip = globals.view_proj * vec4(out.rel, 1.0);
    out.normal = normalize(vec3(-grad.x, 1.0, -grad.y));
    out.world_xz = xz;
    out.level = level;
    return out;
}

// Terrain material weights at a point: the bilinear blend of the four surrounding samples' materials,
// in the slot layout terrain_surface expects (slot = material id - 1).
fn far_weights(level: u32, xz: vec2<f32>) -> array<vec4<f32>, 3> {
    let g = xz / levels[level].spacing;
    let g0 = vec2<i32>(floor(g));
    let f = g - floor(g);
    var w: array<f32, 12>;
    for (var k = 0; k < 4; k++) {
        let o = vec2(k & 1, k >> 1u);
        let p = g0 + o;
        let id = textureLoad(materials, vec2(wrap(p.x), wrap(p.y)), i32(level), 0).r;
        let fx = select(1.0 - f.x, f.x, o.x == 1);
        let fy = select(1.0 - f.y, f.y, o.y == 1);
        if id >= 1u && id <= 12u {
            w[id - 1u] += fx * fy;
        }
    }
    return array(vec4(w[0], w[1], w[2], w[3]), vec4(w[4], w[5], w[6], w[7]), vec4(w[8], w[9], w[10], w[11]));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Detail takes derivatives, so it has to come before the discards below.
    let geometric = normalize(in.normal);
    let world = in.rel + globals.camera.xyz;
    let detail = surface_detail(world, geometric);
    if length(in.rel.xz) < globals.fog.z - CHUNK_OVERLAP {
        discard;
    }
    let finer = levels[in.level].finer;
    if finer >= 0 {
        let f = levels[u32(finer)];
        let half = (f32(MESH_HALF) - 0.5) * f.spacing;
        let c = vec2<f32>(f.center) * f.spacing;
        if all(abs(in.world_xz - c) < vec2(half)) {
            discard;
        }
    }
    let w = far_weights(in.level, in.world_xz);
    var s = terrain_surface(w[0], w[1], w[2], world, geometric, detail.tone);
    s.roughness = aa_roughness(s.roughness, detail.widen);
    let color = shade_terrain(s, detail.normal, in.rel, 1.0, 1.0);
    return vec4(apply_fog(color, in.rel), 1.0);
}
