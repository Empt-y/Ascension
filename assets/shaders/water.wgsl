// Natural water (sea, lakes, rivers): the far-terrain grid at each sample's water surface, drawn
// after the sky with alpha blending.
//
// Every level draws its whole grid (near the camera too: chunk terrain hides whatever is under
// it) and leaves the area of a finer drawn level to it. A sample without water drops its vertex
// far down, so the surface ends inside the banks; where a river steps down, the grid between two
// samples makes a falling sheet. Depth (from the far heights) shades it: clear at the shore,
// darker deep. Chunk columns whose water is simulated are left to fluid.wgsl.

struct LevelInfo {
    center: vec2<i32>,
    spacing: f32,
    finer: i32,
    drawn: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(1) @binding(0) var heights: texture_2d_array<f32>;
@group(1) @binding(1) var<storage, read> levels: array<LevelInfo, 7>;
// x of the first entry = count; then (min x, min z, max x, max z) chunk columns where the water
// is simulated and drawn by fluid.wgsl instead.
@group(1) @binding(3) var<storage, read> fluid_columns: array<vec4<i32>, 1025>;
@group(1) @binding(4) var water_levels: texture_2d_array<f32>;

const MESH_HALF: i32 = 126;
const GRID: u32 = 253u;
const TEX: i32 = 256;
// Below this a sample has no water (clipmap::NO_WATER is -10000).
const NONE: f32 = -5000.0;

fn wrap(g: i32) -> i32 {
    return ((g % TEX) + TEX) % TEX;
}

fn load(t: texture_2d_array<f32>, level: u32, g: vec2<i32>) -> f32 {
    return textureLoad(t, vec2(wrap(g.x), wrap(g.y)), i32(level), 0).r;
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
    @location(1) world_xz: vec2<f32>,
    // 1 where the vertex has water, 0 where it was dropped.
    @location(2) wet: f32,
    // Water depth over the natural ground.
    @location(3) depth: f32,
    @location(4) @interpolate(flat) level: u32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) level: u32) -> VsOut {
    let info = levels[level];
    let offset = vec2(i32(vi % GRID), i32(vi / GRID)) - vec2(MESH_HALF);
    let g = info.center + offset;
    let xz = vec2<f32>(g) * info.spacing;
    let ground = load(heights, level, g);
    var surface = load(water_levels, level, g);
    let wet = surface > NONE;
    if !wet {
        // Dropped well under the ground, so the water's edge ends inside the bank.
        surface = ground - 30.0;
    }
    let cam = globals.camera.xyz;
    var out: VsOut;
    out.rel = vec3(xz.x - cam.x, surface - cam.y, xz.y - cam.z);
    out.clip = globals.view_proj * vec4(out.rel, 1.0);
    out.world_xz = xz;
    out.wet = select(0.0, 1.0, wet);
    out.depth = max(surface - ground, 0.0);
    out.level = level;
    return out;
}

fn ripple_normal(xz: vec2<f32>, t: f32, fade: f32) -> vec3<f32> {
    let p1 = vec3(xz * 0.35 + vec2(t * 0.4, t * 0.25), t * 0.3);
    let p2 = vec3(xz * 1.1 - vec2(t * 0.6, -t * 0.2), t * 0.5);
    let e = 0.15;
    let n1 = value_noise(p1);
    let n2 = value_noise(p2);
    let dx = (value_noise(p1 + vec3(e, 0.0, 0.0)) - n1) + (value_noise(p2 + vec3(e, 0.0, 0.0)) - n2) * 0.5;
    let dz = (value_noise(p1 + vec3(0.0, e, 0.0)) - n1) + (value_noise(p2 + vec3(0.0, e, 0.0)) - n2) * 0.5;
    return normalize(vec3(-dx * 0.9 * fade, 1.0, -dz * 0.9 * fade));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Ripple normals take derivatives of nothing here, so the discards can come first.
    if in.wet < 0.5 {
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
    let column = vec2<i32>(floor(in.world_xz / 32.0));
    let count = fluid_columns[0].x;
    for (var i = 1; i <= count; i++) {
        let c = fluid_columns[i];
        if all(column >= c.xy) && all(column <= c.zw) {
            discard;
        }
    }
    let dist = length(in.rel);
    let v = -in.rel / dist;
    // Ripples fade with distance so the far water doesn't shimmer.
    let n = ripple_normal(in.world_xz, globals.camera.w, 1.0 - smoothstep(60.0, 900.0, dist));
    let up = select(n, -n, v.y < 0.0);

    let ndv = clamp(dot(up, v), 0.0, 1.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    let reflected = sky_base(reflect(-v, up));
    let shallow = vec3(0.06, 0.22, 0.22);
    let deep = vec3(0.005, 0.03, 0.06);
    let body = mix(shallow, deep, smoothstep(0.5, 12.0, in.depth)) * (0.25 + 0.75 * daylight());
    var color = mix(body, reflected, fresnel);
    // Sun glint.
    let h = normalize(v + sun_dir());
    color += sun_light() * pow(max(dot(up, h), 0.0), 300.0) * 4.0;

    // Clear at the very edge so shores blend into the sand, more opaque with depth and at grazing angles.
    let alpha = clamp(
        smoothstep(0.0, 1.2, in.depth) * (0.65 + 0.35 * fresnel) + smoothstep(2.0, 10.0, in.depth) * 0.3,
        0.0,
        1.0,
    );
    return vec4(debug_hot(apply_fog(color, in.rel), 6.0), alpha);
}
