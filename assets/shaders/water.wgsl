// Lakes and seas: one flat surface at sea level, drawn after the sky with alpha blending.
//
// Water depth comes from the far-terrain height textures (natural terrain, before edits): shallow
// water fades out at the shore, deep water darkens, and dry land (even a pit dug below sea level)
// gets no water at all.

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

const SEA_LEVEL: f32 = 2.0;
const TEX: i32 = 256;
const SAMPLE_HALF: i32 = 127;

fn wrap(g: i32) -> i32 {
    return ((g % TEX) + TEX) % TEX;
}

fn height_at(level: u32, g: vec2<i32>) -> f32 {
    return textureLoad(heights, vec2(wrap(g.x), wrap(g.y)), i32(level), 0).r;
}

// Natural terrain height at a point from the finest ready level that covers it; +1000 if none does.
fn terrain_height(xz: vec2<f32>) -> f32 {
    for (var level = 0u; level < 7u; level++) {
        let info = levels[level];
        if info.drawn == 0u {
            continue;
        }
        let g = xz / info.spacing;
        let g0 = vec2<i32>(floor(g));
        if any(abs(g0 - info.center) >= vec2(SAMPLE_HALF)) {
            continue;
        }
        let f = g - floor(g);
        let h00 = height_at(level, g0);
        let h10 = height_at(level, g0 + vec2(1, 0));
        let h01 = height_at(level, g0 + vec2(0, 1));
        let h11 = height_at(level, g0 + vec2(1, 1));
        return mix(mix(h00, h10, f.x), mix(h01, h11, f.x), f.y);
    }
    return 1000.0;
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    // A quad spanning the far range around the camera.
    let corner = vec2(f32(index & 1u), f32((index >> 1u) & 1u)) * 2.0 - 1.0;
    let r = globals.far.x;
    var out: VsOut;
    out.rel = vec3(corner.x * r, SEA_LEVEL - globals.camera.y, corner.y * r);
    out.clip = globals.view_proj * vec4(out.rel, 1.0);
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
    let world = in.rel + globals.camera.xyz;
    let depth = SEA_LEVEL - terrain_height(world.xz);
    if depth <= 0.0 {
        discard;
    }
    let column = vec2<i32>(floor(world.xz / 32.0));
    let count = fluid_columns[0].x;
    for (var i = 1; i <= count; i++) {
        let c = fluid_columns[i];
        if all(column >= c.xy) && all(column <= c.zw) {
            discard;
        }
    }
    let dist = length(in.rel);
    let v = -in.rel / dist;
    // Ripples fade with distance so the far sea doesn't shimmer.
    let n = ripple_normal(world.xz, globals.camera.w, 1.0 - smoothstep(60.0, 900.0, dist));
    let up = select(n, -n, v.y < 0.0);

    let ndv = clamp(dot(up, v), 0.0, 1.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    let reflected = sky_base(reflect(-v, up));
    let shallow = vec3(0.06, 0.22, 0.22);
    let deep = vec3(0.005, 0.03, 0.06);
    let body = mix(shallow, deep, smoothstep(0.5, 12.0, depth)) * (0.25 + 0.75 * daylight());
    var color = mix(body, reflected, fresnel);
    // Sun glint.
    let h = normalize(v + sun_dir());
    color += sun_light() * pow(max(dot(up, h), 0.0), 300.0) * 4.0;

    // Clear at the very edge so shores blend into the sand, more opaque with depth and at grazing angles.
    let alpha = clamp(smoothstep(0.0, 1.2, depth) * (0.65 + 0.35 * fresnel) + smoothstep(2.0, 10.0, depth) * 0.3, 0.0, 1.0);
    return vec4(debug_hot(apply_fog(color, in.rel), 6.0), alpha);
}
