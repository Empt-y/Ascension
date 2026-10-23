// Natural water (sea, lakes, rivers) beyond the loaded chunks: the far-terrain grid at each
// sample's water surface, drawn after the sky with alpha blending. Inside the chunk view radius the
// chunks' own water meshes (fluid.wgsl) take over, as the chunk terrain does from far.wgsl.
//
// Every level draws its whole grid and leaves the area of a finer drawn level to it. A dry sample next to water carries the water's level on, so the surface runs on under the
// bank and the terrain itself (by depth) cuts the shoreline, following the ground rather than the
// grid; samples with no water round them are dropped out of sight. Where a river steps down, the
// grid between two levels makes a falling sheet, which foams. Rivers' flow (per sample) drifts and
// stretches their ripples downstream. Chunk columns whose water is simulated are left to fluid.wgsl.

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
// How far inside the view radius this still draws (as far.wgsl: chunks cover at least that far).
const CHUNK_OVERLAP: f32 = 24.0;

fn wrap(g: i32) -> i32 {
    return ((g % TEX) + TEX) % TEX;
}

fn load(t: texture_2d_array<f32>, level: u32, g: vec2<i32>) -> f32 {
    return textureLoad(t, vec2(wrap(g.x), wrap(g.y)), i32(level), 0).r;
}

fn water_at(level: u32, g: vec2<i32>) -> vec4<f32> {
    return textureLoad(water_levels, vec2(wrap(g.x), wrap(g.y)), i32(level), 0);
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
    @location(1) world_xz: vec2<f32>,
    // 1 where the vertex has (or carries on) water, 0 where it was dropped.
    @location(2) wet: f32,
    // Water depth over the natural ground.
    @location(3) depth: f32,
    @location(4) @interpolate(flat) level: u32,
    // The river's flow (x, z) and how steeply the surface falls here (waterfalls).
    @location(5) flow: vec2<f32>,
    @location(6) fall: f32,
    // 1 where the surface is lava (a volcano's crater or pool).
    @location(7) lava: f32,
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32, @builtin(instance_index) level: u32) -> VsOut {
    let info = levels[level];
    let offset = vec2(i32(vi % GRID), i32(vi / GRID)) - vec2(MESH_HALF);
    let g = info.center + offset;
    let xz = vec2<f32>(g) * info.spacing;
    let ground = load(heights, level, g);
    let here = water_at(level, g);
    var surface = here.x;
    var flow = here.yz;
    var lava = here.w;
    var wet = surface > NONE;
    // The surface's steepest fall to a neighbour, for foam on the steps.
    var fall = 0.0;
    // The lowest neighbouring water (carrying on a higher terrace would flood the land below it).
    var lowest = 1e9;
    for (var dz = -1; dz <= 1; dz++) {
        for (var dx = -1; dx <= 1; dx++) {
            let n = water_at(level, g + vec2(dx, dz));
            if n.x > NONE {
                lowest = min(lowest, n.x);
                if wet {
                    fall = max(fall, abs(n.x - surface) / info.spacing);
                }
                if !wet {
                    flow = n.yz;
                    lava = max(lava, n.w);
                }
            }
        }
    }
    if !wet && lowest < 1e8 {
        // Carry the neighbouring water on under the bank: the terrain cuts the shore.
        surface = lowest;
        wet = true;
    } else if !wet {
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
    out.flow = flow;
    out.fall = fall;
    out.lava = lava;
    return out;
}

// Ripples; on a river they're carried downstream. (Round, not stretched: streaks lying along the
// flow look still as they slide along themselves, and any sideways part of the motion is then all
// the eye sees: the river seemed to run from bank to bank.)
fn ripple_normal(xz_in: vec2<f32>, t: f32, fade: f32, flow: vec2<f32>) -> vec3<f32> {
    let speed = length(flow);
    var xz = xz_in;
    if speed > 0.01 {
        let along = flow / speed;
        let across = vec2(-along.y, along.x);
        // Into flow-aligned coordinates, carried downstream (only: the idle drift of still water has
        // a sideways part).
        xz = vec2(dot(xz_in, along) - t * speed * 0.9, dot(xz_in, across));
        let q1 = vec3(xz * 0.35, t * 0.3);
        let q2 = vec3(xz * 1.1 - vec2(t * speed * 0.6, 0.0), t * 0.5);
        // The slopes come out along and across the flow: back into x and z.
        let g = ripple_slope(q1, q2);
        let w = along * g.x + across * g.y;
        return normalize(vec3(-w.x * 0.9 * fade, 1.0, -w.y * 0.9 * fade));
    }
    let p1 = vec3(xz * 0.35 + vec2(t * 0.4, t * 0.25), t * 0.3);
    let p2 = vec3(xz * 1.1 - vec2(t * 0.6, -t * 0.2), t * 0.5);
    let g = ripple_slope(p1, p2);
    return normalize(vec3(-g.x * 0.9 * fade, 1.0, -g.y * 0.9 * fade));
}

// The ripple field's slope (along its first and second coordinates).
fn ripple_slope(p1: vec3<f32>, p2: vec3<f32>) -> vec2<f32> {
    let e = 0.15;
    let n1 = value_noise(p1);
    let n2 = value_noise(p2);
    let dx = (value_noise(p1 + vec3(e, 0.0, 0.0)) - n1) + (value_noise(p2 + vec3(e, 0.0, 0.0)) - n2) * 0.5;
    let dz = (value_noise(p1 + vec3(0.0, e, 0.0)) - n1) + (value_noise(p2 + vec3(0.0, e, 0.0)) - n2) * 0.5;
    return vec2(dx, dz);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // Ripple normals take derivatives of nothing here, so the discards can come first.
    if in.wet < 0.01 || length(in.rel.xz) < globals.fog.z - CHUNK_OVERLAP {
        discard;
    }
    let finer = levels[in.level].finer;
    if finer >= 0 {
        let f = levels[u32(finer)];
        // Up to the finer grid's very edge: water is see-through, so a strip drawn by both would
        // show as a dark line (terrain, being opaque, can overlap; far.wgsl keeps a margin).
        let half = f32(MESH_HALF) * f.spacing - 0.01;
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
    if in.lava > 0.5 {
        // Lava: opaque and glowing (it blooms), as fluid.wgsl draws it near by.
        let world = vec3(in.world_xz.x, in.rel.y + globals.camera.y, in.world_xz.y);
        return vec4(debug_hot(apply_fog(lava_color(world, globals.camera.w), in.rel), 7.0), 1.0);
    }
    let dist = length(in.rel);
    let v = -in.rel / dist;
    // Ripples fade with distance so the far water doesn't shimmer.
    let n = ripple_normal(in.world_xz, globals.camera.w, 1.0 - smoothstep(60.0, 900.0, dist), in.flow);
    let up = select(n, -n, v.y < 0.0);

    let ndv = clamp(dot(up, v), 0.0, 1.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    let reflected = sky_base(reflect(-v, up));
    let shallow = vec3(0.03, 0.13, 0.15);
    let deep = vec3(0.004, 0.025, 0.05);
    let body = mix(shallow, deep, smoothstep(0.5, 12.0, in.depth)) * (0.25 + 0.75 * daylight());
    var color = mix(body, reflected, fresnel);
    // Sun glint.
    let h = normalize(v + sun_dir());
    color += sun_light() * pow(max(dot(up, h), 0.0), 300.0) * 4.0;
    // White water where the river falls (the steps) and a little on fast stretches, in streaks
    // sliding downstream.
    let speed = length(in.flow);
    let steep = smoothstep(0.1, 0.45, in.fall);
    var foam = steep;
    if speed > 0.01 {
        let along = in.flow / speed;
        let s = dot(in.world_xz, along) - globals.camera.w * (2.0 + 4.0 * steep);
        let streaks = value_noise(vec3(s * 0.8, dot(in.world_xz, vec2(-along.y, along.x)) * 2.5, globals.camera.w * 0.2));
        foam = max(steep * (0.55 + 0.45 * streaks), smoothstep(0.75, 0.95, streaks) * smoothstep(0.8, 1.6, speed) * 0.35);
    }
    let white = vec3(0.8, 0.85, 0.88) * (0.25 + 0.75 * daylight()) * (0.6 + 0.4 * dot(sun_dir(), vec3(0.0, 1.0, 0.0)));
    color = mix(color, white, foam);

    // Clear at the very edge so shores blend into the sand, more opaque with depth and at grazing angles.
    // (Depth comes from the coarse far heights, so it's only rough near shores: keep a floor of
    // opacity, or shallow patches turned see-through and the sand glared through in blotches.)
    let alpha = clamp(
        0.72 * smoothstep(0.0, 0.3, in.depth) + 0.2 * smoothstep(0.5, 4.0, in.depth) + 0.25 * fresnel + foam * 0.5,
        0.0,
        0.97,
    );
    return vec4(debug_hot(apply_fog(color, in.rel), 6.0), alpha);
}
