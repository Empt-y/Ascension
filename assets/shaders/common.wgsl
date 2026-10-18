// Shared by every shader: frame globals, noise, sky colour and fog.
// The renderer prepends this file to each shader before compiling it.

struct Globals {
    // Camera-relative: world positions are offset by -camera before projection.
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    shadow_view_proj: mat4x4<f32>,
    // xyz = camera world position, w = seconds since start.
    camera: vec4<f32>,
    // xyz = direction towards the sun, w = 0..1 how far above the horizon it is.
    sun: vec4<f32>,
    // x = fog density, y = fog height falloff, z = view distance (fade to sky), w = exposure.
    fog: vec4<f32>,
    // xy = viewport size in pixels, zw = 1 / size.
    screen: vec4<f32>,
    // xyz = brush centre (camera-relative), w = brush radius (0 = hidden).
    brush: vec4<f32>,
    // x = distance at which everything has faded into the sky (far terrain range, or the chunk radius
    // when far terrain is off).
    far: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;

const PI: f32 = 3.14159265;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hash of an integer lattice point to [0, 1).
fn hash3i(p: vec3<i32>) -> f32 {
    let h = pcg(bitcast<u32>(p.x) ^ pcg(bitcast<u32>(p.y) ^ pcg(bitcast<u32>(p.z))));
    return f32(h >> 8u) / 16777216.0;
}

fn hash13(p: vec3<f32>) -> f32 {
    return hash3i(vec3<i32>(floor(p)));
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    // Quintic fade hides the lattice better than smoothstep.
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = mix(hash3i(i + vec3(0, 0, 0)), hash3i(i + vec3(1, 0, 0)), u.x);
    let b = mix(hash3i(i + vec3(0, 1, 0)), hash3i(i + vec3(1, 1, 0)), u.x);
    let c = mix(hash3i(i + vec3(0, 0, 1)), hash3i(i + vec3(1, 0, 1)), u.x);
    let d = mix(hash3i(i + vec3(0, 1, 1)), hash3i(i + vec3(1, 1, 1)), u.x);
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}

fn fbm3(p: vec3<f32>) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0; i < 3; i++) {
        sum += value_noise(q) * amp;
        q = q * 2.03 + vec3(1.7, 9.2, 4.1);
        amp *= 0.5;
    }
    return sum / 0.875;
}

fn sun_dir() -> vec3<f32> {
    return globals.sun.xyz;
}

// 0 at night, 1 in full daylight.
fn daylight() -> f32 {
    return smoothstep(-0.08, 0.25, sun_dir().y);
}

// Peaks while the sun is near the horizon.
fn sunset_amount() -> f32 {
    let s = sun_dir().y;
    return smoothstep(-0.25, 0.0, s) * (1.0 - smoothstep(0.0, 0.4, s));
}

// Colour of direct sunlight reaching the ground (HDR).
fn sun_light() -> vec3<f32> {
    let warm = vec3(1.0, 0.55, 0.28);
    let white = vec3(1.0, 0.95, 0.86);
    let c = mix(white, warm, sunset_amount());
    return c * 3.2 * smoothstep(-0.05, 0.12, sun_dir().y);
}

// Stylised sky: gradient from horizon to zenith, warm glow around the sun at dusk, no sun disk.
fn sky_base(dir: vec3<f32>) -> vec3<f32> {
    let day = daylight();
    let dusk = sunset_amount();
    let zenith = mix(vec3(0.004, 0.006, 0.02), vec3(0.10, 0.26, 0.75), day);
    var horizon = mix(vec3(0.015, 0.02, 0.05), vec3(0.55, 0.72, 0.95), day);
    horizon = mix(horizon, vec3(1.0, 0.45, 0.22), dusk * 0.85);
    let up = max(dir.y, 0.0);
    var col = mix(horizon, zenith, pow(up, 0.45));
    // Below the horizon fade to a dim haze so the far edge of the world dissolves.
    col = mix(col, horizon * 0.6, smoothstep(0.0, -0.25, dir.y));
    let mu = max(dot(dir, sun_dir()), 0.0);
    let glow_col = mix(vec3(1.0, 0.85, 0.6), vec3(1.0, 0.4, 0.15), dusk);
    col += glow_col * (pow(mu, 8.0) * 0.35 + pow(mu, 64.0) * 0.8) * (day * 0.6 + dusk);
    return col;
}

// Fraction of light lost to fog along a camera-relative ray (analytic exponential height fog).
fn fog_amount(rel: vec3<f32>) -> f32 {
    let dist = length(rel);
    let dir = rel / max(dist, 1e-4);
    let density = globals.fog.x;
    let falloff = globals.fog.y;
    let cam_y = globals.camera.y;
    let fy = falloff * dir.y * dist;
    var integral = dist;
    if abs(fy) > 1e-4 {
        integral = (1.0 - exp(-fy)) / (falloff * dir.y);
    }
    let optical = density * exp(-falloff * max(cam_y, -50.0)) * integral;
    return 1.0 - exp(-max(optical, 0.0));
}

// Blend a surface colour toward the sky along the view ray, and fully into it at the far range so
// the edge of the world never shows.
fn apply_fog(color: vec3<f32>, rel: vec3<f32>) -> vec3<f32> {
    let dir = normalize(rel);
    let fog_col = sky_base(dir);
    let f = fog_amount(rel);
    let horizontal = length(rel.xz);
    let edge = smoothstep(globals.far.x * 0.75, globals.far.x * 0.97, horizontal);
    return mix(color, fog_col, max(f, edge));
}

// ---- Terrain materials and lighting ----

struct Surface {
    albedo: vec3<f32>,
    roughness: f32,
}

fn triplanar(p: vec3<f32>, w: vec3<f32>) -> f32 {
    return value_noise(p.yzx) * w.x + value_noise(p.xzy) * w.y + value_noise(p) * w.z;
}

// Base colours are linear and deliberately saturated. Materials blend by the interpolated weights.
fn terrain_surface(wa: vec4<f32>, wb: vec4<f32>, world: vec3<f32>, n: vec3<f32>, dist: f32) -> Surface {
    // Triplanar detail: blend three axis-aligned noise projections by the normal.
    var tw = pow(abs(n), vec3(4.0));
    tw /= (tw.x + tw.y + tw.z);
    // The finest octave fades out with distance before it can alias.
    let near = 1.0 - smoothstep(8.0, 45.0, dist);
    let grit = (triplanar(world * 14.0 + 5.0, tw) - 0.5) * near;
    let fine = triplanar(world * 1.3, tw) * 0.6 + triplanar(world * 4.1 + 17.0, tw) * 0.4 + grit * 0.45;
    let broad = fbm3(world * 0.035);
    let detail = 0.74 + 0.52 * fine;

    let soil = mix(vec3(0.22, 0.13, 0.07), vec3(0.30, 0.19, 0.10), broad);
    let lush = mix(vec3(0.10, 0.34, 0.06), vec3(0.24, 0.42, 0.05), broad);
    // Grass gives way to soil on slopes.
    let grass = mix(soil, lush, smoothstep(0.62, 0.82, n.y));
    let rock = mix(vec3(0.24, 0.23, 0.24), vec3(0.36, 0.32, 0.29), broad);
    let sand = mix(vec3(0.34, 0.28, 0.18), vec3(0.40, 0.33, 0.22), broad);
    let snow = vec3(0.85, 0.88, 0.95);
    let fleck = smoothstep(0.72, 0.8, fine);
    let copper = mix(vec3(0.28, 0.25, 0.24), vec3(0.85, 0.42, 0.18), fleck);

    let total = max(dot(wa, vec4(1.0)) + dot(wb, vec4(1.0)), 1e-4);
    var s: Surface;
    s.albedo = (grass * wa.x + soil * wa.y + rock * wa.z + sand * wa.w + snow * wb.x + copper * wb.y) / total;
    s.roughness = (0.9 * (wa.x + wa.y + wa.w) + 0.75 * wa.z + 0.5 * wb.x + mix(0.8, 0.3, fleck) * wb.y) / total;
    s.albedo *= detail;
    return s;
}

fn ggx_specular(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let h = normalize(v + l);
    let a = roughness * roughness;
    let a2 = a * a;
    let ndh = max(dot(n, h), 0.0);
    let d = a2 / (PI * pow(ndh * ndh * (a2 - 1.0) + 1.0, 2.0));
    let ndl = max(dot(n, l), 0.0);
    let ndv = max(dot(n, v), 1e-3);
    let k = a * 0.5;
    let g = (ndl / (ndl * (1.0 - k) + k)) * (ndv / (ndv * (1.0 - k) + k));
    let fresnel = 0.04 + 0.96 * pow(1.0 - max(dot(h, v), 0.0), 5.0);
    return d * g * fresnel / max(4.0 * ndl * ndv, 1e-3) * ndl;
}

// Sun, sky ambient and moonlight on a terrain surface. Shared by near chunks and far terrain so the
// two match where they meet.
fn shade_terrain(s: Surface, n: vec3<f32>, rel: vec3<f32>, occlusion: f32, shadow: f32) -> vec3<f32> {
    let l = sun_dir();
    let v = normalize(-rel);
    let ndl = max(dot(n, l), 0.0);
    let sun = sun_light() * shadow;
    // Hemisphere ambient from the sky above and bounced light from the ground below.
    let sky_col = sky_base(normalize(vec3(n.x, max(n.y, 0.2), n.z)));
    // Desaturate the sky light a little so shade isn't too blue.
    let sky_amb = mix(vec3(dot(sky_col, vec3(0.2126, 0.7152, 0.0722))), sky_col, 0.6) * 0.8;
    let ground_amb = vec3(0.10, 0.08, 0.06) * (0.15 + 0.85 * daylight());
    let ambient = mix(ground_amb, sky_amb, n.y * 0.5 + 0.5) * occlusion;
    // Faint, unshadowed moonlight from opposite the sun so nights aren't black.
    let moon = vec3(0.05, 0.065, 0.12) * (1.0 - daylight()) * max(dot(n, -l), 0.0);
    var color = s.albedo * (sun * ndl + ambient + moon);
    color += sun * ggx_specular(n, v, l, s.roughness);
    return color;
}

// Full-screen triangle from the vertex index.
fn fullscreen_pos(index: u32) -> vec4<f32> {
    let uv = vec2(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
