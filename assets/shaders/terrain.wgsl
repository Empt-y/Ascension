// Terrain: stylised PBR-lite with triplanar procedural detail, sun shadows and aerial fog.

// Camera-relative origin of each drawn chunk, indexed by instance.
@group(0) @binding(1) var<storage, read> chunk_offsets: array<vec4<f32>>;
@group(0) @binding(2) var shadow_map: texture_depth_2d;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;

struct VsIn {
    @location(0) position: vec3<f32>,
    // xyz normal, w ambient occlusion (1 = open sky).
    @location(1) normal: vec4<f32>,
    // Blend weights for material slots 0..3 and 4..7 (material id = slot + 1).
    @location(2) weights_a: vec4<f32>,
    @location(3) weights_b: vec4<f32>,
    @builtin(instance_index) instance: u32,
}

struct VsOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) weights_a: vec4<f32>,
    @location(3) weights_b: vec4<f32>,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let rel = in.position + chunk_offsets[in.instance].xyz;
    var out: VsOut;
    out.clip = globals.view_proj * vec4(rel, 1.0);
    out.rel = rel;
    out.normal = in.normal;
    out.weights_a = in.weights_a;
    out.weights_b = in.weights_b;
    return out;
}

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
    let sand = mix(vec3(0.72, 0.60, 0.38), vec3(0.80, 0.70, 0.46), broad);
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

fn shadow_factor(rel: vec3<f32>, n: vec3<f32>) -> f32 {
    // Normal offset hides acne on slopes facing away from the sun.
    let p = globals.shadow_view_proj * vec4(rel + n * 0.35, 1.0);
    let uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    if any(uv < vec2(0.0)) || any(uv > vec2(1.0)) || p.z > 1.0 {
        return 1.0;
    }
    let texel = 1.0 / f32(textureDimensions(shadow_map).x);
    var lit = 0.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            lit += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + vec2(f32(x), f32(y)) * texel, p.z - 0.0015);
        }
    }
    return lit / 9.0;
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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal.xyz);
    let occlusion = clamp(in.normal.w, 0.0, 1.0);
    let world = in.rel + globals.camera.xyz;
    let s = terrain_surface(in.weights_a, in.weights_b, world, n, length(in.rel));
    let l = sun_dir();
    let v = normalize(-in.rel);

    let ndl = max(dot(n, l), 0.0);
    var shadow = 1.0;
    if ndl > 0.0 {
        shadow = shadow_factor(in.rel, n);
    }
    let sun = sun_light() * shadow;
    // Hemisphere ambient from the sky above and bounced light from the ground below.
    let sky_col = sky_base(normalize(vec3(n.x, max(n.y, 0.2), n.z)));
    // Desaturate the sky light a little so shade isn't too blue.
    let sky_amb = mix(vec3(dot(sky_col, vec3(0.2126, 0.7152, 0.0722))), sky_col, 0.6) * 0.8;
    let ground_amb = vec3(0.10, 0.08, 0.06) * (0.15 + 0.85 * daylight());
    let ambient = mix(ground_amb, sky_amb, n.y * 0.5 + 0.5) * occlusion;

    var color = s.albedo * (sun * ndl + ambient);
    color += sun * ggx_specular(n, v, l, s.roughness);

    // Brush preview: a soft glowing rim where the surface meets the dig sphere.
    if globals.brush.w > 0.0 {
        let d = abs(length(in.rel - globals.brush.xyz) - globals.brush.w);
        color += vec3(0.25, 0.8, 1.0) * (1.0 - smoothstep(0.0, 0.18, d)) * 1.5;
    }

    return vec4(apply_fog(color, in.rel), 1.0);
}

// Depth-only pass from the sun.
@vertex
fn vs_shadow(in: VsIn) -> @builtin(position) vec4<f32> {
    let rel = in.position + chunk_offsets[in.instance].xyz;
    return globals.shadow_view_proj * vec4(rel, 1.0);
}
