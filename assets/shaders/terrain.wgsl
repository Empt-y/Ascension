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

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal.xyz);
    let occlusion = clamp(in.normal.w, 0.0, 1.0);
    let world = in.rel + globals.camera.xyz;
    let s = terrain_surface(in.weights_a, in.weights_b, world, n, length(in.rel));
    var shadow = 1.0;
    if dot(n, sun_dir()) > 0.0 {
        shadow = shadow_factor(in.rel, n);
    }
    var color = shade_terrain(s, n, in.rel, occlusion, shadow);

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
