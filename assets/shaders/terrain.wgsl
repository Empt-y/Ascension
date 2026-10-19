// Terrain: stylised PBR-lite with triplanar procedural detail, sun shadows and aerial fog.

// Camera-relative origin of each drawn chunk, indexed by instance.
@group(0) @binding(1) var<storage, read> chunk_offsets: array<vec4<f32>>;

struct VsIn {
    @location(0) position: vec3<f32>,
    // xyz normal, w ambient occlusion (1 = open sky).
    @location(1) normal: vec4<f32>,
    // Blend weights for material slots 0..3, 4..7 and 8..11 (material id = slot + 1).
    @location(2) weights_a: vec4<f32>,
    @location(3) weights_b: vec4<f32>,
    @location(4) weights_c: vec4<f32>,
    @builtin(instance_index) instance: u32,
}

struct VsOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) weights_a: vec4<f32>,
    @location(3) weights_b: vec4<f32>,
    @location(4) weights_c: vec4<f32>,
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
    out.weights_c = in.weights_c;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let geometric = normalize(in.normal.xyz);
    let occlusion = clamp(in.normal.w, 0.0, 1.0);
    let world = in.rel + globals.camera.xyz;
    let detail = surface_detail(world, geometric);
    let n = detail.normal;
    var s = terrain_surface(in.weights_a, in.weights_b, in.weights_c, world, geometric, detail.tone);
    s.roughness = aa_roughness(s.roughness, detail.widen);
    var shadow = 1.0;
    if dot(geometric, sun_dir()) > 0.0 {
        shadow = shadow_factor(in.rel, geometric);
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
    return shadow_caster_matrix() * vec4(rel, 1.0);
}
