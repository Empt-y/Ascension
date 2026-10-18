// Building parts (and later machines): instanced static meshes with metallic-roughness materials,
// lit like the terrain, casting and receiving sun shadows. Also the translucent placement ghost.

struct Material {
    // rgb = base colour (linear), a = roughness.
    base: vec4<f32>,
    // x = metallic.
    extra: vec4<f32>,
}

@group(1) @binding(0) var<uniform> material: Material;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    // xyz = camera-relative origin, w = quarter turns about +Y (0..3) + 4 * ghost state (0 none,
    // 1 valid, 2 invalid).
    @location(2) instance: vec4<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) @interpolate(flat) ghost: u32,
}

fn rotate_y(v: vec3<f32>, quarter_turns: u32) -> vec3<f32> {
    switch quarter_turns % 4u {
        case 1u: { return vec3(v.z, v.y, -v.x); }
        case 2u: { return vec3(-v.x, v.y, -v.z); }
        case 3u: { return vec3(-v.z, v.y, v.x); }
        default: { return v; }
    }
}

fn unpack(w: f32) -> vec2<u32> {
    let code = u32(round(w));
    return vec2(code % 4u, code / 4u);
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    let code = unpack(in.instance.w);
    var out: VsOut;
    out.rel = rotate_y(in.position, code.x) + in.instance.xyz;
    out.normal = rotate_y(in.normal, code.x);
    out.clip = globals.view_proj * vec4(out.rel, 1.0);
    out.ghost = code.y;
    return out;
}

@vertex
fn vs_shadow(in: VsIn) -> @builtin(position) vec4<f32> {
    let code = unpack(in.instance.w);
    let rel = rotate_y(in.position, code.x) + in.instance.xyz;
    return globals.shadow_view_proj * vec4(rel, 1.0);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let geometric = normalize(in.normal);
    let world = in.rel + globals.camera.xyz;
    // The terrain's procedural detail, toned down: a little grain and relief so flat faces read as
    // material rather than plastic.
    let detail = surface_detail(world, geometric);
    // Polished metal shows bump noise as blotchy glints, so metals get much less of it.
    let n = normalize(mix(geometric, detail.normal, mix(0.35, 0.06, material.extra.x)));
    let albedo = material.base.rgb * (1.0 + detail.tone * 0.35);
    var shadow = 1.0;
    if dot(geometric, sun_dir()) > 0.0 {
        shadow = shadow_factor(in.rel, geometric);
    }
    let color = shade_pbr(albedo, material.base.a, material.extra.x, n, in.rel, 1.0, shadow);
    return vec4(apply_fog(color, in.rel), 1.0);
}

// Placement preview: the part tinted blue (fits) or red (doesn't), translucent, with a rim.
@fragment
fn fs_ghost(in: VsOut) -> @location(0) vec4<f32> {
    let n = normalize(in.normal);
    let v = normalize(-in.rel);
    let tint = select(vec3(1.0, 0.25, 0.2), vec3(0.3, 0.75, 1.0), in.ghost == 1u);
    let rim = pow(1.0 - abs(dot(n, v)), 2.0);
    let light = 0.5 + 0.5 * max(dot(n, sun_dir()), 0.0);
    return vec4(tint * (0.6 * light + 1.4 * rim), 0.35 + 0.4 * rim);
}
