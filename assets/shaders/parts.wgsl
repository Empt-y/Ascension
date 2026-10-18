// Static models placed in the world: building parts and nature scatter. Instanced meshes with
// metallic-roughness materials, lit like the terrain, casting and receiving sun shadows.
//
// Leaf cards (a cut-out texture) take the foliage path: alpha becomes MSAA coverage for smooth
// edges, light wraps round and shines through the leaves, and each instance gets a slight colour
// shift. Anything with `sway` bends in the wind. Also the translucent placement ghost.

struct Material {
    // rgb = base colour (linear), a = roughness.
    base: vec4<f32>,
    // x = metallic, y = texture layer (-1 = untextured), z = alpha cutoff, w = 1 for foliage.
    extra: vec4<f32>,
}

@group(1) @binding(0) var<uniform> material: Material;
@group(1) @binding(1) var model_textures: texture_2d_array<f32>;
@group(1) @binding(2) var model_sampler: sampler;

// Set when the pipeline turns alpha into MSAA coverage; otherwise cut-outs are discarded.
override ALPHA_TO_COVERAGE: bool = false;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    // rgb = tint, a = ambient occlusion.
    @location(3) color: vec4<f32>,
    // xyz = camera-relative origin, w = yaw (radians).
    @location(4) origin: vec4<f32>,
    // x = scale, y = ghost state (0 none, 1 valid, 2 invalid), z = sway, w = fade (1 = solid).
    @location(5) params: vec4<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) color: vec4<f32>,
    @location(4) @interpolate(flat) ghost: u32,
    // 0..1 per instance, for colour variation.
    @location(5) @interpolate(flat) variety: f32,
    @location(6) @interpolate(flat) fade: f32,
}

fn rotate_y(v: vec3<f32>, yaw: f32) -> vec3<f32> {
    let c = cos(yaw);
    let s = sin(yaw);
    return vec3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

// Camera-relative position of a vertex: scaled, turned, placed, and pushed by the wind.
fn place(in: VsIn) -> vec3<f32> {
    let local = in.position * in.params.x;
    var rel = rotate_y(local, in.origin.w) + in.origin.xyz;
    let sway = in.params.z;
    if sway > 0.0 {
        let world = in.origin.xyz + globals.camera.xyz;
        let phase = dot(world.xz, vec2(0.37, 0.61));
        let t = globals.camera.w;
        // The whole model bends with height (trunks stay rooted)...
        let h = max(local.y, 0.0);
        let gust = sin(t * 1.1 + phase) + 0.45 * sin(t * 2.7 + phase * 1.7);
        rel += vec3(1.0, 0.0, 0.55) * gust * h * h * 0.0011 * sway;
        // ...and leaf cards flutter, more towards their free ends.
        if material.extra.w > 0.5 {
            let f = sin(t * 5.3 + phase * 3.0 + dot(local, vec3(2.1, 1.7, 2.9)));
            rel += vec3(0.5, 0.35, -0.4) * f * 0.035 * in.uv.y * sway;
        }
    }
    return rel;
}

fn hash11(x: f32) -> f32 {
    return fract(sin(x * 12.9898) * 43758.547);
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.rel = place(in);
    out.normal = rotate_y(in.normal, in.origin.w);
    out.clip = globals.view_proj * vec4(out.rel, 1.0);
    out.uv = in.uv;
    out.color = in.color;
    out.ghost = u32(round(in.params.y));
    let world = in.origin.xyz + globals.camera.xyz;
    out.variety = hash11(dot(floor(world * 4.0), vec3(1.0, 57.0, 113.0)));
    out.fade = in.params.w;
    return out;
}

struct ShadowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_shadow(in: VsIn) -> ShadowOut {
    var out: ShadowOut;
    out.clip = globals.shadow_view_proj * vec4(place(in), 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_shadow(in: ShadowOut) {
    if material.extra.y >= 0.0 && material.extra.z > 0.0 {
        let a = textureSample(model_textures, model_sampler, in.uv, i32(material.extra.y)).a;
        if a < material.extra.z {
            discard;
        }
    }
}

// Sun and sky on foliage: wrapped diffuse (leaves face every way), light transmitted through the
// leaves when looking towards the sun, and ambient darkened deep inside the crown.
fn shade_foliage(albedo: vec3<f32>, n: vec3<f32>, rel: vec3<f32>, occlusion: f32, shadow: f32) -> vec3<f32> {
    let l = sun_dir();
    let v = normalize(-rel);
    var color = shade_pbr(albedo, 0.85, 0.0, n, rel, occlusion, shadow);
    let sun = sun_light() * shadow;
    let ndl = dot(n, l);
    let wrap = max(ndl + 0.45, 0.0) / 1.45 - max(ndl, 0.0);
    let through = pow(max(dot(-v, l), 0.0), 3.0) * 0.55 + 0.12;
    color += albedo * sun * (wrap * 0.8 + through * vec3(1.0, 1.08, 0.75)) * mix(0.35, 1.0, occlusion);
    return color;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let geometric = normalize(in.normal);
    let world = in.rel + globals.camera.xyz;
    // Derivatives first: everything below may branch or discard.
    let detail = surface_detail(world, geometric);
    let textured = material.extra.y >= 0.0;
    var albedo = material.base.rgb * in.color.rgb;
    var alpha = 1.0;
    if textured {
        let t = textureSample(model_textures, model_sampler, in.uv, i32(material.extra.y));
        albedo *= t.rgb;
        if material.extra.z > 0.0 {
            // Sharpen alpha around the cutoff so coverage gives crisp, antialiased edges.
            alpha = clamp((t.a - material.extra.z) / max(fwidth(t.a), 1e-4) + 0.5, 0.0, 1.0);
        }
    }
    let occlusion = in.color.a;
    var color: vec3<f32>;
    if material.extra.w > 0.5 {
        let shift = in.variety - 0.5;
        albedo *= vec3(1.0 + shift * 0.25, 1.0 + shift * 0.12, 1.0 - shift * 0.2) * (0.9 + 0.2 * hash11(in.variety * 7.0));
        let shadow = shadow_factor(in.rel, geometric);
        color = shade_foliage(albedo, geometric, in.rel, occlusion, shadow);
    } else {
        // The terrain's procedural detail, toned down: a little grain and relief so flat faces read
        // as material rather than plastic. Polished metal shows bump noise as blotchy glints, so
        // metals get much less of it.
        let n = normalize(mix(geometric, detail.normal, mix(0.35, 0.06, material.extra.x)));
        albedo *= 1.0 + detail.tone * 0.35;
        var shadow = 1.0;
        if dot(geometric, sun_dir()) > 0.0 {
            shadow = shadow_factor(in.rel, geometric);
        }
        color = shade_pbr(albedo, material.base.a, material.extra.x, n, in.rel, occlusion, shadow);
    }
    // Distance fade: thins out through MSAA coverage (a dissolve, not a pop).
    alpha *= in.fade;
    if !ALPHA_TO_COVERAGE && alpha < 0.5 {
        discard;
    }
    return vec4(apply_fog(color, in.rel), alpha);
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
