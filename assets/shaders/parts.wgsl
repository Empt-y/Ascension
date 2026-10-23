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
    // x = scale, y = ghost state (0 none, 1 valid, 2 invalid), z = sway, w = fade (1 solid, 0 gone).
    @location(5) params: vec4<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
    // Centroid: under MSAA a pixel only partly covered by a triangle would otherwise take these at
    // its centre, outside the triangle, extrapolated. On thin leaf cards seen edge-on the light
    // and fog extrapolated to many times their real values: single white pixels at leaf edges,
    // which bloom spread into flashes. (UVs stay at the centre: texture derivatives need them.)
    @location(1) @interpolate(perspective, centroid) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) @interpolate(perspective, centroid) color: vec4<f32>,
    @location(4) @interpolate(flat) ghost: u32,
    // 0..1 per instance, for colour variation.
    @location(5) @interpolate(flat) variety: f32,
    @location(6) @interpolate(flat) fade: f32,
    // Foliage only, lit per vertex (it's drawn many layers deep, so per-pixel work is precious):
    // sky and ground ambient plus moonlight, sun colour (already shadowed), and fog colour (rgb)
    // and amount (a).
    @location(7) @interpolate(perspective, centroid) ambient: vec3<f32>,
    @location(8) @interpolate(perspective, centroid) sun: vec3<f32>,
    @location(9) @interpolate(perspective, centroid) fog: vec4<f32>,
    // Where the vertex is now and was last frame, both through this frame's camera: their
    // difference is the wind's motion, which TAA can't infer from depth.
    @location(10) @interpolate(perspective, centroid) now_clip: vec4<f32>,
    @location(11) @interpolate(perspective, centroid) prev_clip: vec4<f32>,
}

struct SceneOut {
    @location(0) color: vec4<f32>,
    // NDC motion this frame beyond the camera's.
    @location(1) motion: vec2<f32>,
}

fn rotate_y(v: vec3<f32>, yaw: f32) -> vec3<f32> {
    let c = cos(yaw);
    let s = sin(yaw);
    return vec3(v.x * c + v.z * s, v.y, -v.x * s + v.z * c);
}

// Camera-relative position of a vertex: scaled, turned, placed, and (with `wind`) pushed by the
// wind. Shadows are cast without wind: moving leaves are fine, flickering shadow patches aren't.
fn place(in: VsIn, wind: bool) -> vec3<f32> {
    return place_at(in, select(0.0, globals.shadow.z, wind), globals.camera.w);
}

// `place` with wind of `strength` at time `t`.
fn place_at(in: VsIn, strength: f32, t: f32) -> vec3<f32> {
    let local = in.position * in.params.x;
    var rel = rotate_y(local, in.origin.w) + in.origin.xyz;
    let sway = in.params.z * strength;
    if sway > 0.0 {
        let world = in.origin.xyz + globals.camera.xyz;
        let phase = dot(world.xz, vec2(0.37, 0.61));
        // Movement smaller than a pixel only makes thin edges (birch branches, twig tips) flicker,
        // so it dies away with distance: flutter by 30 m, all bending by 60 m.
        let dist = length(in.origin.xyz);
        // The whole model bends with height (trunks stay rooted)...
        let h = max(local.y, 0.0);
        let gust = sin(t * 1.1 + phase) + 0.45 * sin(t * 2.7 + phase * 1.7);
        let bend = 1.0 - smoothstep(20.0, 60.0, dist);
        rel += vec3(1.0, 0.0, 0.55) * gust * h * h * 0.0011 * sway * bend;
        // ...and leaf cards flutter, more towards their free ends. The phase changes slowly across
        // the model, so each card moves nearly as one piece: a fast phase change between a card's own
        // corners made it ripple, and its edges swam across pixels as sparkle.
        let flutter = 1.0 - smoothstep(12.0, 30.0, dist);
        if material.extra.w > 0.5 && flutter > 0.0 {
            let f = sin(t * 4.0 + phase * 3.0 + dot(local, vec3(0.7, 0.55, 0.95)));
            rel += vec3(0.5, 0.35, -0.4) * f * 0.025 * in.uv.y * sway * flutter;
        }
    }
    return rel;
}

// How far towards the sun foliage looks its shadow up from (metres).
const FOLIAGE_SHADOW_LIFT: f32 = 0.8;

fn hash11(x: f32) -> f32 {
    return fract(sin(x * 12.9898) * 43758.547);
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.rel = place(in, true);
    out.normal = rotate_y(in.normal, in.origin.w);
    out.clip = globals.view_proj * vec4(out.rel, 1.0);
    out.now_clip = out.clip;
    out.prev_clip = out.clip;
    if in.params.z > 0.0 {
        out.prev_clip = globals.view_proj * vec4(place_at(in, globals.motion.y, globals.motion.x), 1.0);
    }
    out.uv = in.uv;
    out.color = in.color;
    out.ghost = u32(round(in.params.y));
    let world = in.origin.xyz + globals.camera.xyz;
    out.variety = hash11(dot(floor(world * 4.0), vec3(1.0, 57.0, 113.0)));
    out.fade = in.params.w;
    if material.extra.w > 0.5 {
        let n = normalize(out.normal);
        let sky_col = sky_base(normalize(vec3(n.x, max(n.y, 0.2), n.z)));
        let sky_amb = mix(vec3(dot(sky_col, vec3(0.2126, 0.7152, 0.0722))), sky_col, 0.6) * 0.8;
        let ground_amb = vec3(0.10, 0.08, 0.06) * (0.15 + 0.85 * daylight());
        let moon = vec3(0.05, 0.065, 0.12) * (1.0 - daylight()) * max(dot(n, -sun_dir()), 0.0);
        out.ambient = mix(ground_amb, sky_amb, n.y * 0.5 + 0.5) + moon;
        // Shadow from where the leaf rests, not where the wind has pushed it: a swaying leaf near a
        // shadow edge would otherwise flash between lit and shadowed (sparkle).
        // Looked up from a point towards the sun: a grass blade's base sits at the ground's own
        // depth in the shadow map, and the soft lookup reaches the ground around it, so the ground
        // itself shadowed it (dark bases on sunlit tufts).
        out.sun = sun_light() * shadow_factor_foliage(place(in, false) + sun_dir() * FOLIAGE_SHADOW_LIFT, n);
        let edge = smoothstep(globals.far.x * 0.75, globals.far.x * 0.97, length(out.rel.xz));
        out.fog = vec4(sky_base(normalize(out.rel)), max(fog_total(out.rel), edge));
        if underwater() {
            out.fog = vec4(water_fog_color(), water_fog_amount(out.rel));
        }
    }
    return out;
}

struct ShadowOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_shadow(in: VsIn) -> ShadowOut {
    var out: ShadowOut;
    out.clip = shadow_caster_matrix() * vec4(place(in, false), 1.0);
    out.uv = in.uv;
    return out;
}

@fragment
fn fs_shadow(in: ShadowOut) {
    if material.extra.y >= 0.0 && material.extra.z > 0.0 {
        // A blurred mip: leaves cast soft, dappled shadows rather than sharp speckle, which would
        // shimmer whenever the shadow maps are redrawn.
        let a = textureSampleLevel(model_textures, model_sampler, in.uv, i32(material.extra.y), 2.5).a;
        if a < material.extra.z {
            discard;
        }
    }
}

// Sun and sky on foliage, from the per-vertex light: wrapped diffuse (leaves face every way), light
// transmitted through the leaves when looking towards the sun, and ambient darkened deep inside
// the crown. No specular: leaf cards are matte at the sizes they're seen.
fn shade_foliage(in: VsOut, albedo: vec3<f32>, n: vec3<f32>, occlusion: f32) -> vec3<f32> {
    let l = sun_dir();
    let v = normalize(-in.rel);
    let wrap = max(dot(n, l) + 0.45, 0.0) / 1.45;
    let through = pow(max(dot(-v, l), 0.0), 3.0) * 0.55 + 0.12;
    let sun = in.sun * (wrap + through * vec3(1.0, 1.08, 0.75)) * mix(0.35, 1.0, occlusion);
    let color = albedo * (sun + in.ambient * occlusion + point_light(in.rel, n, 0.5));
    return mix(color, in.fog.rgb, in.fog.a);
}

@fragment
fn fs_main(in: VsOut) -> SceneOut {
    let geometric = normalize(in.normal);
    let world = in.rel + globals.camera.xyz;
    let textured = material.extra.y >= 0.0;
    var albedo = material.base.rgb * in.color.rgb;
    var alpha = 1.0;
    if textured {
        let t = textureSample(model_textures, model_sampler, in.uv, i32(material.extra.y));
        albedo *= t.rgb;
        if material.extra.z > 0.0 {
            // Up close: sharpen alpha around the cutoff so coverage gives crisp, antialiased
            // edges, over about two pixels: across one, a gap between leaves narrower than that
            // (with bright sky or ground behind) snapped open and shut as the leaves swayed and
            // flashed. Further out, crisp leaf edges become pixel-sized noise, so they soften into
            // partial coverage instead.
            let sharp = clamp((t.a - material.extra.z) / max(2.0 * fwidth(t.a), 1e-4) + 0.5, 0.0, 1.0);
            let soft = smoothstep(material.extra.z - 0.3, material.extra.z + 0.3, t.a);
            alpha = mix(sharp, soft, smoothstep(12.0, 45.0, length(in.rel)));
        }
    }
    let occlusion = in.color.a;
    var color: vec3<f32>;
    let foliage = material.extra.w > 0.5;
    if foliage {
        let shift = in.variety - 0.5;
        albedo *= vec3(1.0 + shift * 0.25, 1.0 + shift * 0.12, 1.0 - shift * 0.2) * (0.9 + 0.2 * hash11(in.variety * 7.0));
        // Leaves are layered many deep, so their shadow comes from the vertices (in `in.sun`).
        color = shade_foliage(in, albedo, geometric, occlusion);
    } else {
        // The terrain's procedural detail, toned down: a little grain and relief so flat faces read
        // as material rather than plastic. Polished metal shows bump noise as blotchy glints, so
        // metals get much less of it. (It takes derivatives; the branch is on a uniform, so this is
        // still uniform control flow. Leaf cards skip it: their texture is their detail.)
        let detail = surface_detail(world, geometric);
        let n = normalize(mix(geometric, detail.normal, mix(0.35, 0.06, material.extra.x)));
        albedo *= 1.0 + detail.tone * 0.35;
        var shadow = 1.0;
        if dot(geometric, sun_dir()) > 0.0 {
            shadow = shadow_factor(in.rel, geometric);
        }
        let roughness = aa_roughness(material.base.a, detail.widen);
        color = apply_fog(shade_pbr(albedo, roughness, material.extra.x, n, in.rel, occlusion, shadow), in.rel);
    }
    // Distance and LOD fades. With MSAA they go through coverage, which blends smoothly over the
    // samples; a screen-space dither pattern stays put while things move across it and fizzes.
    // Without MSAA, fall back to a dither.
    if ALPHA_TO_COVERAGE {
        alpha *= in.fade;
    } else {
        let dither = fract(52.9829189 * fract(dot(in.clip.xy, vec2(0.06711056, 0.00583715))));
        if dither >= in.fade {
            discard;
        }
        if alpha < 0.5 {
            discard;
        }
    }
    var out: SceneOut;
    out.color = vec4(debug_hot(color, select(3.0, 2.0, foliage)), alpha);
    out.motion = in.now_clip.xy / in.now_clip.w - in.prev_clip.xy / in.prev_clip.w;
    return out;
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
