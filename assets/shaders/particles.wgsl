// Particles: camera-facing soft billboards, drawn after the water, testing depth without writing
// it, premultiplied-alpha blended. Smoke (kind 0) is lit by the sun and sky (and from below by any
// glow it carries, from lava under it) and blends over the scene; glow (kind 1: lava spray,
// bombs) adds light and blooms.

struct VsIn {
    @builtin(vertex_index) vertex: u32,
    // xyz world position, w radius.
    @location(0) pos_size: vec4<f32>,
    // rgb colour, a opacity (smoke) or brightness (glow).
    @location(1) color: vec4<f32>,
    // x kind (0 smoke, 1 glow), y glow from below (smoke), z spin (radians), w seed.
    @location(2) params: vec4<f32>,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) params: vec4<f32>,
    @location(3) rel: vec3<f32>,
    // The sun's direction across the billboard (for lighting a puff on its sunny side).
    @location(4) sun_uv: vec2<f32>,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var corners = array<vec2<f32>, 6>(
        vec2(-1.0, -1.0), vec2(1.0, -1.0), vec2(1.0, 1.0),
        vec2(-1.0, -1.0), vec2(1.0, 1.0), vec2(-1.0, 1.0),
    );
    let c = corners[in.vertex];
    let centre = in.pos_size.xyz - globals.camera.xyz;
    let to_cam = normalize(-centre);
    var right = cross(vec3(0.0, 1.0, 0.0), to_cam);
    if length(right) < 1e-3 {
        right = vec3(1.0, 0.0, 0.0);
    }
    right = normalize(right);
    let up = cross(to_cam, right);
    let s = sin(in.params.z);
    let k = cos(in.params.z);
    let spun = vec2(c.x * k - c.y * s, c.x * s + c.y * k);
    var out: VsOut;
    out.rel = centre + (right * spun.x + up * spun.y) * in.pos_size.w;
    out.clip = globals.view_proj * vec4(out.rel, 1.0);
    out.uv = c;
    let l = sun_dir();
    // (In the billboard's spun frame, as `uv` is.)
    let su = vec2(dot(l, right), dot(l, up));
    out.sun_uv = vec2(su.x * k + su.y * s, -su.x * s + su.y * k);
    out.color = in.color;
    out.params = in.params;
    return out;
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let d = length(in.uv);
    if d > 1.0 {
        discard;
    }
    let seed = in.params.w;
    let t = globals.camera.w;
    // A lumpy, slowly churning puff rather than a flat disc: billows round its edge, denser and
    // thinner patches inside.
    let lumps = value_noise(vec3(in.uv * 2.2 + vec2(seed * 7.3, seed * 3.1), seed * 5.0 + t * 0.07)) * 0.65
        + value_noise(vec3(in.uv * 5.3 + vec2(seed * 2.0), t * 0.13 + seed)) * 0.35;
    let edge = d + (0.5 - lumps) * 0.8;
    let soft = (1.0 - smoothstep(0.55, 0.9, edge)) * (0.5 + 0.8 * lumps);
    if in.params.x > 0.5 {
        // Glow: adds light, fading out with fog.
        let glow = in.color.rgb * in.color.a * soft * soft;
        return vec4(glow * (1.0 - fog_total(in.rel)), 0.0);
    }
    // Smoke: lit on its sunny side (as a ball would be: the billboard's sunward half and the
    // lumps on it), shadowed on the other, plus the sky from above; warmed from below by glow.
    let up = in.uv.y * 0.5 + 0.5;
    let ball = vec3(in.uv, sqrt(max(1.0 - d * d, 0.0)));
    let sunlit = clamp(dot(ball.xy, in.sun_uv) + ball.z * 0.5, 0.0, 1.0) * (0.6 + 0.6 * lumps);
    // (The sky's light as grey: blue sky light alone turned dark ash navy.)
    let sky = dot(sky_base(vec3(0.0, 1.0, 0.0)), vec3(0.2126, 0.7152, 0.0722));
    let light = sun_light() * 0.45 * sunlit + vec3(sky) * (0.45 + 0.35 * up);
    let under = vec3(1.0, 0.32, 0.06) * in.params.y * (1.0 - up) * 2.0;
    var rgb = in.color.rgb * light + under;
    rgb = apply_fog(rgb, in.rel);
    let alpha = clamp(in.color.a * soft, 0.0, 1.0);
    return vec4(rgb * alpha, alpha);
}
