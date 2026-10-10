// Sky dome drawn wherever terrain left the depth buffer at the far plane (reversed-Z: 0).

struct VsOut {
    @builtin(position) @invariant clip: vec4<f32>,
    @location(0) ndc: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    let p = fullscreen_pos(index);
    var out: VsOut;
    // z = 0 is the far plane with reversed-Z.
    out.clip = vec4(p.xy, 0.0, 1.0);
    out.ndc = p.xy;
    return out;
}

fn stars(dir: vec3<f32>) -> vec3<f32> {
    let cell = floor(dir * 220.0);
    let h = hash13(cell);
    let star = step(0.9975, h) * (0.5 + 2.5 * hash13(cell + 7.0));
    let night = 1.0 - smoothstep(-0.2, 0.05, sun_dir().y);
    return vec3(star * night * smoothstep(0.0, 0.15, dir.y));
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    let far = globals.inv_view_proj * vec4(in.ndc, 1e-6, 1.0);
    let dir = normalize(far.xyz / far.w);
    var col = sky_base(dir);
    let mu = dot(dir, sun_dir());
    let disk = smoothstep(0.99955, 0.9998, mu);
    col += sun_light() * disk * 25.0;
    // A pale moon opposite the sun.
    let moon = smoothstep(0.9993, 0.9996, dot(dir, -sun_dir()));
    col += vec3(0.6, 0.65, 0.8) * moon * (1.0 - daylight()) * 2.0;
    col += stars(dir);
    return vec4(col, 1.0);
}
