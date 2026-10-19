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

// Stars: soft round points at fixed places on the sky, each at least a pixel wide. (Whole hashed
// cells switched hard on and off gained and lost pixels as the view turned, and glinted.)
fn stars(dir: vec3<f32>, pixel: f32) -> vec3<f32> {
    let night = 1.0 - smoothstep(-0.2, 0.05, sun_dir().y);
    if night <= 0.0 {
        return vec3(0.0);
    }
    let scale = 220.0;
    let p = dir * scale;
    let base = floor(p) - 1.0;
    var light = 0.0;
    // The 27 cells around this direction; each may hold one star.
    for (var i = 0; i < 27; i++) {
        let cell = base + vec3(f32(i % 3), f32((i / 3) % 3), f32(i / 9));
        if hash13(cell) < 0.995 {
            continue;
        }
        let jitter = vec3(hash13(cell + 3.1), hash13(cell + 5.7), hash13(cell + 9.3)) - 0.5;
        let star = normalize((cell + 0.5 + jitter * 0.5) / scale);
        // Cells are 3D, so a star's point on the sky can fall in a neighbouring cell; keep it only
        // in the cell it lands in, so it's whole and drawn once.
        if any(floor(star * scale) != cell) {
            continue;
        }
        // Angular distance, against a width of at least a pixel (so the star never aliases).
        let d = length(dir - star);
        let width = max(pixel * 0.9, 0.00035);
        let brightness = 0.35 + 0.9 * hash13(cell + 7.0);
        light += brightness * exp(-(d * d) / (width * width)) * min(1.0, 0.0006 / width);
    }
    return vec3(light * night * smoothstep(0.0, 0.15, dir.y));
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
    // Pixel size in radians, from how fast the view direction changes across the screen.
    col += stars(dir, length(fwidth(dir)));
    return vec4(col, 1.0);
}
