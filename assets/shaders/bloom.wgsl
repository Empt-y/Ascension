// Bloom: a bright-pass downsample, a chain of 13-tap downsamples, then tent-filter upsamples that are
// added back up the chain (blending is One + One).

@group(1) @binding(0) var src: texture_2d<f32>;
@group(1) @binding(2) var linear_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> VsOut {
    let p = fullscreen_pos(index);
    var out: VsOut;
    out.clip = p;
    out.uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5);
    return out;
}

fn tap(uv: vec2<f32>) -> vec3<f32> {
    return textureSampleLevel(src, linear_sampler, uv, 0.0).rgb;
}

fn down13(uv: vec2<f32>) -> vec3<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src));
    let a = tap(uv + t * vec2(-2.0, -2.0));
    let b = tap(uv + t * vec2(0.0, -2.0));
    let c = tap(uv + t * vec2(2.0, -2.0));
    let d = tap(uv + t * vec2(-2.0, 0.0));
    let e = tap(uv);
    let f = tap(uv + t * vec2(2.0, 0.0));
    let g = tap(uv + t * vec2(-2.0, 2.0));
    let h = tap(uv + t * vec2(0.0, 2.0));
    let i = tap(uv + t * vec2(2.0, 2.0));
    let j = tap(uv + t * vec2(-1.0, -1.0));
    let k = tap(uv + t * vec2(1.0, -1.0));
    let l = tap(uv + t * vec2(-1.0, 1.0));
    let m = tap(uv + t * vec2(1.0, 1.0));
    return e * 0.125 + (a + c + g + i) * 0.03125 + (b + d + f + h) * 0.0625 + (j + k + l + m) * 0.125;
}

// The first downsample, as five 2x2 groups each weighted by 1 / (1 + luma) (Karis average): a lone
// bright pixel (a glint on a twig) can't dominate its neighbourhood, so it doesn't bloom into a
// sparkle that flickers as the view moves.
fn down13_karis(uv: vec2<f32>) -> vec3<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src));
    let a = tap(uv + t * vec2(-2.0, -2.0));
    let b = tap(uv + t * vec2(0.0, -2.0));
    let c = tap(uv + t * vec2(2.0, -2.0));
    let d = tap(uv + t * vec2(-2.0, 0.0));
    let e = tap(uv);
    let f = tap(uv + t * vec2(2.0, 0.0));
    let g = tap(uv + t * vec2(-2.0, 2.0));
    let h = tap(uv + t * vec2(0.0, 2.0));
    let i = tap(uv + t * vec2(2.0, 2.0));
    let j = tap(uv + t * vec2(-1.0, -1.0));
    let k = tap(uv + t * vec2(1.0, -1.0));
    let l = tap(uv + t * vec2(-1.0, 1.0));
    let m = tap(uv + t * vec2(1.0, 1.0));
    let groups = array<vec3<f32>, 5>(
        (j + k + l + m) * 0.25,
        (a + b + d + e) * 0.25,
        (b + c + e + f) * 0.25,
        (d + e + g + h) * 0.25,
        (e + f + h + i) * 0.25,
    );
    let weights = array<f32, 5>(0.5, 0.125, 0.125, 0.125, 0.125);
    var sum = vec3(0.0);
    var total = 0.0;
    for (var n = 0; n < 5; n++) {
        let w = weights[n] / (1.0 + dot(groups[n], vec3(0.2126, 0.7152, 0.0722)));
        sum += groups[n] * w;
        total += w;
    }
    return sum / total;
}

@fragment
fn fs_prefilter(in: VsOut) -> @location(0) vec4<f32> {
    let c = down13_karis(in.uv);
    // Soft knee: only light well above "white" blooms.
    let bright = max(max(c.r, c.g), c.b);
    let w = smoothstep(1.0, 3.0, bright);
    return vec4(min(c * w, vec3(64.0)), 1.0);
}

@fragment
fn fs_down(in: VsOut) -> @location(0) vec4<f32> {
    return vec4(down13(in.uv), 1.0);
}

@fragment
fn fs_up(in: VsOut) -> @location(0) vec4<f32> {
    let t = 1.0 / vec2<f32>(textureDimensions(src));
    var c = tap(in.uv) * 4.0;
    c += (tap(in.uv + t * vec2(-1.0, 0.0)) + tap(in.uv + t * vec2(1.0, 0.0)) + tap(in.uv + t * vec2(0.0, -1.0))
        + tap(in.uv + t * vec2(0.0, 1.0))) * 2.0;
    c += tap(in.uv + t * vec2(-1.0, -1.0)) + tap(in.uv + t * vec2(1.0, -1.0)) + tap(in.uv + t * vec2(-1.0, 1.0))
        + tap(in.uv + t * vec2(1.0, 1.0));
    return vec4(c / 16.0, 1.0);
}
