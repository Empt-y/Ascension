// HDR -> display: exposure, ACES tonemap, a little extra saturation. The sRGB swapchain encodes.

@group(1) @binding(0) var hdr: texture_2d<f32>;
@group(1) @binding(1) var bloom: texture_2d<f32>;
@group(1) @binding(2) var linear_sampler: sampler;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_pos(index);
}

fn aces(x: vec3<f32>) -> vec3<f32> {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    return clamp((x * (a * x + b)) / (x * (c * x + d) + e), vec3(0.0), vec3(1.0));
}

// What a pixel of the image shows after exposure and tonemapping (before the saturation boost).
fn tonemapped(px: vec2<i32>, size: vec2<i32>, glow: vec3<f32>) -> vec3<f32> {
    let c = textureLoad(hdr, clamp(px, vec2(0), size - 1), 0).rgb;
    return aces((c + glow) * globals.fog.w);
}

fn max3(v: vec3<f32>) -> f32 {
    return max(v.x, max(v.y, v.z));
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag.xy * globals.output.zw;
    let px = vec2<i32>(frag.xy);
    let size = vec2<i32>(textureDimensions(hdr));
    let glow = textureSampleLevel(bloom, linear_sampler, uv, 0.0).rgb * 0.06;
    var col = tonemapped(px, size, glow);
    if globals.taa.x > 0.5 {
        // TAA (and upscaling) soften the image: give back edge contrast with AMD's RCAS (from
        // FidelityFX FSR 1, MIT licence). It sharpens each pixel against its four neighbours as far
        // as it can without leaving their range, so it can't ring or clip.
        let n = tonemapped(px + vec2(0, -1), size, glow);
        let w = tonemapped(px + vec2(-1, 0), size, glow);
        let e = tonemapped(px + vec2(1, 0), size, glow);
        let s = tonemapped(px + vec2(0, 1), size, glow);
        let mn = min(min(n, w), min(e, s));
        let mx = max(max(n, w), max(e, s));
        let hit_min = mn / (4.0 * mx + 1e-4);
        let hit_max = (1.0 - mx) / (4.0 * mn - 4.0 - 1e-4);
        let lobe_rgb = max(-hit_min, hit_max);
        let limit = 0.25 - 1.0 / 16.0;
        let lobe = max(-limit, min(max3(lobe_rgb), 0.0)) * globals.upscale.y;
        col = clamp((lobe * (n + w + e + s) + col) / (4.0 * lobe + 1.0), vec3(0.0), vec3(1.0));
    }
    let luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = max(mix(vec3(luma), col, 1.12), vec3(0.0));
    return vec4(col, 1.0);
}
