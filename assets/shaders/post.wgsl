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

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag.xy * globals.screen.zw;
    let px = vec2<i32>(frag.xy);
    let size = vec2<i32>(textureDimensions(hdr));
    var col = textureLoad(hdr, px, 0).rgb;
    if globals.taa.x > 0.5 {
        // TAA softens a little; give back some edge contrast (a light unsharp mask, limited so it
        // can't ring).
        let n = textureLoad(hdr, clamp(px + vec2(0, -1), vec2(0), size - 1), 0).rgb;
        let s = textureLoad(hdr, clamp(px + vec2(0, 1), vec2(0), size - 1), 0).rgb;
        let e = textureLoad(hdr, clamp(px + vec2(1, 0), vec2(0), size - 1), 0).rgb;
        let w = textureLoad(hdr, clamp(px + vec2(-1, 0), vec2(0), size - 1), 0).rgb;
        let lo = min(min(min(n, s), min(e, w)), col);
        let hi = max(max(max(n, s), max(e, w)), col);
        col = clamp(col + (col * 4.0 - n - s - e - w) * 0.32, lo, hi);
    }
    col += textureSampleLevel(bloom, linear_sampler, uv, 0.0).rgb * 0.06;
    col *= globals.fog.w;
    col = aces(col);
    let luma = dot(col, vec3(0.2126, 0.7152, 0.0722));
    col = max(mix(vec3(luma), col, 1.12), vec3(0.0));
    return vec4(col, 1.0);
}
