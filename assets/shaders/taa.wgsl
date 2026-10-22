// Temporal anti-aliasing: blend this frame (rendered with a sub-pixel jitter) into the history of
// past frames, so detail thinner than a pixel resolves to a steady average instead of shimmering as
// the view moves.
//
// Each pixel is reprojected into the previous frame from its depth and the camera's movement
// (`taa_reproject`), which is exact for everything static, then moved back by its own motion
// (`motion_tex`: how far the wind swayed it). History is clamped to the colour range of
// this frame's 3x3 neighbourhood, so anything that moved (wind, the player's edits) or was hidden
// last frame can't leave ghosts. Blending and clamping happen on squashed colour (c / (1 + luma)),
// so a single bright pixel can't keep flickering through the average.
//
// Upscaling: the scene renders below the window's resolution (`globals.screen`, against
// `globals.output`) and this pass writes window-sized history. Each window pixel takes the jittered
// render samples around it, weighted by how close each landed to its centre, and blends them in by
// how close the nearest one came: over the jitter sequence every window pixel gets samples right on
// it, so the history builds up the full resolution (the idea behind FSR 2 and other temporal
// upscalers). At native scale this is the usual TAA, with a 3x3 reconstruction of the current frame.
//
// `DEPTH_TEXTURE` is replaced with texture_depth_multisampled_2d or texture_depth_2d to match MSAA.

@group(1) @binding(0) var current: texture_2d<f32>;
@group(1) @binding(1) var history: texture_2d<f32>;
@group(1) @binding(2) var linear_sampler: sampler;
@group(1) @binding(3) var depth_tex: DEPTH_TEXTURE;
@group(1) @binding(4) var motion_tex: texture_2d<f32>;

@vertex
fn vs_main(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return fullscreen_pos(index);
}

fn luma(c: vec3<f32>) -> f32 {
    return dot(c, vec3(0.2126, 0.7152, 0.0722));
}

// A reversible squash of HDR colour: blending and clamping here stops bright outliers dominating
// without biasing the image darker (blending raw colour with 1 / (1 + luma) weights did).
fn squash(c: vec3<f32>) -> vec3<f32> {
    return c / (1.0 + luma(c));
}

fn unsquash(c: vec3<f32>) -> vec3<f32> {
    return c / max(1.0 - luma(c), 1e-4);
}

// History, sharpened by a 9-tap Catmull-Rom filter so repeated resampling doesn't blur it.
fn sample_history(uv: vec2<f32>) -> vec3<f32> {
    let size = vec2<f32>(textureDimensions(history));
    let p = uv * size - 0.5;
    let base = floor(p) + 0.5;
    let f = p - floor(p);
    let w0 = f * (-0.5 + f * (1.0 - 0.5 * f));
    let w1 = 1.0 + f * f * (-2.5 + 1.5 * f);
    let w2 = f * (0.5 + f * (2.0 - 1.5 * f));
    let w3 = f * f * (-0.5 + 0.5 * f);
    let w12 = w1 + w2;
    let o12 = w2 / w12;
    let t0 = (base - 1.0) / size;
    let t3 = (base + 2.0) / size;
    let t12 = (base + o12) / size;
    var c = vec3(0.0);
    c += textureSampleLevel(history, linear_sampler, vec2(t0.x, t0.y), 0.0).rgb * w0.x * w0.y;
    c += textureSampleLevel(history, linear_sampler, vec2(t12.x, t0.y), 0.0).rgb * w12.x * w0.y;
    c += textureSampleLevel(history, linear_sampler, vec2(t3.x, t0.y), 0.0).rgb * w3.x * w0.y;
    c += textureSampleLevel(history, linear_sampler, vec2(t0.x, t12.y), 0.0).rgb * w0.x * w12.y;
    c += textureSampleLevel(history, linear_sampler, vec2(t12.x, t12.y), 0.0).rgb * w12.x * w12.y;
    c += textureSampleLevel(history, linear_sampler, vec2(t3.x, t12.y), 0.0).rgb * w3.x * w12.y;
    c += textureSampleLevel(history, linear_sampler, vec2(t0.x, t3.y), 0.0).rgb * w0.x * w3.y;
    c += textureSampleLevel(history, linear_sampler, vec2(t12.x, t3.y), 0.0).rgb * w12.x * w3.y;
    c += textureSampleLevel(history, linear_sampler, vec2(t3.x, t3.y), 0.0).rgb * w3.x * w3.y;
    return max(c, vec3(0.0));
}

@fragment
fn fs_main(@builtin(position) frag: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = frag.xy * globals.output.zw;
    let rsize = globals.screen.xy;
    let size = vec2<i32>(rsize);
    // Where this window pixel's centre falls among the render pixels, unjittered; render pixel i
    // sampled the scene at i + 0.5 - shift.
    let p = uv * rsize;
    let shift = vec2(globals.taa.z, -globals.taa.w) * rsize * 0.5;
    let centre = vec2<i32>(floor(p + shift));

    // Reconstruct this frame here from the 3x3 render samples around it (Gaussian weights by
    // distance in render pixels), and gather their spread for the history clamp and the nearest
    // depth (reversed-Z: largest), so edges reproject with the object in front.
    var sum = vec3(0.0);
    var weights = 0.0;
    var closest = 0.0;
    var m1 = vec3(0.0);
    var m2 = vec3(0.0);
    var depth = 0.0;
    var nearest = centre;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let q = clamp(centre + vec2(x, y), vec2(0), size - 1);
            let c = squash(textureLoad(current, q, 0).rgb);
            let d = vec2<f32>(q) + 0.5 - shift - p;
            let w = exp(-2.29 * dot(d, d));
            sum += c * w;
            weights += w;
            closest = max(closest, w);
            m1 += c;
            m2 += c * c;
            let z = textureLoad(depth_tex, q, 0);
            if z > depth {
                depth = z;
                nearest = q;
            }
        }
    }
    let now = sum / max(weights, 1e-4);
    // The motion of whatever is nearest, like the depth.
    let motion = textureLoad(motion_tex, nearest, 0).xy;
    let mean = m1 / 9.0;
    let sigma = sqrt(max(m2 / 9.0 - mean * mean, vec3(0.0)));
    // Wide enough that a thin feature visible at only some jitter offsets (a far trunk, a gap of sky
    // in a crown) settles into its average instead of being reset each frame. Upscaling widens it:
    // with a longer jitter cycle and bigger render pixels, more frames miss such a feature, and a
    // tight clamp threw its history away on each of them, so it flashed back in.
    let width = mix(3.0, 1.6, clamp((globals.upscale.x - 0.5) / 0.5, 0.0, 1.0));
    let lo = mean - sigma * width;
    let hi = mean + sigma * width;

    // Where this pixel was last frame. A sample landing on the pixel counts fully; one half a
    // render pixel away much less, so upscaled pixels gather detail over the jitter sequence.
    let ndc = vec2(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0);
    let prev = globals.taa_reproject * vec4(ndc, depth, 1.0);
    var feedback = globals.taa.y * mix(0.2, 1.0, closest);
    var past = now;
    if globals.taa.x > 0.5 && prev.w > 1e-6 {
        let pp = prev.xy / prev.w - motion;
        let puv = vec2(pp.x * 0.5 + 0.5, 0.5 - pp.y * 0.5);
        if all(puv >= vec2(0.0)) && all(puv <= vec2(1.0)) {
            past = clamp(squash(sample_history(puv)), lo, hi);
        } else {
            feedback = 1.0;
        }
    } else {
        feedback = 1.0;
    }
    return vec4(unsquash(mix(past, now, feedback)), 1.0);
}
