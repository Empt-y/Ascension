// Shared by every shader: frame globals, noise, sky colour and fog.
// The renderer prepends this file to each shader before compiling it.

struct Globals {
    // Camera-relative: world positions are offset by -camera before projection.
    view_proj: mat4x4<f32>,
    inv_view_proj: mat4x4<f32>,
    // Sun shadow cascades, nearest first: two sets of three (layer = set * 3 + cascade).
    shadow_view_proj: array<mat4x4<f32>, 6>,
    // xyz = camera world position, w = seconds since start.
    camera: vec4<f32>,
    // xyz = direction towards the sun, w = 0..1 how far above the horizon it is.
    sun: vec4<f32>,
    // x = fog density, y = fog height falloff, z = view distance (fade to sky), w = exposure.
    fog: vec4<f32>,
    // xy = the scene's render size in pixels, zw = 1 / size. Below the window size when upscaling.
    screen: vec4<f32>,
    // xyz = brush centre (camera-relative), w = brush radius (0 = hidden).
    brush: vec4<f32>,
    // x = distance at which everything has faded into the sky (far terrain range, or the chunk radius
    // when far terrain is off).
    far: vec4<f32>,
    // x = the layer being rendered (shadow passes only), y = 1 if shadows are on, z = wind strength,
    // w = how far the current set has faded in over the previous one.
    shadow: vec4<f32>,
    // xyz = half-width in metres of each cascade, w = the current set.
    shadow_radii: vec4<f32>,
    // TAA: this frame's (unjittered) NDC and depth to last frame's clip space.
    taa_reproject: mat4x4<f32>,
    // x = 1 if the history is usable, y = weight of the new frame, zw = this frame's jitter in NDC.
    taa: vec4<f32>,
    // Last frame's x = time (camera.w) and y = wind strength, for plants' motion vectors.
    motion: vec4<f32>,
    // xy = the window (output) size in pixels, zw = 1 / size.
    output: vec4<f32>,
    // x = render scale (render size / output size), y = RCAS sharpness (0 = none, 1 = most).
    upscale: vec4<f32>,
}

@group(0) @binding(0) var<uniform> globals: Globals;
// Sun shadow map, used by anything drawn in the scene pass with the scene layout.
@group(0) @binding(2) var shadow_map: texture_depth_2d_array;
@group(0) @binding(3) var shadow_sampler: sampler_comparison;

const PI: f32 = 3.14159265;

fn pcg(v: u32) -> u32 {
    let state = v * 747796405u + 2891336453u;
    let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

// Hash of an integer lattice point to [0, 1).
fn hash3i(p: vec3<i32>) -> f32 {
    let h = pcg(bitcast<u32>(p.x) ^ pcg(bitcast<u32>(p.y) ^ pcg(bitcast<u32>(p.z))));
    return f32(h >> 8u) / 16777216.0;
}

fn hash13(p: vec3<f32>) -> f32 {
    return hash3i(vec3<i32>(floor(p)));
}

fn value_noise(p: vec3<f32>) -> f32 {
    let i = vec3<i32>(floor(p));
    let f = fract(p);
    // Quintic fade hides the lattice better than smoothstep.
    let u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
    let a = mix(hash3i(i + vec3(0, 0, 0)), hash3i(i + vec3(1, 0, 0)), u.x);
    let b = mix(hash3i(i + vec3(0, 1, 0)), hash3i(i + vec3(1, 1, 0)), u.x);
    let c = mix(hash3i(i + vec3(0, 0, 1)), hash3i(i + vec3(1, 0, 1)), u.x);
    let d = mix(hash3i(i + vec3(0, 1, 1)), hash3i(i + vec3(1, 1, 1)), u.x);
    return mix(mix(a, b, u.y), mix(c, d, u.y), u.z);
}

fn fbm3(p: vec3<f32>) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var q = p;
    for (var i = 0; i < 3; i++) {
        sum += value_noise(q) * amp;
        q = q * 2.03 + vec3(1.7, 9.2, 4.1);
        amp *= 0.5;
    }
    return sum / 0.875;
}

fn sun_dir() -> vec3<f32> {
    return globals.sun.xyz;
}

// 0 at night, 1 in full daylight.
fn daylight() -> f32 {
    return smoothstep(-0.08, 0.25, sun_dir().y);
}

// Peaks while the sun is near the horizon.
fn sunset_amount() -> f32 {
    let s = sun_dir().y;
    return smoothstep(-0.25, 0.0, s) * (1.0 - smoothstep(0.0, 0.4, s));
}

// Colour of direct sunlight reaching the ground (HDR).
fn sun_light() -> vec3<f32> {
    let warm = vec3(1.0, 0.55, 0.28);
    let white = vec3(1.0, 0.95, 0.86);
    let c = mix(white, warm, sunset_amount());
    let y = sun_dir().y;
    // A low sun shines through far more air, so it dims well before it sets (atmospheric
    // extinction); otherwise walls facing a sunset blow out to white.
    let extinction = mix(0.3, 1.0, smoothstep(0.0, 0.5, y));
    return c * 3.2 * extinction * smoothstep(-0.05, 0.12, y);
}

// Stylised sky: gradient from horizon to zenith, warm glow around the sun at dusk, no sun disk.
fn sky_base(dir: vec3<f32>) -> vec3<f32> {
    let day = daylight();
    let dusk = sunset_amount();
    let zenith = mix(vec3(0.004, 0.006, 0.02), vec3(0.10, 0.26, 0.75), day);
    var horizon = mix(vec3(0.015, 0.02, 0.05), vec3(0.55, 0.72, 0.95), day);
    horizon = mix(horizon, vec3(1.0, 0.45, 0.22), dusk * 0.85);
    let up = max(dir.y, 0.0);
    var col = mix(horizon, zenith, pow(up, 0.45));
    // Below the horizon fade to a dim haze so the far edge of the world dissolves.
    col = mix(col, horizon * 0.6, smoothstep(0.0, -0.25, dir.y));
    let mu = max(dot(dir, sun_dir()), 0.0);
    let glow_col = mix(vec3(1.0, 0.85, 0.6), vec3(1.0, 0.4, 0.15), dusk);
    col += glow_col * (pow(mu, 8.0) * 0.35 + pow(mu, 64.0) * 0.8) * (day * 0.6 + dusk);
    return col;
}

// Fraction of light lost to fog along a camera-relative ray (analytic exponential height fog).
fn fog_amount(rel: vec3<f32>) -> f32 {
    let dist = length(rel);
    let dir = rel / max(dist, 1e-4);
    let density = globals.fog.x;
    let falloff = globals.fog.y;
    let cam_y = globals.camera.y;
    let fy = falloff * dir.y * dist;
    var integral = dist;
    if abs(fy) > 1e-4 {
        integral = (1.0 - exp(-fy)) / (falloff * dir.y);
    }
    let optical = density * exp(-falloff * max(cam_y, -50.0)) * integral;
    return 1.0 - exp(-max(optical, 0.0));
}

// Blend a surface colour toward the sky along the view ray, and fully into it at the far range so
// the edge of the world never shows.
fn apply_fog(color: vec3<f32>, rel: vec3<f32>) -> vec3<f32> {
    let dir = normalize(rel);
    let fog_col = sky_base(dir);
    let f = fog_amount(rel);
    let horizontal = length(rel.xz);
    let edge = smoothstep(globals.far.x * 0.75, globals.far.x * 0.97, horizontal);
    return mix(color, fog_col, max(f, edge));
}

// ---- Terrain materials and lighting ----

struct Surface {
    albedo: vec3<f32>,
    roughness: f32,
    metallic: f32,
}

fn triplanar(p: vec3<f32>, w_in: vec3<f32>) -> f32 {
    // Skip projections that barely contribute (most surfaces need one or two), fading them out
    // smoothly: a hard cut-off is a tiny step in the value, which bump mapping turns into a line.
    let w = w_in * smoothstep(vec3(0.01), vec3(0.06), w_in);
    var sum = 0.0;
    if w.x > 0.0 {
        sum += value_noise(p.yzx) * w.x;
    }
    if w.y > 0.0 {
        sum += value_noise(p.xzy) * w.y;
    }
    if w.z > 0.0 {
        sum += value_noise(p) * w.z;
    }
    return sum / max(w.x + w.y + w.z, 1e-4);
}

// Surface detail shared by the albedo and the bump-mapped normal.
struct Detail {
    // Roughly -0.5..0.5 brightness variation.
    tone: f32,
    // Normal with the fine relief applied.
    normal: vec3<f32>,
    // How much that normal varies across the pixel, as extra GGX alpha squared: pass the material's
    // roughness through `aa_roughness` with it.
    widen: f32,
}

// Specular anti-aliasing (Tokuyoshi & Kaplanyan): a pixel covering bumps or curvature sees a spread
// of normals, which a sharp highlight samples at one point and so flickers between frames. Widening
// the lobe by that spread gives the pixel the average highlight instead.
fn aa_roughness(roughness: f32, widen: f32) -> f32 {
    let a = roughness * roughness;
    return sqrt(sqrt(clamp(a * a + widen, 0.0, 1.0)));
}

const DETAIL_OCTAVES: u32 = 5u;
const DETAIL_FREQ: array<f32, 5> = array(0.45, 1.3, 4.1, 12.0, 36.0);
const DETAIL_TONE: array<f32, 5> = array(0.30, 0.34, 0.26, 0.18, 0.12);

// Procedural detail filtered by pixel footprint: each octave fades out as its wavelength approaches
// the size of a pixel on the surface (the procedural equivalent of mipmapping), so near ground is
// crisp and distant or grazing ground is smooth rather than shimmering. The same octaves, as a small
// height field, bump the normal using screen-space derivatives (Mikkelsen's surface gradient).
//
// Must be called in uniform control flow (before any discard): it takes derivatives.
fn surface_detail(world: vec3<f32>, n: vec3<f32>) -> Detail {
    let dpx = dpdx(world);
    let dpy = dpdy(world);
    // At grazing angles a pixel's footprint is long one way and short the other. Filtering colour
    // by the long side (isotropic) blurs everything; the geometric mean keeps detail along the short
    // side, the procedural cousin of anisotropic filtering. Relief keeps the stricter long side.
    let footprint = max(length(dpx), length(dpy));
    let tone_footprint = sqrt(length(dpx) * length(dpy));
    var tw = pow(abs(n), vec3(4.0));
    tw /= (tw.x + tw.y + tw.z);
    var tone = 0.0;
    var height = 0.0;
    for (var i = 0u; i < DETAIL_OCTAVES; i++) {
        let f = DETAIL_FREQ[i];
        // Colour detail survives down to a wavelength of about 2-5 pixels; relief only to about 4-12,
        // because screen-space derivatives come per 2x2 pixel quad and turn finer bumps blocky.
        let keep = 1.0 - smoothstep(0.2, 0.55, tone_footprint * f);
        let relief = 1.0 - smoothstep(0.08, 0.25, footprint * f);
        if keep > 0.0 {
            let v = triplanar(world * f + f32(i) * 17.31, tw) - 0.5;
            tone += v * DETAIL_TONE[i] * keep;
            // Relief proportional to wavelength keeps slopes similar at every scale.
            height += v * (0.22 / f) * relief;
        }
    }
    let dhx = dpdx(height);
    let dhy = dpdy(height);
    let r1 = cross(dpy, n);
    let r2 = cross(n, dpx);
    let det = dot(dpx, r1);
    let grad = sign(det) * (dhx * r1 + dhy * r2);
    var d: Detail;
    d.tone = tone;
    d.normal = normalize(abs(det) * n - grad);
    if abs(det) < 1e-12 {
        d.normal = n;
    }
    let dnx = dpdx(d.normal);
    let dny = dpdy(d.normal);
    d.widen = min(0.5 * (dot(dnx, dnx) + dot(dny, dny)), 0.18);
    return d;
}

// Base colours are linear and deliberately saturated. Materials blend by the interpolated weights:
// wa = grass, soil, rock, sand; wb = snow, ore, dry grass, moss; wc = mud, ice, sandstone.
// `ore` is how the ore there looks: rgb = square root of its linear colour, a = metallic (high
// nibble) and roughness (low nibble), as packed by OreTable::look.
fn terrain_surface(wa: vec4<f32>, wb: vec4<f32>, wc: vec4<f32>, ore: vec4<f32>, world: vec3<f32>, n: vec3<f32>, tone: f32) -> Surface {
    let broad = fbm3(world * 0.035);
    let detail = 1.0 + tone;
    let fine = 0.5 + tone;

    let soil = mix(vec3(0.22, 0.13, 0.07), vec3(0.30, 0.19, 0.10), broad);
    let lush = mix(vec3(0.10, 0.34, 0.06), vec3(0.24, 0.42, 0.05), broad);
    // Grass gives way to soil on slopes.
    let grass = mix(soil, lush, smoothstep(0.62, 0.82, n.y));
    let rock = mix(vec3(0.24, 0.23, 0.24), vec3(0.36, 0.32, 0.29), broad);
    let sand = mix(vec3(0.34, 0.28, 0.18), vec3(0.40, 0.33, 0.22), broad);
    let snow = vec3(0.85, 0.88, 0.95);
    // Ore: clumps and streaks of the mineral through darker host rock.
    let ore_color = ore.rgb * ore.rgb;
    let packed = round(ore.a * 255.0);
    let ore_metal = floor(packed / 16.0) / 15.0;
    let ore_rough = (packed - floor(packed / 16.0) * 16.0) / 15.0;
    // (Only where there is ore: the noise is the dearest part of the terrain's shading.)
    var fleck = 0.0;
    if wb.y > 0.0 {
        let clump = value_noise(world * vec3(1.6, 3.2, 1.6)) * 0.7 + value_noise(world * 5.0) * 0.3;
        fleck = smoothstep(0.42, 0.56, clump);
    }
    let ore_albedo = mix(rock * 0.75, ore_color, fleck);
    let flat = smoothstep(0.62, 0.82, n.y);
    let dry = mix(soil, mix(vec3(0.36, 0.29, 0.11), vec3(0.47, 0.38, 0.15), broad), flat);
    let moss = mix(soil, mix(vec3(0.11, 0.16, 0.05), vec3(0.21, 0.22, 0.09), broad), flat);
    let mud = mix(vec3(0.09, 0.07, 0.045), vec3(0.15, 0.11, 0.065), broad);
    let ice = mix(vec3(0.50, 0.68, 0.86), vec3(0.62, 0.78, 0.92), broad);
    let sandstone = mix(vec3(0.44, 0.28, 0.16), vec3(0.56, 0.38, 0.23), broad);

    let total = max(dot(wa, vec4(1.0)) + dot(wb, vec4(1.0)) + dot(wc, vec4(1.0)), 1e-4);
    var s: Surface;
    s.albedo = (grass * wa.x + soil * wa.y + rock * wa.z + sand * wa.w + snow * wb.x + ore_albedo * wb.y
        + dry * wb.z + moss * wb.w + mud * wc.x + ice * wc.y + sandstone * wc.z) / total;
    s.roughness = (0.9 * (wa.x + wa.y + wa.w + wb.z + wb.w) + 0.75 * wa.z + 0.5 * wb.x
        + mix(0.8, max(ore_rough, 0.2), fleck) * wb.y + 0.45 * wc.x + 0.12 * wc.y + 0.85 * wc.z) / total;
    s.metallic = ore_metal * fleck * wb.y / total;
    s.albedo *= detail;
    return s;
}

// GGX specular lobe times n.l, without Fresnel (the caller applies it, so metals can tint it).
fn ggx_lobe(n: vec3<f32>, v: vec3<f32>, l: vec3<f32>, roughness: f32) -> f32 {
    let h = normalize(v + l);
    let a = roughness * roughness;
    let a2 = a * a;
    let ndh = max(dot(n, h), 0.0);
    let d = a2 / (PI * pow(ndh * ndh * (a2 - 1.0) + 1.0, 2.0));
    let ndl = max(dot(n, l), 0.0);
    let ndv = max(dot(n, v), 1e-3);
    let k = a * 0.5;
    let g = (ndl / (ndl * (1.0 - k) + k)) * (ndv / (ndv * (1.0 - k) + k));
    return d * g / max(4.0 * ndl * ndv, 1e-3) * ndl;
}

fn fresnel_schlick(f0: vec3<f32>, v: vec3<f32>, h: vec3<f32>) -> vec3<f32> {
    return f0 + (1.0 - f0) * pow(1.0 - max(dot(h, v), 0.0), 5.0);
}

// Depth range of every cascade, metres (matches SHADOW_DEPTH in the renderer).
const SHADOW_DEPTH: f32 = 500.0;
// Shadow edges are this soft in the world (metres), whichever cascade draws them.
const PENUMBRA: f32 = 0.07;
const FOLIAGE_PENUMBRA: f32 = 0.6;

// The matrix of the layer a shadow pass is rendering.
fn shadow_caster_matrix() -> mat4x4<f32> {
    return globals.shadow_view_proj[u32(globals.shadow.x)];
}

// A point's place in cascade c of layer `layer`: xy = map coordinates, z = depth, w = how far out
// (1 = edge).
fn cascade_coords(layer: u32, c: u32, rel: vec3<f32>, n: vec3<f32>) -> vec4<f32> {
    let texel = 2.0 * globals.shadow_radii[c] / f32(textureDimensions(shadow_map).x);
    // Normal offset of a couple of texels hides acne without detaching shadows from their casters.
    let p = globals.shadow_view_proj[layer] * vec4(rel + n * texel * 2.5, 1.0);
    return vec4(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5, p.z, max(abs(p.x), abs(p.y)));
}

// Percentage-closer filtering over a Poisson disc sized to PENUMBRA, each tap itself bilinear.
fn cascade_pcf(layer: u32, c: u32, s: vec4<f32>, taps: u32, penumbra: f32) -> f32 {
    if s.z > 1.0 {
        return 1.0;
    }
    var disc = array<vec2<f32>, 12>(
        vec2(-0.326, -0.406), vec2(0.962, -0.195), vec2(-0.840, -0.074), vec2(0.519, 0.767),
        vec2(-0.696, 0.457), vec2(0.185, -0.893), vec2(0.507, 0.064), vec2(-0.321, -0.933),
        vec2(-0.203, 0.621), vec2(0.896, 0.412), vec2(0.473, -0.480), vec2(-0.792, -0.598),
    );
    let size = f32(textureDimensions(shadow_map).x);
    let texel = 2.0 * globals.shadow_radii[c] / size;
    let bias = texel * 1.5 / SHADOW_DEPTH;
    let radius = max(1.0, penumbra / texel) / size;
    var lit = 0.0;
    for (var i = 0u; i < taps; i++) {
        lit += textureSampleCompareLevel(shadow_map, shadow_sampler, s.xy + disc[i] * radius, i32(layer), s.z - bias);
    }
    return lit / f32(taps);
}

// Sun visibility from one set of cascades: the nearest cascade that covers the point, blended into
// the next one over its outer edge (and into no shadow past the last), so there are no seams.
fn shadow_lookup_set(group: u32, rel: vec3<f32>, n: vec3<f32>, taps: u32, penumbra: f32) -> f32 {
    for (var c = 0u; c < 3u; c++) {
        let layer = group * 3u + c;
        let s = cascade_coords(layer, c, rel, n);
        if s.w < 1.0 {
            var lit = cascade_pcf(layer, c, s, taps, penumbra);
            // A wide blend into the next cascade, so the change in detail has no visible edge.
            let blend = smoothstep(0.6, 1.0, s.w);
            if blend > 0.0 {
                if c < 2u {
                    lit = mix(lit, cascade_pcf(layer + 1u, c + 1u, cascade_coords(layer + 1u, c + 1u, rel, n), taps, penumbra), blend);
                } else {
                    lit = mix(lit, 1.0, blend);
                }
            }
            return lit;
        }
    }
    return 1.0;
}

// Sun visibility at a point: the current cascade set, dissolving in over the previous one after
// each step of the sun.
fn shadow_lookup(rel: vec3<f32>, n: vec3<f32>, taps: u32, penumbra: f32) -> f32 {
    if globals.shadow.y < 0.5 {
        return 1.0;
    }
    let current = u32(globals.shadow_radii.w);
    var lit = shadow_lookup_set(current, rel, n, taps, penumbra);
    let fade = globals.shadow.w;
    if fade < 1.0 {
        lit = mix(shadow_lookup_set(1u - current, rel, n, taps, penumbra), lit, fade);
    }
    return lit;
}

fn shadow_factor(rel: vec3<f32>, n: vec3<f32>) -> f32 {
    return shadow_lookup(rel, n, 8u, PENUMBRA);
}

// Foliage (looked up per vertex): a wide, soft penumbra. Leaves shadow each other in leaf-sized
// patches, so a crisp lookup lit or darkened single leaves at random; as they fluttered, that
// speckle of bright and dark leaves shuffled and glinted. Soft, a bush shades smoothly from its
// sunlit side into its middle.
fn shadow_factor_foliage(rel: vec3<f32>, n: vec3<f32>) -> f32 {
    return shadow_lookup(rel, n, 8u, FOLIAGE_PENUMBRA);
}

// Sun, sky ambient and moonlight on any opaque surface (metallic-roughness PBR, stylised ambient).
// Shared by terrain, far terrain and building parts so they all sit in the same light.
fn shade_pbr(albedo: vec3<f32>, roughness: f32, metallic: f32, n: vec3<f32>, rel: vec3<f32>, occlusion: f32, shadow: f32) -> vec3<f32> {
    let l = sun_dir();
    let v = normalize(-rel);
    let ndl = max(dot(n, l), 0.0);
    let sun = sun_light() * shadow;
    // Hemisphere ambient from the sky above and bounced light from the ground below.
    let sky_col = sky_base(normalize(vec3(n.x, max(n.y, 0.2), n.z)));
    // Desaturate the sky light a little so shade isn't too blue.
    let sky_amb = mix(vec3(dot(sky_col, vec3(0.2126, 0.7152, 0.0722))), sky_col, 0.6) * 0.8;
    let ground_amb = vec3(0.10, 0.08, 0.06) * (0.15 + 0.85 * daylight());
    let ambient = mix(ground_amb, sky_amb, n.y * 0.5 + 0.5) * occlusion;
    // Faint, unshadowed moonlight from opposite the sun so nights aren't black.
    let moon = vec3(0.05, 0.065, 0.12) * (1.0 - daylight()) * max(dot(n, -l), 0.0);
    let diffuse = albedo * (1.0 - metallic);
    let f0 = mix(vec3(0.04), albedo, metallic);
    let fres = fresnel_schlick(f0, v, normalize(v + l));
    var color = diffuse * (sun * ndl + ambient + moon);
    // Metals have no diffuse, so give them a dim reflection of the ambient instead.
    color += f0 * ambient * metallic * 0.8;
    color += sun * fres * ggx_lobe(n, v, l, roughness);
    return color;
}

fn shade_terrain(s: Surface, n: vec3<f32>, rel: vec3<f32>, occlusion: f32, shadow: f32) -> vec3<f32> {
    return shade_pbr(s.albedo, s.roughness, s.metallic, n, rel, occlusion, shadow);
}

// Full-screen triangle from the vertex index.
fn fullscreen_pos(index: u32) -> vec4<f32> {
    let uv = vec2(f32((index << 1u) & 2u), f32(index & 2u));
    return vec4(uv * 2.0 - 1.0, 0.0, 1.0);
}
