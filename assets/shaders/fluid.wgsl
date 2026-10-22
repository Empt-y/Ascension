// Simulated water near the camera: surface-nets meshes of the fluid cells of every materialised
// chunk (the rest of the water is the flat sea in water.wgsl). Drawn after the sky and the sea,
// blended, without writing depth; both faces, so it reads from underwater too.

@group(0) @binding(1) var<storage, read> chunk_offsets: array<vec4<f32>>;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec4<f32>,
    // Material weights: slot 0 water, slot 1 lava (see the client's fluid mesher).
    @location(2) weights_a: vec4<f32>,
    @builtin(instance_index) instance: u32,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
    @location(1) @interpolate(perspective, centroid) normal: vec3<f32>,
    @location(2) @interpolate(perspective, centroid) lava: f32,
}

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.rel = in.position + chunk_offsets[in.instance].xyz;
    out.clip = globals.view_proj * vec4(out.rel, 1.0);
    out.normal = in.normal.xyz;
    out.lava = in.weights_a.y / max(in.weights_a.x + in.weights_a.y, 1e-4);
    return out;
}

fn ripples(xz: vec2<f32>, t: f32) -> vec2<f32> {
    let p1 = vec3(xz * 0.35 + vec2(t * 0.4, t * 0.25), t * 0.3);
    let p2 = vec3(xz * 1.1 - vec2(t * 0.6, -t * 0.2), t * 0.5);
    let e = 0.15;
    let n1 = value_noise(p1);
    let n2 = value_noise(p2);
    let dx = (value_noise(p1 + vec3(e, 0.0, 0.0)) - n1) + (value_noise(p2 + vec3(e, 0.0, 0.0)) - n2) * 0.5;
    let dz = (value_noise(p1 + vec3(0.0, e, 0.0)) - n1) + (value_noise(p2 + vec3(0.0, e, 0.0)) - n2) * 0.5;
    return vec2(dx, dz) * 0.9;
}

// Molten rock: a glowing orange body under a dark crust that drifts and cracks slowly.
fn lava_color(world: vec3<f32>, t: f32) -> vec3<f32> {
    let flow = vec3(world.x * 0.35 + t * 0.05, world.y * 0.2, world.z * 0.35 - t * 0.03);
    let crust = smoothstep(0.45, 0.7, value_noise(flow) * 0.65 + value_noise(flow * 2.7 + vec3(t * 0.1)) * 0.35);
    let hot = vec3(6.0, 1.7, 0.25) * (0.85 + 0.3 * value_noise(world * 1.3 + vec3(0.0, t * 0.4, 0.0)));
    let dark = vec3(0.06, 0.02, 0.012);
    return mix(hot, dark, crust);
}

@fragment
fn fs_main(in: VsOut, @builtin(front_facing) front: bool) -> @location(0) vec4<f32> {
    let world = in.rel + globals.camera.xyz;
    if in.lava > 0.5 {
        // Lava is opaque and gives its own light (it blooms); fog still covers it at a distance.
        return vec4(debug_hot(apply_fog(lava_color(world, globals.camera.w), in.rel), 7.0), 1.0);
    }
    let dist = length(in.rel);
    let v = -in.rel / dist;
    // The mesh normal points out of the water; seen from inside it, flip it towards the eye.
    var n = normalize(in.normal);
    // Ripples on surfaces facing up (not on the sides of falling water).
    let flat = smoothstep(0.6, 0.9, abs(n.y));
    let r = ripples(world.xz, globals.camera.w) * flat * (1.0 - smoothstep(30.0, 200.0, dist));
    n = normalize(n + vec3(-r.x, 0.0, -r.y));
    if dot(n, v) < 0.0 {
        n = -n;
    }
    let ndv = clamp(dot(n, v), 0.0, 1.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    let reflected = sky_base(reflect(-v, n));
    let body = vec3(0.04, 0.16, 0.17) * (0.25 + 0.75 * daylight());
    var color = mix(body, reflected, fresnel);
    let h = normalize(v + sun_dir());
    color += sun_light() * pow(max(dot(n, h), 0.0), 300.0) * 4.0 * flat;
    let alpha = clamp(0.62 + 0.35 * fresnel, 0.0, 0.95);
    return vec4(debug_hot(apply_fog(color, in.rel), 6.0), alpha);
}
