// Water and lava near the camera: surface-nets meshes of each loaded chunk's fluid, natural (from
// worldgen, built with the chunk) or simulated (materialised chunks). The water beyond the loaded
// chunks is water.wgsl. Drawn after the sky, blended; both faces, so it reads from underwater too.
// Natural river water carries its flow per vertex; simulated water drifts down its surface's tilt.

@group(0) @binding(1) var<storage, read> chunk_offsets: array<vec4<f32>>;

struct VsIn {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec4<f32>,
    // Material weights: slot 0 water, slot 1 lava (see the client's fluid mesher).
    @location(2) weights_a: vec4<f32>,
    // Natural water: the river's flow (xy, around 0.5) and z = 1 to say it's there.
    @location(6) flow: vec4<f32>,
    @builtin(instance_index) instance: u32,
}

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) rel: vec3<f32>,
    @location(1) @interpolate(perspective, centroid) normal: vec3<f32>,
    @location(2) @interpolate(perspective, centroid) lava: f32,
    // xy: flow, z: how much of it is given (1) rather than taken from the tilt (0).
    @location(3) flow: vec3<f32>,
}

// fluid::FLOW_SCALE: bytes per m/s.
const FLOW_SCALE: f32 = 60.0;

@vertex
fn vs_main(in: VsIn) -> VsOut {
    var out: VsOut;
    out.rel = in.position + chunk_offsets[in.instance].xyz;
    out.clip = globals.view_proj * vec4(out.rel, 1.0);
    out.normal = in.normal.xyz;
    out.lava = in.weights_a.y / max(in.weights_a.x + in.weights_a.y, 1e-4);
    out.flow = vec3((in.flow.xy - vec2(128.0 / 255.0)) * 255.0 / FLOW_SCALE, in.flow.z);
    return out;
}

// Ripples carried downstream by `drift` (round, not stretched: streaks lying along the flow look
// still as they slide along themselves); still water just shimmers.
fn ripples(xz: vec2<f32>, t: f32, drift: vec2<f32>) -> vec2<f32> {
    let p1 = vec3(xz * 0.35 - drift * t * 0.35, t * 0.3);
    let p2 = vec3(xz * 1.1 - drift * t * 0.75, t * 0.5);
    let e = 0.15;
    let n1 = value_noise(p1);
    let n2 = value_noise(p2);
    let dx = (value_noise(p1 + vec3(e, 0.0, 0.0)) - n1) + (value_noise(p2 + vec3(e, 0.0, 0.0)) - n2) * 0.5;
    let dz = (value_noise(p1 + vec3(0.0, e, 0.0)) - n1) + (value_noise(p2 + vec3(0.0, e, 0.0)) - n2) * 0.5;
    return vec2(dx, dz) * 0.9;
}

// Molten rock: a glowing orange body under a dark crust that drifts and cracks slowly.
fn lava_color(world: vec3<f32>, t: f32) -> vec3<f32> {
    let flow = vec3(world.x * 0.6 + t * 0.05, world.y * 0.3, world.z * 0.6 - t * 0.03);
    // Plates of cooling crust with glowing cracks between them, and a brighter churn under them.
    let plates = value_noise(flow) * 0.6 + value_noise(flow * 3.1 + vec3(t * 0.08)) * 0.4;
    let crust = smoothstep(0.38, 0.55, plates);
    let glow = 0.75 + 0.5 * value_noise(world * 0.9 + vec3(0.0, t * 0.3, 0.0));
    let hot = vec3(3.2, 0.85, 0.12) * glow;
    let dark = mix(vec3(0.05, 0.018, 0.01), vec3(0.4, 0.08, 0.02), 1.0 - smoothstep(0.55, 0.8, plates));
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
    // Rivers carry their flow; simulated water runs downhill, the way its surface tilts (the
    // normal leans that way).
    let tilt = vec2(n.x, n.z);
    let tilted = tilt / max(length(tilt), 1e-3) * min(length(tilt) * 12.0, 2.0);
    let given = clamp(in.flow.z, 0.0, 1.0);
    let drift = mix(tilted, in.flow.xy, given);
    let r = ripples(world.xz, globals.camera.w, drift) * flat * (1.0 - smoothstep(30.0, 200.0, dist));
    n = normalize(n + vec3(-r.x, 0.0, -r.y));
    if dot(n, v) < 0.0 {
        n = -n;
    }
    let ndv = clamp(dot(n, v), 0.0, 1.0);
    let fresnel = 0.02 + 0.98 * pow(1.0 - ndv, 5.0);
    let reflected = sky_base(reflect(-v, n));
    let body = vec3(0.03, 0.13, 0.15) * (0.25 + 0.75 * daylight());
    var color = mix(body, reflected, fresnel);
    let h = normalize(v + sun_dir());
    color += sun_light() * pow(max(dot(n, h), 0.0), 300.0) * 4.0 * flat;
    // White water: where the surface falls steeply (waterfalls, rapids), and in patches drifting
    // down fast rivers.
    let speed = length(drift);
    let falling = 1.0 - smoothstep(0.55, 0.92, abs(normalize(in.normal).y));
    let churn = value_noise(vec3((world.xz - drift * globals.camera.w * 1.4) * 0.7, globals.camera.w * 0.6));
    let foam = max(falling * (0.55 + 0.45 * churn), smoothstep(0.72, 0.92, churn) * smoothstep(0.8, 1.8, speed) * 0.3);
    let white = vec3(0.8, 0.85, 0.88) * (0.25 + 0.75 * daylight()) * (0.6 + 0.4 * max(sun_dir().y, 0.0));
    color = mix(color, white, foam);
    let alpha = clamp(0.78 + 0.2 * fresnel + foam * 0.4, 0.0, 0.97);
    return vec4(debug_hot(apply_fog(color, in.rel), 6.0), alpha);
}
