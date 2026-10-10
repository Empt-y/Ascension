//! Compiles every pipeline on a real (headless) GPU. Skips quietly on machines without one.

use asc_render::{ShaderLib, build_pipelines_headless};

fn device() -> Option<wgpu::Device> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        // The game runs on the discrete GPU; this laptop's Intel Vulkan driver crashes headless.
        power_preference: wgpu::PowerPreference::HighPerformance,
        ..Default::default()
    }))
    .ok()?;
    let (device, _queue) =
        pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor::default())).ok()?;
    Some(device)
}

#[test]
fn all_pipelines_build() {
    let Some(device) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    build_pipelines_headless(&device, &ShaderLib::new()).expect("pipelines should build");
}

#[test]
fn a_broken_shader_is_an_error_not_a_crash() {
    let Some(device) = device() else {
        eprintln!("no GPU adapter; skipping");
        return;
    };
    let lib = ShaderLib::with_override("terrain", "this is not wgsl {");
    let err = build_pipelines_headless(&device, &lib).expect_err("broken shader must fail");
    assert!(format!("{err:#}").contains("terrain"), "error should name the shader: {err:#}");
}
