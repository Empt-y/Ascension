//! WGSL loading. Debug builds read `assets/shaders` from disk so F5 can hot-reload them; release builds
//! embed the same files. `common.wgsl` is prepended to every shader.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};

const EMBEDDED: &[(&str, &str)] = &[
    ("common", include_str!("../../../assets/shaders/common.wgsl")),
    ("terrain", include_str!("../../../assets/shaders/terrain.wgsl")),
    ("sky", include_str!("../../../assets/shaders/sky.wgsl")),
    ("post", include_str!("../../../assets/shaders/post.wgsl")),
    ("bloom", include_str!("../../../assets/shaders/bloom.wgsl")),
];

pub struct ShaderLib {
    dir: Option<PathBuf>,
    /// Replacement source for one shader, for tests.
    override_src: Option<(String, String)>,
}

impl Default for ShaderLib {
    fn default() -> Self {
        Self::new()
    }
}

impl ShaderLib {
    pub fn new() -> Self {
        let dir = cfg!(debug_assertions)
            .then(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/shaders"))
            .filter(|d| d.is_dir());
        if let Some(d) = &dir {
            log::info!("shaders load from {} (F5 reloads)", d.display());
        }
        Self { dir, override_src: None }
    }

    /// A library where `name` compiles from `src` instead of its file. For tests.
    pub fn with_override(name: &str, src: &str) -> Self {
        Self { override_src: Some((name.to_owned(), src.to_owned())), ..Self::new() }
    }

    pub fn hot_reload_available(&self) -> bool {
        self.dir.is_some()
    }

    fn raw(&self, name: &str) -> Result<String> {
        if let Some((n, src)) = &self.override_src
            && n == name
        {
            return Ok(src.clone());
        }
        if let Some(dir) = &self.dir {
            let path = dir.join(format!("{name}.wgsl"));
            return std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()));
        }
        match EMBEDDED.iter().find(|(n, _)| *n == name) {
            Some((_, src)) => Ok((*src).to_owned()),
            None => bail!("no embedded shader named {name}"),
        }
    }

    /// Source for `name` with the common prelude in front.
    pub fn source(&self, name: &str) -> Result<String> {
        Ok(format!("{}\n{}", self.raw("common")?, self.raw(name)?))
    }

    /// Compile a shader, returning validation errors instead of panicking.
    pub fn module(&self, device: &wgpu::Device, name: &str) -> Result<wgpu::ShaderModule> {
        let src = self.source(name)?;
        let scope = device.push_error_scope(wgpu::ErrorFilter::Validation);
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(name),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        if let Some(err) = pollster::block_on(scope.pop()) {
            bail!("shader {name}: {err}");
        }
        Ok(module)
    }
}
