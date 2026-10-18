//! Finding the `assets` directory and loading the building-part models from it.

use std::path::PathBuf;

use anyhow::{Context, Result};
use asc_build::{BuildingSpec, PartKind};
use asc_render::{ModelId, Renderer};

/// The assets directory: `ASC_ASSETS` if set, otherwise the nearest `assets` folder above the
/// executable (so `target/release/ascension.exe` finds the repository's), otherwise the crate's
/// source tree.
pub fn root() -> Result<PathBuf> {
    if let Ok(dir) = std::env::var("ASC_ASSETS") {
        return Ok(PathBuf::from(dir));
    }
    if let Ok(exe) = std::env::current_exe() {
        for dir in exe.ancestors().skip(1) {
            let candidate = dir.join("assets");
            if candidate.join("data").join("building.toml").is_file() {
                return Ok(candidate);
            }
        }
    }
    let dev = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets");
    if dev.join("data").join("building.toml").is_file() {
        return Ok(dev);
    }
    anyhow::bail!("can't find the assets directory (set ASC_ASSETS)")
}

/// The building spec plus the renderer's model for every (tier, part).
pub struct PartModels {
    pub spec: BuildingSpec,
    /// `models[tier][kind]`, in `PartKind::ALL` order.
    pub models: Vec<[ModelId; 3]>,
}

impl PartModels {
    pub fn load(renderer: &mut Renderer) -> Result<Self> {
        let root = root()?;
        let spec_path = root.join("data").join("building.toml");
        let spec = BuildingSpec::parse(
            &std::fs::read_to_string(&spec_path)
                .with_context(|| format!("reading {}", spec_path.display()))?,
        )?;
        let mut models = Vec::new();
        let mut triangles = 0;
        for tier in 0..spec.tiers.len() {
            let mut ids = [ModelId(0); 3];
            for (slot, kind) in PartKind::ALL.into_iter().enumerate() {
                let path = root.join("models").join(spec.model_file(tier, kind));
                let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
                let data = asc_render::models::load_glb(&bytes)
                    .with_context(|| format!("loading {}", path.display()))?;
                triangles += data.triangles();
                ids[slot] = renderer.add_model(&data);
            }
            models.push(ids);
        }
        log::info!("loaded {} part models ({triangles} triangles) from {}", models.len() * 3, root.display());
        Ok(Self { spec, models })
    }

    pub fn model(&self, tier: usize, kind: PartKind) -> ModelId {
        let slot = PartKind::ALL.iter().position(|k| *k == kind).unwrap_or(0);
        self.models[tier][slot]
    }
}
