//! Finding the `assets` directory and loading the building-part and nature models from it.

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

/// One entry of `nature.toml`: a scatter model and where it grows.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct NatureSpec {
    pub id: String,
    pub kind: String,
    pub biomes: Vec<String>,
}

/// How one kind (tree, grass, ...) is scattered and drawn; see `nature.toml`.
#[derive(Clone, Debug, serde::Deserialize)]
pub struct KindSpec {
    pub distance: f32,
    pub fade: f32,
    pub scale: [f32; 2],
    pub sway: f32,
    pub max_slope: f32,
    /// Casts sun shadows out to this distance (0 for none).
    pub shadow_distance: f32,
    pub collide: Collide,
    /// Beyond this distance the kind is drawn as its impostor (`<id>_far.glb`), if it has one.
    #[serde(default)]
    pub lod: Option<f32>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Collide {
    Trunk,
    Rock,
    None,
}

#[derive(serde::Deserialize)]
struct NatureFile {
    kinds: std::collections::BTreeMap<String, KindSpec>,
    density: std::collections::BTreeMap<String, std::collections::BTreeMap<String, f32>>,
    model: Vec<NatureSpec>,
}

/// A loaded nature model: its spec entry, renderer id and model-space bounds.
pub struct NatureModel {
    pub spec: NatureSpec,
    pub id: ModelId,
    pub min: glam::Vec3,
    pub max: glam::Vec3,
}

/// The natural scatter models (trees, plants, rocks...) plus how each kind is scattered.
pub struct NatureModels {
    pub models: Vec<NatureModel>,
    pub kinds: std::collections::BTreeMap<String, KindSpec>,
    /// Instances per square metre, `density[biome][kind]`.
    pub density: std::collections::BTreeMap<String, std::collections::BTreeMap<String, f32>>,
}

impl NatureModels {
    pub fn load(renderer: &mut Renderer) -> Result<Self> {
        let root = root()?;
        let spec_path = root.join("data").join("nature.toml");
        let file: NatureFile = toml::from_str(
            &std::fs::read_to_string(&spec_path)
                .with_context(|| format!("reading {}", spec_path.display()))?,
        )
        .with_context(|| format!("parsing {}", spec_path.display()))?;
        let mut models = Vec::new();
        let mut triangles = 0;
        for spec in file.model {
            anyhow::ensure!(file.kinds.contains_key(&spec.kind), "{}: unknown kind {}", spec.id, spec.kind);
            for b in &spec.biomes {
                anyhow::ensure!(asc_world::Biome::from_name(b).is_some(), "{}: unknown biome {b}", spec.id);
            }
            let path = root.join("models").join("nature").join(format!("{}.glb", spec.id));
            let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            let data = asc_render::models::load_glb(&bytes)
                .with_context(|| format!("loading {}", path.display()))?;
            triangles += data.triangles();
            let id = renderer.add_model(&data);
            let far_path = root.join("models").join("nature").join(format!("{}_far.glb", spec.id));
            if let Some(distance) = file.kinds[&spec.kind].lod
                && far_path.is_file()
            {
                let bytes =
                    std::fs::read(&far_path).with_context(|| format!("reading {}", far_path.display()))?;
                let far = asc_render::models::load_glb(&bytes)
                    .with_context(|| format!("loading {}", far_path.display()))?;
                let far = renderer.add_model(&far);
                renderer.set_lod(id, far, distance);
            }
            models.push(NatureModel { spec, id, min: data.min, max: data.max });
        }
        for biome in file.density.keys() {
            anyhow::ensure!(asc_world::Biome::from_name(biome).is_some(), "density: unknown biome {biome}");
        }
        log::info!("loaded {} nature models ({triangles} triangles)", models.len());
        Ok(Self { models, kinds: file.kinds, density: file.density })
    }
}

/// Decode the Blender preview renders into egui textures, `[tier][kind]`.
pub fn load_previews(ctx: &egui::Context, spec: &BuildingSpec) -> Result<Vec<[egui::TextureHandle; 3]>> {
    let dir = root()?.join("models").join("previews");
    let mut out = Vec::new();
    for tier in 0..spec.tiers.len() {
        let mut row = Vec::new();
        for kind in PartKind::ALL {
            let name = format!("{}_{}.png", spec.tiers[tier].id, kind.id());
            let bytes = std::fs::read(dir.join(&name)).with_context(|| format!("reading preview {name}"))?;
            row.push(ctx.load_texture(name, decode_png(&bytes)?, egui::TextureOptions::LINEAR));
        }
        out.push(row.try_into().map_err(|_| anyhow::anyhow!("three parts per tier"))?);
    }
    Ok(out)
}

fn decode_png(bytes: &[u8]) -> Result<egui::ColorImage> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size().context("preview too large")?];
    let info = reader.next_frame(&mut buf)?;
    let pixels = &buf[..info.buffer_size()];
    let size = [info.width as usize, info.height as usize];
    Ok(match info.color_type {
        png::ColorType::Rgba => egui::ColorImage::from_rgba_unmultiplied(size, pixels),
        png::ColorType::Rgb => egui::ColorImage::from_rgb(size, pixels),
        other => anyhow::bail!("unsupported preview colour type {other:?}"),
    })
}
