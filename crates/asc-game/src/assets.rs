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
    /// Beyond this distance, its mid-distance model (`<id>_mid.glb`: no twigs, fewer cards).
    #[serde(default)]
    pub mid_lod: Option<f32>,
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
            let kind = &file.kinds[&spec.kind];
            let mut levels = Vec::new();
            for (suffix, distance) in [("mid", kind.mid_lod), ("far", kind.lod)] {
                let path = root.join("models").join("nature").join(format!("{}_{suffix}.glb", spec.id));
                if let Some(distance) = distance
                    && path.is_file()
                {
                    let bytes =
                        std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
                    let level = asc_render::models::load_glb(&bytes)
                        .with_context(|| format!("loading {}", path.display()))?;
                    levels.push((renderer.add_model(&level), distance));
                }
            }
            if !levels.is_empty() {
                renderer.set_lods(id, levels);
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

/// A loaded ore node model: renderer id and model-space bounds.
pub struct OreModel {
    pub id: ModelId,
    pub min: glam::Vec3,
    pub max: glam::Vec3,
}

/// Every ore's two node models (`models/ores/<id>_surface.glb` and `<id>_deep.glb`), indexed like
/// `asc_world::ores::builtin().ores`.
pub struct OreModels {
    pub surface: Vec<OreModel>,
    pub deep: Vec<OreModel>,
}

impl OreModels {
    pub fn load(renderer: &mut Renderer) -> Result<Self> {
        let dir = root()?.join("models").join("ores");
        let mut load = |name: String| -> Result<OreModel> {
            let path = dir.join(name);
            let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            let data = asc_render::models::load_glb(&bytes)
                .with_context(|| format!("loading {}", path.display()))?;
            Ok(OreModel { id: renderer.add_model(&data), min: data.min, max: data.max })
        };
        let (mut surface, mut deep) = (Vec::new(), Vec::new());
        for ore in &asc_world::ores::builtin().ores {
            surface.push(load(format!("{}_surface.glb", ore.id))?);
            deep.push(load(format!("{}_deep.glb", ore.id))?);
        }
        log::info!("loaded {} ore node models", surface.len() + deep.len());
        Ok(Self { surface, deep })
    }
}

/// The volcanic bombs, smallest first (tools/build_volcanic.py).
pub struct VolcanicModels {
    pub bombs: Vec<asc_render::ModelId>,
}

impl VolcanicModels {
    pub fn load(renderer: &mut Renderer) -> Result<Self> {
        let dir = root()?.join("models").join("volcanic");
        let mut bombs = Vec::new();
        for n in 0..asc_sim::volcanism::BOMB_SIZES {
            let path = dir.join(format!("bomb_{n}.glb"));
            let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            let data = asc_render::models::load_glb(&bytes).with_context(|| format!("loading {}", path.display()))?;
            bombs.push(renderer.add_model(&data));
        }
        Ok(Self { bombs })
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
