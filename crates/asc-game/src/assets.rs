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

pub use asc_world::nature::Collide;

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
    /// Each model's burnt stages (scorched, charred, burnt out), where it has them.
    pub burnt: std::collections::HashMap<ModelId, [ModelId; 3]>,
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
        let mut burnt = std::collections::HashMap::new();
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
            // Its burnt stages, if built (tools/blender/nature.py burnt): drawn at every distance
            // (they have no lighter versions of their own).
            let mut stages = Vec::new();
            for stage in 1..=3 {
                let path = root.join("models").join("nature").join(format!("{}_burnt{stage}.glb", spec.id));
                if !path.is_file() {
                    break;
                }
                let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
                let data = asc_render::models::load_glb(&bytes).with_context(|| format!("loading {}", path.display()))?;
                stages.push(renderer.add_model(&data));
            }
            if let Ok(stages) = <[ModelId; 3]>::try_from(stages) {
                burnt.insert(id, stages);
            }
            models.push(NatureModel { spec, id, min: data.min, max: data.max });
        }
        for biome in file.density.keys() {
            anyhow::ensure!(asc_world::Biome::from_name(biome).is_some(), "density: unknown biome {biome}");
        }
        log::info!("loaded {} nature models ({triangles} triangles)", models.len());
        Ok(Self { models, burnt, kinds: file.kinds, density: file.density })
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

/// The handheld tools (tools/build_tools.py), as one-joint skinned models: each is drawn wherever
/// its one matrix (the hand's, or the eyes') puts it.
pub struct ToolModels {
    pub tools: std::collections::HashMap<asc_sim::tools::ToolKind, asc_render::SkinnedId>,
    /// The multitool's mode lights: laser, rifle, flamer.
    pub mode_lights: [asc_render::SkinnedId; 3],
}

impl ToolModels {
    pub fn load(renderer: &mut Renderer) -> Result<Self> {
        let dir = root()?.join("models").join("tools");
        let mut load = |name: &str| -> Result<asc_render::SkinnedId> {
            let path = dir.join(format!("{name}.glb"));
            let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            let data = asc_render::models::load_glb(&bytes).with_context(|| format!("loading {}", path.display()))?;
            Ok(renderer.add_skinned_model(&asc_render::models::rigid(&data)))
        };
        let mut tools = std::collections::HashMap::new();
        for kind in asc_sim::tools::ToolKind::ALL {
            tools.insert(kind, load(kind.name())?);
        }
        let mode_lights = [load("multitool_light_laser")?, load("multitool_light_rifle")?, load("multitool_light_flamer")?];
        Ok(Self { tools, mode_lights })
    }
}

/// Where each tool's business end is, in its model (from tools/blender/tools.py's report).
pub fn tool_tip(kind: asc_sim::tools::ToolKind) -> glam::Vec3 {
    use asc_sim::tools::ToolKind::*;
    match kind {
        Multitool => glam::Vec3::new(0.0, 0.025, 0.425),
        Scanner => glam::Vec3::new(0.0, 0.17, 0.065),
        Manipulator => glam::Vec3::new(0.0, 0.03, 0.29),
        Torch => glam::Vec3::new(0.0, 0.175, 0.0),
        Beacon => glam::Vec3::new(0.0, 0.8, 0.0),
        Injector => glam::Vec3::new(0.0, 0.02, 0.17),
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

/// A skinned model on the GPU with what animates it: its skeleton, and clips bound to it.
pub struct Animated {
    pub id: asc_render::SkinnedId,
    pub skeleton: asc_render::skeleton::Skeleton,
    pub clips: Vec<(asc_render::skeleton::Clip, asc_render::skeleton::Binding)>,
}

impl Animated {
    fn new(renderer: &mut Renderer, data: &asc_render::models::SkinnedModelData, clips: Vec<asc_render::skeleton::Clip>) -> Self {
        let skeleton = data.skeleton.clone();
        let clips = clips
            .into_iter()
            .map(|c| {
                let b = skeleton.bind(&c);
                (c, b)
            })
            .collect();
        Self { id: renderer.add_skinned_model(data), skeleton, clips }
    }

    /// The clip called `name`, if it has one.
    pub fn clip(&self, name: &str) -> Option<usize> {
        self.clips.iter().position(|(c, _)| c.name == name)
    }
}

fn read_skinned(path: &std::path::Path) -> Result<asc_render::models::SkinnedModelData> {
    let bytes = std::fs::read(path).with_context(|| format!("reading {}", path.display()))?;
    asc_render::models::load_skinned_glb(&bytes).with_context(|| format!("loading {}", path.display()))
}

/// Each creature kind's models (tools/build_creatures.py), in its `models` order; one not built
/// yet is `None` and the kind's others stand in for it.
pub struct CreatureModels {
    kinds: Vec<Vec<Option<std::sync::Arc<Animated>>>>,
}

impl CreatureModels {
    pub fn load(renderer: &mut Renderer) -> Result<Self> {
        let dir = root()?.join("models").join("creatures");
        let mut loaded: std::collections::HashMap<String, Option<std::sync::Arc<Animated>>> = Default::default();
        let mut kinds = Vec::new();
        for kind in asc_sim::creatures::kinds() {
            let mut models = Vec::new();
            for name in &kind.models {
                if !loaded.contains_key(name) {
                    let path = dir.join(format!("{name}.glb"));
                    let model = if path.is_file() {
                        let data = read_skinned(&path)?;
                        let clips = data.clips.clone();
                        Some(std::sync::Arc::new(Animated::new(renderer, &data, clips)))
                    } else {
                        log::warn!("no {}: {} herds will do without it", path.display(), kind.id);
                        None
                    };
                    loaded.insert(name.clone(), model);
                }
                models.push(loaded[name].clone());
            }
            kinds.push(models);
        }
        log::info!("loaded {} creature models", loaded.values().filter(|m| m.is_some()).count());
        Ok(Self { kinds })
    }

    /// Kind `kind`'s model `model`, or another of its kind's if that one isn't built.
    pub fn get(&self, kind: u16, model: u8) -> Option<&std::sync::Arc<Animated>> {
        let models = self.kinds.get(kind as usize)?;
        models.get(model as usize).and_then(Option::as_ref).or_else(|| models.iter().flatten().next())
    }
}

/// The player's body (Quaternius's Universal Base Character), the animation libraries bound to it,
/// and what's worn on it (hair, eyebrows): skinned to the same skeleton, drawn with its palette.
pub struct PlayerModel {
    pub body: Animated,
    pub attachments: Vec<asc_render::SkinnedId>,
}

/// The hair's colour (its texture is white, to be tinted): a dark brown, linear RGB.
const HAIR_COLOR: [f32; 3] = [0.09, 0.055, 0.035];

impl PlayerModel {
    pub fn load(renderer: &mut Renderer) -> Result<Self> {
        let dir = root()?.join("models").join("player");
        let body_data = read_skinned(&dir.join("body.glb"))?;
        let mut clips: Vec<asc_render::skeleton::Clip> = Vec::new();
        for lib in ["animations_1.glb", "animations_2.glb"] {
            let path = dir.join(lib);
            let bytes = std::fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
            for c in asc_render::models::load_clips(&bytes).with_context(|| format!("loading {}", path.display()))? {
                if !clips.iter().any(|x| x.name == c.name) {
                    clips.push(c);
                }
            }
        }
        let body = Animated::new(renderer, &body_data, clips);
        let mut attachments = Vec::new();
        for part in ["hair.glb", "eyebrows.glb"] {
            let mut data = read_skinned(&dir.join(part))?;
            // Onto the body's joints (its own skin may list them in another order), and tinted.
            let remap: Vec<u16> = data
                .skeleton
                .names
                .iter()
                .map(|n| body.skeleton.index(n).unwrap_or(0) as u16)
                .collect();
            for p in &mut data.primitives {
                for v in &mut p.vertices {
                    v.joints = v.joints.map(|j| remap[j as usize]);
                }
                p.material.base_color = HAIR_COLOR;
                p.material.roughness = p.material.roughness.max(0.6);
            }
            attachments.push(renderer.add_skinned_model(&data));
        }
        log::info!(
            "player: {} triangles, {} animations",
            body_data.triangles(),
            body.clips.len()
        );
        Ok(Self { body, attachments })
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

/// `assets/data/colliders.toml`: the collider of every rock and ore node model at unit scale, from
/// its bounds, for the server (which places them to keep animals out, but doesn't load models).
/// `ASC_WRITE_COLLIDERS=1 cargo test -p asc-game colliders` rewrites it; the test fails when it's
/// out of date.
#[cfg(test)]
mod colliders {
    use std::fmt::Write;

    use super::*;

    fn bounds(path: &std::path::Path) -> (glam::Vec3, glam::Vec3) {
        let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        let data = asc_render::models::load_glb(&bytes).unwrap();
        (data.min, data.max)
    }

    #[test]
    fn colliders_file_matches_the_models() {
        let root = root().unwrap();
        let nature: NatureFile =
            toml::from_str(&std::fs::read_to_string(root.join("data/nature.toml")).unwrap()).unwrap();
        let mut out = String::from(
            "# Generated from the models' bounds by `ASC_WRITE_COLLIDERS=1 cargo test -p asc-game colliders`:\n\
             # each rock and ore node model's collider sphere at unit scale, [radius, centre height].\n\n[rocks]\n",
        );
        for m in nature.model.iter().filter(|m| nature.kinds[&m.kind].collide == Collide::Rock) {
            let (min, max) = bounds(&root.join("models/nature").join(format!("{}.glb", m.id)));
            let (r, c) = crate::scatter::rock_collider(min, max);
            writeln!(out, "{} = [{r:.3}, {c:.3}]", m.id).unwrap();
        }
        for kind in ["surface", "deep"] {
            writeln!(out, "\n[ore_{kind}]").unwrap();
            for ore in &asc_world::ores::builtin().ores {
                let (min, max) = bounds(&root.join("models/ores").join(format!("{}_{kind}.glb", ore.id)));
                let (r, c) = crate::scatter::rock_collider(min, max);
                writeln!(out, "{} = [{r:.3}, {c:.3}]", ore.id).unwrap();
            }
        }
        let path = root.join("data/colliders.toml");
        if std::env::var("ASC_WRITE_COLLIDERS").is_ok() {
            std::fs::write(&path, &out).unwrap();
        }
        let have = std::fs::read_to_string(&path).unwrap_or_default().replace("\r\n", "\n");
        assert_eq!(have, out, "{} is out of date: run with ASC_WRITE_COLLIDERS=1", path.display());
    }
}
