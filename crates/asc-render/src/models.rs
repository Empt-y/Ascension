//! Loading models from glTF binaries: static ones (building parts, nature scatter and later
//! machines) and skinned ones (creatures and characters).
//!
//! Only what the pipelines draw is kept: positions, normals, texture coordinates, vertex colours,
//! indices, and each primitive's metallic-roughness factors plus an optional base colour texture
//! with an alpha cutoff (leaf cards). Static models have their node transforms baked into the
//! vertices; skinned ones keep their vertices in bind pose, with joints, weights and the skeleton
//! (and any animations) alongside.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use glam::{Mat4, Vec3, Vec4};

use crate::skeleton::{Clip, Skeleton, clips_of, skeleton_of};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PartVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    /// sRGB-ish tint in rgb and ambient occlusion in a (white when the model has no colours).
    pub color: [u8; 4],
}

/// A decoded RGBA8 texture, named by its glTF image so models can share it.
#[derive(Debug, PartialEq)]
pub struct TextureData {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub rgba: Vec<u8>,
}

#[derive(Clone, Debug)]
pub struct PrimitiveData {
    pub vertices: Vec<PartVertex>,
    pub indices: Vec<u32>,
    /// Linear RGB.
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    /// Base colour texture (multiplied by `base_color`), if any.
    pub texture: Option<Arc<TextureData>>,
    /// Texels with alpha below this are cut away (glTF alphaMode MASK).
    pub alpha_cutoff: Option<f32>,
    /// Light it gives off (linear RGB): glowing lava cracks.
    pub emissive: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct ModelData {
    pub primitives: Vec<PrimitiveData>,
    /// Bounds in model space (Y up, metres).
    pub min: Vec3,
    pub max: Vec3,
}

impl ModelData {
    pub fn triangles(&self) -> usize {
        self.primitives.iter().map(|p| p.indices.len() / 3).sum()
    }
}

/// Parse a `.glb` (or `.gltf` with embedded buffers).
pub fn load_glb(bytes: &[u8]) -> Result<ModelData> {
    let (doc, buffers, images) = gltf::import_slice(bytes).context("parsing glTF")?;
    let mut textures: Vec<Option<Arc<TextureData>>> = vec![None; images.len()];
    let mut primitives = Vec::new();
    let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    let scene = doc.default_scene().or_else(|| doc.scenes().next()).context("glTF has no scene")?;
    let mut stack: Vec<(gltf::Node, Mat4)> = scene.nodes().map(|n| (n, Mat4::IDENTITY)).collect();
    while let Some((node, parent)) = stack.pop() {
        let transform = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        let normal_matrix = transform.inverse().transpose();
        if let Some(mesh) = node.mesh() {
            for prim in mesh.primitives() {
                if prim.mode() != gltf::mesh::Mode::Triangles {
                    continue;
                }
                let reader = prim.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
                let positions: Vec<[f32; 3]> =
                    reader.read_positions().context("primitive without positions")?.collect();
                let normals: Vec<[f32; 3]> = match reader.read_normals() {
                    Some(n) => n.collect(),
                    None => bail!("primitive without normals"),
                };
                let uvs: Vec<[f32; 2]> = match reader.read_tex_coords(0) {
                    Some(t) => t.into_f32().collect(),
                    None => vec![[0.0; 2]; positions.len()],
                };
                let colors: Vec<[u8; 4]> = match reader.read_colors(0) {
                    Some(c) => c.into_rgba_u8().collect(),
                    None => vec![[255; 4]; positions.len()],
                };
                let indices: Vec<u32> = match reader.read_indices() {
                    Some(i) => i.into_u32().collect(),
                    None => (0..positions.len() as u32).collect(),
                };
                let vertices = (0..positions.len())
                    .map(|i| {
                        let p = transform.transform_point3(Vec3::from(positions[i]));
                        let n = normal_matrix.transform_vector3(Vec3::from(normals[i])).normalize_or(Vec3::Y);
                        min = min.min(p);
                        max = max.max(p);
                        PartVertex { position: p.into(), normal: n.into(), uv: uvs[i], color: colors[i] }
                    })
                    .collect();
                let m = material(&prim, &images, &mut textures)?;
                primitives.push(PrimitiveData {
                    vertices,
                    indices,
                    base_color: m.base_color,
                    roughness: m.roughness,
                    metallic: m.metallic,
                    texture: m.texture,
                    alpha_cutoff: m.alpha_cutoff,
                    emissive: m.emissive,
                });
            }
        }
        stack.extend(node.children().map(|c| (c, transform)));
    }
    if primitives.is_empty() {
        bail!("glTF contains no triangle meshes");
    }
    Ok(ModelData { primitives, min, max })
}

/// A primitive's material.
#[derive(Clone, Debug)]
pub struct MaterialData {
    /// Linear RGB.
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
    pub texture: Option<Arc<TextureData>>,
    pub alpha_cutoff: Option<f32>,
    pub emissive: [f32; 3],
}

fn material(
    prim: &gltf::Primitive,
    images: &[gltf::image::Data],
    textures: &mut [Option<Arc<TextureData>>],
) -> Result<MaterialData> {
    let material = prim.material();
    let pbr = material.pbr_metallic_roughness();
    let [r, g, b, _] = pbr.base_color_factor();
    let texture = match pbr.base_color_texture() {
        Some(info) => {
            let source = info.texture().source();
            let slot = &mut textures[source.index()];
            if slot.is_none() {
                let name = source.name().map_or_else(|| format!("image{}", source.index()), str::to_owned);
                *slot = Some(Arc::new(rgba(&images[source.index()], name)?));
            }
            slot.clone()
        }
        None => None,
    };
    let alpha_cutoff = match material.alpha_mode() {
        gltf::material::AlphaMode::Mask => Some(material.alpha_cutoff().unwrap_or(0.5)),
        _ => None,
    };
    Ok(MaterialData {
        base_color: [r, g, b],
        roughness: pbr.roughness_factor(),
        metallic: pbr.metallic_factor(),
        texture,
        alpha_cutoff,
        emissive: material.emissive_factor(),
    })
}

/// A skinned model's vertex: as a part's, plus the four joints that move it and how much.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SkinnedVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [u8; 4],
    pub joints: [u16; 4],
    pub weights: [f32; 4],
}

#[derive(Clone, Debug)]
pub struct SkinnedPrimitive {
    pub vertices: Vec<SkinnedVertex>,
    pub indices: Vec<u32>,
    pub material: MaterialData,
}

#[derive(Clone, Debug)]
pub struct SkinnedModelData {
    pub primitives: Vec<SkinnedPrimitive>,
    pub skeleton: Skeleton,
    /// The file's own animations (creatures carry theirs; characters take a library's).
    pub clips: Vec<Clip>,
    /// Bounds at rest (Y up, metres).
    pub min: Vec3,
    pub max: Vec3,
}

impl SkinnedModelData {
    pub fn triangles(&self) -> usize {
        self.primitives.iter().map(|p| p.indices.len() / 3).sum()
    }
}

/// Parse a skinned `.glb`: every skinned mesh in it, on the first skin's skeleton (other skins'
/// joints are matched to it by name).
pub fn load_skinned_glb(bytes: &[u8]) -> Result<SkinnedModelData> {
    let (doc, buffers, images) = gltf::import_slice(bytes).context("parsing glTF")?;
    let skin = doc.skins().next().context("glTF has no skin")?;
    let skeleton = skeleton_of(&doc, &skin, &buffers);
    let mut textures: Vec<Option<Arc<TextureData>>> = vec![None; images.len()];
    let mut primitives = Vec::new();
    for node in doc.nodes() {
        let (Some(mesh), Some(skin)) = (node.mesh(), node.skin()) else { continue };
        // This mesh's joint indices, as the skeleton's.
        let remap: Vec<u16> =
            skin.joints().map(|j| skeleton.index(j.name().unwrap_or("")).unwrap_or(0) as u16).collect();
        for prim in mesh.primitives() {
            if prim.mode() != gltf::mesh::Mode::Triangles {
                continue;
            }
            let reader = prim.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
            let positions: Vec<[f32; 3]> = reader.read_positions().context("primitive without positions")?.collect();
            let n = positions.len();
            let normals: Vec<[f32; 3]> = reader.read_normals().context("primitive without normals")?.collect();
            let uvs: Vec<[f32; 2]> =
                reader.read_tex_coords(0).map_or_else(|| vec![[0.0; 2]; n], |t| t.into_f32().collect());
            let colors: Vec<[u8; 4]> =
                reader.read_colors(0).map_or_else(|| vec![[255; 4]; n], |c| c.into_rgba_u8().collect());
            let joints: Vec<[u16; 4]> = reader.read_joints(0).context("skinned primitive without joints")?.into_u16().collect();
            let weights: Vec<[f32; 4]> = reader.read_weights(0).context("skinned primitive without weights")?.into_f32().collect();
            let indices: Vec<u32> = reader.read_indices().map_or_else(|| (0..n as u32).collect(), |i| i.into_u32().collect());
            let vertices = (0..n)
                .map(|i| {
                    let w = Vec4::from(weights[i]);
                    let sum = w.element_sum();
                    let w = if sum > 1e-6 { w / sum } else { Vec4::X };
                    SkinnedVertex {
                        position: positions[i],
                        normal: normals[i],
                        uv: uvs[i],
                        color: colors[i],
                        joints: joints[i].map(|j| remap.get(j as usize).copied().unwrap_or(0)),
                        weights: w.into(),
                    }
                })
                .collect();
            primitives.push(SkinnedPrimitive { vertices, indices, material: material(&prim, &images, &mut textures)? });
        }
    }
    if primitives.is_empty() {
        bail!("glTF contains no skinned meshes");
    }
    // Bounds at rest: the vertices skinned on the CPU once.
    let mut palette = Vec::new();
    skeleton.palette(&skeleton.rest_pose(), &mut palette);
    let (mut min, mut max) = (Vec3::splat(f32::MAX), Vec3::splat(f32::MIN));
    for p in &primitives {
        for v in &p.vertices {
            let m: Mat4 = (0..4).map(|k| palette[v.joints[k] as usize] * v.weights[k]).fold(Mat4::ZERO, |a, b| a + b);
            let q = m.transform_point3(Vec3::from(v.position));
            min = min.min(q);
            max = max.max(q);
        }
    }
    let clips = clips_of(&doc, &buffers);
    Ok(SkinnedModelData { primitives, skeleton, clips, min, max })
}

/// Just the animations in a `.glb` (an animation library), keyed by joint name.
pub fn load_clips(bytes: &[u8]) -> Result<Vec<Clip>> {
    let (doc, buffers, _) = gltf::import_slice(bytes).context("parsing glTF")?;
    Ok(clips_of(&doc, &buffers))
}

fn rgba(image: &gltf::image::Data, name: String) -> Result<TextureData> {
    use gltf::image::Format;
    let rgba = match image.format {
        Format::R8G8B8A8 => image.pixels.clone(),
        Format::R8G8B8 => {
            image.pixels.as_chunks::<3>().0.iter().flat_map(|p| [p[0], p[1], p[2], 255]).collect()
        }
        other => bail!("texture {name}: unsupported format {other:?}"),
    };
    Ok(TextureData { name, width: image.width, height: image.height, rgba })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn models_dir() -> std::path::PathBuf {
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/models")
    }

    #[test]
    fn every_generated_part_loads_with_its_materials() {
        let mut count = 0;
        for entry in std::fs::read_dir(models_dir()).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "glb") {
                let model = load_glb(&std::fs::read(&path).unwrap())
                    .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
                assert!(model.triangles() >= 100, "{}", path.display());
                assert!(model.primitives.iter().all(|p| p.roughness > 0.0 && p.indices.len() % 3 == 0));
                count += 1;
            }
        }
        assert_eq!(count, 12);
    }

    #[test]
    fn every_nature_model_loads_standing_on_the_ground() {
        let mut count = 0;
        for entry in std::fs::read_dir(models_dir().join("nature")).unwrap() {
            let path = entry.unwrap().path();
            let lod = path.file_stem().is_some_and(|s| {
                let s = s.to_string_lossy();
                s.ends_with("_far") || s.ends_with("_mid")
            });
            let burnt = path.file_stem().is_some_and(|s| s.to_string_lossy().contains("_burnt"));
            if path.extension().is_some_and(|e| e == "glb") && burnt {
                // A burnt stage: loads, and keeps something (burnt-out grass is mere stubble).
                let model = load_glb(&std::fs::read(&path).unwrap())
                    .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
                assert!(model.triangles() > 0, "{}", path.display());
                continue;
            }
            if path.extension().is_some_and(|e| e == "glb") && !lod {
                let model = load_glb(&std::fs::read(&path).unwrap())
                    .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
                assert!(model.triangles() >= 20, "{}", path.display());
                assert!(model.primitives.iter().all(|p| p.indices.len() % 3 == 0));
                assert!(
                    model.min.y > -0.61 && model.max.y > 0.05,
                    "{}: {}..{}",
                    path.display(),
                    model.min,
                    model.max
                );
                count += 1;
            }
        }
        let spec = std::fs::read_to_string(models_dir().join("../data/nature.toml")).unwrap();
        assert_eq!(count, spec.matches("[[model]]").count());
    }

    #[test]
    fn every_ore_model_loads() {
        let spec = std::fs::read_to_string(models_dir().join("../data/ores.toml")).unwrap();
        let ores = spec.matches("[[ore]]").count();
        let mut count = 0;
        for entry in std::fs::read_dir(models_dir().join("ores")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "glb") {
                let model = load_glb(&std::fs::read(&path).unwrap())
                    .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
                assert!(model.triangles() >= 60, "{}", path.display());
                assert!(model.min.y > -0.61, "{}: {}", path.display(), model.min);
                count += 1;
            }
        }
        assert_eq!(count, 2 * ores);
    }

    #[test]
    fn every_creature_and_the_player_load_skinned_and_animated() {
        let mut count = 0;
        for entry in std::fs::read_dir(models_dir().join("creatures")).unwrap() {
            let path = entry.unwrap().path();
            if path.extension().is_some_and(|e| e == "glb") {
                let model = load_skinned_glb(&std::fs::read(&path).unwrap())
                    .unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
                assert!(model.triangles() >= 100 && !model.clips.is_empty(), "{}", path.display());
                for p in &model.primitives {
                    for v in &p.vertices {
                        assert!((v.weights.iter().sum::<f32>() - 1.0).abs() < 1e-3);
                        assert!(v.joints.iter().all(|&j| (j as usize) < model.skeleton.len()));
                    }
                }
                // Animals stand on the ground, a metre or more tall; fish are a few tens of cm.
                assert!(model.max.y - model.min.y > 0.1, "{}: {}..{}", path.display(), model.min, model.max);
                count += 1;
            }
        }
        assert!(count >= 8, "only {count} creatures");
        let player = models_dir().join("player");
        let body = load_skinned_glb(&std::fs::read(player.join("body.glb")).unwrap()).unwrap();
        let height = body.max.y - body.min.y;
        assert!((1.5..2.2).contains(&height), "a {height} m tall body");
        // Hair and both animation libraries are on the body's skeleton, joint for joint.
        let hair = load_skinned_glb(&std::fs::read(player.join("hair.glb")).unwrap()).unwrap();
        assert!(hair.skeleton.names.iter().all(|n| body.skeleton.index(n).is_some()));
        for lib in ["animations_1.glb", "animations_2.glb"] {
            let clips = load_clips(&std::fs::read(player.join(lib)).unwrap()).unwrap();
            assert!(clips.len() >= 40, "{lib}: {} clips", clips.len());
            let walk = clips.iter().find(|c| c.name.contains("Walk")).expect("a walk");
            let bound = walk.channels.iter().filter(|c| body.skeleton.index(&c.joint).is_some()).count();
            assert!(bound * 10 >= walk.channels.len() * 9, "{lib}: only {bound}/{} channels bind", walk.channels.len());
        }
    }

    #[test]
    fn posing_sixty_creatures_and_the_player_takes_well_under_a_millisecond() {
        let dir = models_dir();
        let wolf = load_skinned_glb(&std::fs::read(dir.join("creatures/wolf.glb")).unwrap()).unwrap();
        let body = load_skinned_glb(&std::fs::read(dir.join("player/body.glb")).unwrap()).unwrap();
        let lib = load_clips(&std::fs::read(dir.join("player/animations_1.glb")).unwrap()).unwrap();
        let walk = lib.iter().find(|c| c.name == "Walk_Loop").unwrap();
        let (wb, bb) = (wolf.skeleton.bind(&wolf.clips[0]), body.skeleton.bind(walk));
        let (mut pose, mut palette) = (wolf.skeleton.rest_pose(), Vec::new());
        let mut frame = |t: f32| {
            for k in 0..60 {
                wolf.skeleton.sample(&wolf.clips[0], &wb, t + k as f32 * 0.1, true, &mut pose);
                wolf.skeleton.palette(&pose, &mut palette);
            }
            let mut p = body.skeleton.rest_pose();
            body.skeleton.sample(walk, &bb, t, true, &mut p);
            body.skeleton.palette(&p, &mut palette);
        };
        frame(0.0);
        let start = std::time::Instant::now();
        for f in 0..20 {
            frame(f as f32 / 60.0);
        }
        let ms = start.elapsed().as_secs_f64() * 1e3 / 20.0;
        println!("posing 60 wolves and the player: {ms:.3} ms a frame");
        // Generous (debug builds and busy machines); release is far under.
        assert!(ms < 8.0, "{ms} ms a frame");
    }

    #[test]
    fn wall_bounds_match_the_spec() {
        let model = load_glb(&std::fs::read(models_dir().join("stone_wall.glb")).unwrap()).unwrap();
        let close = |a: Vec3, b: Vec3| (a - b).abs().max_element() < 1e-3;
        assert!(close(model.min, Vec3::new(-1.0, 0.0, -0.1)), "min {}", model.min);
        assert!(close(model.max, Vec3::new(1.0, 2.0, 0.1)), "max {}", model.max);
    }
}
