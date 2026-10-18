//! Loading static models (building parts, nature scatter and later machines) from glTF binaries.
//!
//! Only what the parts pipeline draws is kept: positions, normals, texture coordinates, vertex
//! colours, indices, and each primitive's metallic-roughness factors plus an optional base colour
//! texture with an alpha cutoff (leaf cards). Node transforms are baked into the vertices.

use std::sync::Arc;

use anyhow::{Context, Result, bail};
use glam::{Mat4, Vec3};

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
                let material = prim.material();
                let pbr = material.pbr_metallic_roughness();
                let [r, g, b, _] = pbr.base_color_factor();
                let texture = match pbr.base_color_texture() {
                    Some(info) => {
                        let source = info.texture().source();
                        let slot = &mut textures[source.index()];
                        if slot.is_none() {
                            let name = source
                                .name()
                                .map_or_else(|| format!("image{}", source.index()), str::to_owned);
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
                primitives.push(PrimitiveData {
                    vertices,
                    indices,
                    base_color: [r, g, b],
                    roughness: pbr.roughness_factor(),
                    metallic: pbr.metallic_factor(),
                    texture,
                    alpha_cutoff,
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
            if path.extension().is_some_and(|e| e == "glb") {
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
    fn wall_bounds_match_the_spec() {
        let model = load_glb(&std::fs::read(models_dir().join("stone_wall.glb")).unwrap()).unwrap();
        let close = |a: Vec3, b: Vec3| (a - b).abs().max_element() < 1e-3;
        assert!(close(model.min, Vec3::new(-1.0, 0.0, -0.1)), "min {}", model.min);
        assert!(close(model.max, Vec3::new(1.0, 2.0, 0.1)), "max {}", model.max);
    }
}
