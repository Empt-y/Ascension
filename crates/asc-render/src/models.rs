//! Loading static models (building parts and later machines) from glTF binaries.
//!
//! Only what the parts pipeline draws is kept: positions, normals, indices and each primitive's
//! metallic-roughness factors. Node transforms are baked into the vertices.

use anyhow::{Context, Result, bail};
use glam::{Mat4, Vec3};

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, bytemuck::Pod, bytemuck::Zeroable)]
pub struct PartVertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
}

#[derive(Clone, Debug)]
pub struct PrimitiveData {
    pub vertices: Vec<PartVertex>,
    pub indices: Vec<u32>,
    /// Linear RGB.
    pub base_color: [f32; 3],
    pub roughness: f32,
    pub metallic: f32,
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
    let (doc, buffers, _images) = gltf::import_slice(bytes).context("parsing glTF")?;
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
                let indices: Vec<u32> = match reader.read_indices() {
                    Some(i) => i.into_u32().collect(),
                    None => (0..positions.len() as u32).collect(),
                };
                let vertices = positions
                    .iter()
                    .zip(&normals)
                    .map(|(p, n)| {
                        let p = transform.transform_point3(Vec3::from(*p));
                        let n = normal_matrix.transform_vector3(Vec3::from(*n)).normalize_or(Vec3::Y);
                        min = min.min(p);
                        max = max.max(p);
                        PartVertex { position: p.into(), normal: n.into() }
                    })
                    .collect();
                let pbr = prim.material().pbr_metallic_roughness();
                let [r, g, b, _] = pbr.base_color_factor();
                primitives.push(PrimitiveData {
                    vertices,
                    indices,
                    base_color: [r, g, b],
                    roughness: pbr.roughness_factor(),
                    metallic: pbr.metallic_factor(),
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
    fn wall_bounds_match_the_spec() {
        let model = load_glb(&std::fs::read(models_dir().join("stone_wall.glb")).unwrap()).unwrap();
        let close = |a: Vec3, b: Vec3| (a - b).abs().max_element() < 1e-3;
        assert!(close(model.min, Vec3::new(-1.0, 0.0, -0.1)), "min {}", model.min);
        assert!(close(model.max, Vec3::new(1.0, 2.0, 0.1)), "max {}", model.max);
    }
}
