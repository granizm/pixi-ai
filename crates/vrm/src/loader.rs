use glam::{Mat4, Quat, Vec3};
use renderer::vertex::Vertex;

use crate::blendshape::BlendShapeGroup;
use crate::bone::HumanoidBones;
use crate::error::VrmError;
use crate::look_at::LookAtApplyer;
use crate::model::{Material, MeshData, MorphTargetData, NodeTransform, SkinJoint, VrmModel};
use crate::spring_bone::SpringBoneGroup;

/// glTFアクセサからバイト列を読む低レベルヘルパー
fn read_accessor_data(blob: &[u8], accessor: &gltf::Accessor) -> Vec<u8> {
    let view = accessor.view().expect("accessor must have buffer view");
    let offset = view.offset() + accessor.offset();
    let stride = view.stride().unwrap_or(accessor.size());
    let count = accessor.count();

    if stride == accessor.size() {
        // Tightly packed - direct copy
        let length = count * accessor.size();
        blob[offset..offset + length].to_vec()
    } else {
        // Interleaved - copy element by element
        let elem_size = accessor.size();
        let mut result = Vec::with_capacity(count * elem_size);
        for i in 0..count {
            let start = offset + i * stride;
            result.extend_from_slice(&blob[start..start + elem_size]);
        }
        result
    }
}

/// Pod型にキャストする型付きヘルパー
fn read_accessor_as<T: bytemuck::Pod>(blob: &[u8], accessor: &gltf::Accessor) -> Vec<T> {
    let bytes = read_accessor_data(blob, accessor);
    bytemuck::cast_slice::<u8, T>(&bytes).to_vec()
}

/// Apply MToon parameters from VRM extension materialProperties to parsed materials.
fn apply_vrm_mtoon_properties(vrm_json: &serde_json::Value, materials: &mut [Material]) {
    if let Some(mat_props) = vrm_json
        .get("materialProperties")
        .and_then(|v| v.as_array())
    {
        for (i, prop) in mat_props.iter().enumerate() {
            if i >= materials.len() {
                break;
            }
            // VRM 0.x materialProperties contain vectorProperties and floatProperties
            if let Some(floats) = prop.get("floatProperties").and_then(|v| v.as_object()) {
                if let Some(v) = floats.get("_ShadeShift").and_then(|v| v.as_f64()) {
                    materials[i].shade_shift = v as f32;
                }
                if let Some(v) = floats.get("_ShadeToony").and_then(|v| v.as_f64()) {
                    materials[i].shade_toony = v as f32;
                }
                if let Some(v) = floats.get("_RimLightingMix").and_then(|v| v.as_f64()) {
                    materials[i].rim_power = v as f32;
                }
                if let Some(v) = floats.get("_RimFresnelPower").and_then(|v| v.as_f64()) {
                    materials[i].rim_power = v as f32;
                }
                if let Some(v) = floats.get("_RimLift").and_then(|v| v.as_f64()) {
                    materials[i].rim_lift = v as f32;
                }
            }
            if let Some(vecs) = prop.get("vectorProperties").and_then(|v| v.as_object()) {
                if let Some(shade) = vecs.get("_ShadeColor").and_then(|v| v.as_array()) {
                    if shade.len() >= 3 {
                        materials[i].shade_color = [
                            shade[0].as_f64().unwrap_or(0.5) as f32,
                            shade[1].as_f64().unwrap_or(0.5) as f32,
                            shade[2].as_f64().unwrap_or(0.5) as f32,
                            shade.get(3).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                        ];
                    }
                }
                if let Some(rim) = vecs.get("_RimColor").and_then(|v| v.as_array()) {
                    if rim.len() >= 3 {
                        materials[i].rim_color = [
                            rim[0].as_f64().unwrap_or(0.0) as f32,
                            rim[1].as_f64().unwrap_or(0.0) as f32,
                            rim[2].as_f64().unwrap_or(0.0) as f32,
                            rim.get(3).and_then(|v| v.as_f64()).unwrap_or(1.0) as f32,
                        ];
                    }
                }
            }
        }
    }
}

/// VRMファイルをロードしてVrmModelを返す
pub fn load(path: &str) -> Result<VrmModel, VrmError> {
    let gltf = gltf::Gltf::open(path)?;
    let blob = gltf
        .blob
        .as_ref()
        .ok_or_else(|| VrmError::MissingData("glTF binary blob not found".into()))?;

    // Parse node transforms
    let node_transforms: Vec<NodeTransform> = gltf
        .document
        .nodes()
        .map(|node| {
            let (t, r, s) = node.transform().decomposed();
            NodeTransform {
                translation: Vec3::from(t),
                rotation: Quat::from_array(r),
                scale: Vec3::from(s),
                children: node.children().map(|c| c.index()).collect(),
            }
        })
        .collect();

    // Load glTF images from the binary blob
    let loaded_images: Vec<Option<image::DynamicImage>> = gltf
        .document
        .images()
        .map(|img| match img.source() {
            gltf::image::Source::View { view, mime_type: _ } => {
                let offset = view.offset();
                let length = view.length();
                let data = &blob[offset..offset + length];
                image::load_from_memory(data)
                    .map_err(|e| {
                        log::warn!("Failed to decode image {}: {}", img.index(), e);
                        e
                    })
                    .ok()
            }
            gltf::image::Source::Uri { .. } => {
                log::warn!("URI-based images not supported in GLB");
                None
            }
        })
        .collect();

    // Parse materials (with MToon extension if present)
    let materials: Vec<Material> = gltf
        .document
        .materials()
        .map(|mat| {
            let pbr = mat.pbr_metallic_roughness();
            let base_color = pbr.base_color_factor();
            let base_color_texture = pbr.base_color_texture().and_then(|info| {
                let tex_index = info.texture().source().index();
                loaded_images.get(tex_index).and_then(|opt| opt.clone())
            });
            Material {
                base_color,
                base_color_texture,
                ..Material::default()
            }
        })
        .collect();

    // Build mesh-to-skin mapping and skin offsets for multi-skin JOINTS_0 correction.
    // In glTF, JOINTS_0 values are relative to each skin's own joint array.
    // When we concatenate all skins into one flat array, we must offset JOINTS_0
    // for meshes belonging to skins other than the first.
    let mut mesh_to_skin: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for node in gltf.document.nodes() {
        if let (Some(mesh), Some(skin)) = (node.mesh(), node.skin()) {
            mesh_to_skin.insert(mesh.index(), skin.index());
        }
    }
    let skin_offsets: Vec<usize> = {
        let mut offsets = Vec::new();
        let mut offset = 0usize;
        for skin in gltf.document.skins() {
            offsets.push(offset);
            offset += skin.joints().count();
        }
        offsets
    };

    // Parse meshes
    let mut meshes = Vec::new();
    for mesh in gltf.document.meshes() {
        let joint_offset = mesh_to_skin
            .get(&mesh.index())
            .and_then(|&skin_idx| skin_offsets.get(skin_idx))
            .copied()
            .unwrap_or(0) as u32;

        for primitive in mesh.primitives() {
            let mut vertices = Vec::new();

            // Read positions
            let positions: Vec<[f32; 3]> = primitive
                .get(&gltf::Semantic::Positions)
                .map(|acc| read_accessor_as(blob, &acc))
                .unwrap_or_default();

            // Read normals
            let normals: Vec<[f32; 3]> = primitive
                .get(&gltf::Semantic::Normals)
                .map(|acc| read_accessor_as(blob, &acc))
                .unwrap_or_default();

            // Read UVs
            let uvs: Vec<[f32; 2]> = primitive
                .get(&gltf::Semantic::TexCoords(0))
                .map(|acc| read_accessor_as(blob, &acc))
                .unwrap_or_default();

            // Read joint indices (JOINTS_0) - typically u8 or u16 in glTF
            let joint_indices: Vec<[u32; 4]> =
                if let Some(acc) = primitive.get(&gltf::Semantic::Joints(0)) {
                    let bytes = read_accessor_data(blob, &acc);
                    match acc.data_type() {
                        gltf::accessor::DataType::U8 => bytes
                            .chunks_exact(4)
                            .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32, c[3] as u32])
                            .collect(),
                        gltf::accessor::DataType::U16 => bytemuck::cast_slice::<u8, u16>(&bytes)
                            .chunks_exact(4)
                            .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32, c[3] as u32])
                            .collect(),
                        _ => vec![[0u32; 4]; positions.len()],
                    }
                } else {
                    vec![[0u32; 4]; positions.len()]
                };

            // Read joint weights (WEIGHTS_0) - typically f32 in glTF
            let joint_weights: Vec<[f32; 4]> = primitive
                .get(&gltf::Semantic::Weights(0))
                .map(|acc| read_accessor_as(blob, &acc))
                .unwrap_or_else(|| vec![[0.0f32; 4]; positions.len()]);

            for (i, &pos) in positions.iter().enumerate() {
                vertices.push(Vertex {
                    position: pos,
                    normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
                    uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                    joint_indices: {
                        let ji = joint_indices.get(i).copied().unwrap_or([0; 4]);
                        [
                            ji[0] + joint_offset,
                            ji[1] + joint_offset,
                            ji[2] + joint_offset,
                            ji[3] + joint_offset,
                        ]
                    },
                    joint_weights: joint_weights.get(i).copied().unwrap_or([0.0; 4]),
                });
            }

            // Read indices
            let indices: Vec<u32> = primitive
                .indices()
                .map(|acc| {
                    let bytes = read_accessor_data(blob, &acc);
                    match acc.data_type() {
                        gltf::accessor::DataType::U16 => bytemuck::cast_slice::<u8, u16>(&bytes)
                            .iter()
                            .map(|&i| i as u32)
                            .collect(),
                        gltf::accessor::DataType::U32 => {
                            bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
                        }
                        _ => vec![],
                    }
                })
                .unwrap_or_default();

            // Parse morph targets
            let morph_targets: Vec<MorphTargetData> = primitive
                .morph_targets()
                .map(|target| {
                    let position_deltas: Vec<[f32; 3]> = target
                        .positions()
                        .map(|acc| read_accessor_as(blob, &acc))
                        .unwrap_or_default();
                    let normal_deltas: Vec<[f32; 3]> = target
                        .normals()
                        .map(|acc| read_accessor_as(blob, &acc))
                        .unwrap_or_default();
                    MorphTargetData {
                        position_deltas,
                        normal_deltas,
                    }
                })
                .collect();

            let material_index = primitive.material().index();

            meshes.push(MeshData {
                vertices,
                indices,
                morph_targets,
                material_index,
                gltf_mesh_index: mesh.index(),
            });
        }
    }

    // Parse skins
    let mut skins = Vec::new();
    for skin in gltf.document.skins() {
        let ibms: Vec<Mat4> = skin
            .inverse_bind_matrices()
            .map(|acc| {
                let data: Vec<[[f32; 4]; 4]> = read_accessor_as(blob, &acc);
                data.into_iter()
                    .map(|m| Mat4::from_cols_array_2d(&m))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        for (i, joint) in skin.joints().enumerate() {
            skins.push(SkinJoint {
                node_index: joint.index(),
                inverse_bind_matrix: ibms.get(i).copied().unwrap_or(Mat4::IDENTITY),
            });
        }
    }

    // Parse VRM extension from raw JSON
    // gltf crate's Document doesn't expose extensions directly,
    // so we read the file as raw JSON to extract the VRM extension.
    let raw_bytes = std::fs::read(path)
        .map_err(|e| VrmError::MissingData(format!("Failed to read file: {e}")))?;

    // For GLB files, the JSON chunk starts after a 12-byte header + 8-byte chunk header
    let vrm_json = if raw_bytes.starts_with(b"glTF") {
        // GLB format: parse JSON chunk
        let json_length =
            u32::from_le_bytes([raw_bytes[12], raw_bytes[13], raw_bytes[14], raw_bytes[15]])
                as usize;
        let json_bytes = &raw_bytes[20..20 + json_length];
        let root: serde_json::Value = serde_json::from_slice(json_bytes)?;
        root.get("extensions")
            .and_then(|e| e.get("VRM"))
            .cloned()
            .ok_or_else(|| VrmError::MissingExtension("VRM".into()))?
    } else {
        // Plain glTF JSON
        let root: serde_json::Value = serde_json::from_slice(&raw_bytes)?;
        root.get("extensions")
            .and_then(|e| e.get("VRM"))
            .cloned()
            .ok_or_else(|| VrmError::MissingExtension("VRM".into()))?
    };

    // Apply MToon material properties from VRM extension
    let mut materials = materials;
    apply_vrm_mtoon_properties(&vrm_json, &mut materials);

    let humanoid_bones = HumanoidBones::from_vrm_json(&vrm_json, &node_transforms)?;
    let blend_shapes = BlendShapeGroup::from_vrm_json(&vrm_json)?;
    let spring_bone_groups = SpringBoneGroup::from_vrm_json(&vrm_json)?;
    let look_at = LookAtApplyer::from_vrm_json(&vrm_json).ok();

    Ok(VrmModel {
        meshes,
        materials,
        skins,
        humanoid_bones,
        blend_shapes,
        node_transforms,
        spring_bone_groups,
        look_at,
    })
}

/// Load a VRM model from in-memory bytes (for Android assets or embedded data).
pub fn load_from_bytes(data: &[u8]) -> Result<VrmModel, VrmError> {
    let gltf = gltf::Gltf::from_slice(data)?;
    let blob = gltf
        .blob
        .as_ref()
        .ok_or_else(|| VrmError::MissingData("glTF binary blob not found".into()))?;

    // Parse node transforms
    let node_transforms: Vec<NodeTransform> = gltf
        .document
        .nodes()
        .map(|node| {
            let (t, r, s) = node.transform().decomposed();
            NodeTransform {
                translation: Vec3::from(t),
                rotation: Quat::from_array(r),
                scale: Vec3::from(s),
                children: node.children().map(|c| c.index()).collect(),
            }
        })
        .collect();

    // Load glTF images from the binary blob
    let loaded_images: Vec<Option<image::DynamicImage>> = gltf
        .document
        .images()
        .map(|img| match img.source() {
            gltf::image::Source::View { view, mime_type: _ } => {
                let offset = view.offset();
                let length = view.length();
                let data = &blob[offset..offset + length];
                image::load_from_memory(data)
                    .map_err(|e| {
                        log::warn!("Failed to decode image {}: {}", img.index(), e);
                        e
                    })
                    .ok()
            }
            gltf::image::Source::Uri { .. } => {
                log::warn!("URI-based images not supported in GLB");
                None
            }
        })
        .collect();

    // Parse materials (with MToon extension if present)
    let materials: Vec<Material> = gltf
        .document
        .materials()
        .map(|mat| {
            let pbr = mat.pbr_metallic_roughness();
            let base_color = pbr.base_color_factor();
            let base_color_texture = pbr.base_color_texture().and_then(|info| {
                let tex_index = info.texture().source().index();
                loaded_images.get(tex_index).and_then(|opt| opt.clone())
            });
            Material {
                base_color,
                base_color_texture,
                ..Material::default()
            }
        })
        .collect();

    // Build mesh-to-skin mapping and skin offsets for multi-skin JOINTS_0 correction.
    let mut mesh_to_skin: std::collections::HashMap<usize, usize> =
        std::collections::HashMap::new();
    for node in gltf.document.nodes() {
        if let (Some(mesh), Some(skin)) = (node.mesh(), node.skin()) {
            mesh_to_skin.insert(mesh.index(), skin.index());
        }
    }
    let skin_offsets: Vec<usize> = {
        let mut offsets = Vec::new();
        let mut offset = 0usize;
        for skin in gltf.document.skins() {
            offsets.push(offset);
            offset += skin.joints().count();
        }
        offsets
    };

    // Parse meshes
    let mut meshes = Vec::new();
    for mesh in gltf.document.meshes() {
        let joint_offset = mesh_to_skin
            .get(&mesh.index())
            .and_then(|&skin_idx| skin_offsets.get(skin_idx))
            .copied()
            .unwrap_or(0) as u32;

        for primitive in mesh.primitives() {
            let mut vertices = Vec::new();

            let positions: Vec<[f32; 3]> = primitive
                .get(&gltf::Semantic::Positions)
                .map(|acc| read_accessor_as(blob, &acc))
                .unwrap_or_default();

            let normals: Vec<[f32; 3]> = primitive
                .get(&gltf::Semantic::Normals)
                .map(|acc| read_accessor_as(blob, &acc))
                .unwrap_or_default();

            let uvs: Vec<[f32; 2]> = primitive
                .get(&gltf::Semantic::TexCoords(0))
                .map(|acc| read_accessor_as(blob, &acc))
                .unwrap_or_default();

            let joint_indices: Vec<[u32; 4]> =
                if let Some(acc) = primitive.get(&gltf::Semantic::Joints(0)) {
                    let bytes = read_accessor_data(blob, &acc);
                    match acc.data_type() {
                        gltf::accessor::DataType::U8 => bytes
                            .chunks_exact(4)
                            .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32, c[3] as u32])
                            .collect(),
                        gltf::accessor::DataType::U16 => bytemuck::cast_slice::<u8, u16>(&bytes)
                            .chunks_exact(4)
                            .map(|c| [c[0] as u32, c[1] as u32, c[2] as u32, c[3] as u32])
                            .collect(),
                        _ => vec![[0u32; 4]; positions.len()],
                    }
                } else {
                    vec![[0u32; 4]; positions.len()]
                };

            let joint_weights: Vec<[f32; 4]> = primitive
                .get(&gltf::Semantic::Weights(0))
                .map(|acc| read_accessor_as(blob, &acc))
                .unwrap_or_else(|| vec![[0.0f32; 4]; positions.len()]);

            for (i, &pos) in positions.iter().enumerate() {
                vertices.push(Vertex {
                    position: pos,
                    normal: normals.get(i).copied().unwrap_or([0.0, 1.0, 0.0]),
                    uv: uvs.get(i).copied().unwrap_or([0.0, 0.0]),
                    joint_indices: {
                        let ji = joint_indices.get(i).copied().unwrap_or([0; 4]);
                        [
                            ji[0] + joint_offset,
                            ji[1] + joint_offset,
                            ji[2] + joint_offset,
                            ji[3] + joint_offset,
                        ]
                    },
                    joint_weights: joint_weights.get(i).copied().unwrap_or([0.0; 4]),
                });
            }

            let indices: Vec<u32> = primitive
                .indices()
                .map(|acc| {
                    let bytes = read_accessor_data(blob, &acc);
                    match acc.data_type() {
                        gltf::accessor::DataType::U16 => bytemuck::cast_slice::<u8, u16>(&bytes)
                            .iter()
                            .map(|&i| i as u32)
                            .collect(),
                        gltf::accessor::DataType::U32 => {
                            bytemuck::cast_slice::<u8, u32>(&bytes).to_vec()
                        }
                        _ => vec![],
                    }
                })
                .unwrap_or_default();

            let morph_targets: Vec<MorphTargetData> = primitive
                .morph_targets()
                .map(|target| {
                    let position_deltas: Vec<[f32; 3]> = target
                        .positions()
                        .map(|acc| read_accessor_as(blob, &acc))
                        .unwrap_or_default();
                    let normal_deltas: Vec<[f32; 3]> = target
                        .normals()
                        .map(|acc| read_accessor_as(blob, &acc))
                        .unwrap_or_default();
                    MorphTargetData {
                        position_deltas,
                        normal_deltas,
                    }
                })
                .collect();

            let material_index = primitive.material().index();

            meshes.push(MeshData {
                vertices,
                indices,
                morph_targets,
                material_index,
                gltf_mesh_index: mesh.index(),
            });
        }
    }

    // Parse skins
    let mut skins = Vec::new();
    for skin in gltf.document.skins() {
        let ibms: Vec<Mat4> = skin
            .inverse_bind_matrices()
            .map(|acc| {
                let data: Vec<[[f32; 4]; 4]> = read_accessor_as(blob, &acc);
                data.into_iter()
                    .map(|m| Mat4::from_cols_array_2d(&m))
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        for (i, joint) in skin.joints().enumerate() {
            skins.push(SkinJoint {
                node_index: joint.index(),
                inverse_bind_matrix: ibms.get(i).copied().unwrap_or(Mat4::IDENTITY),
            });
        }
    }

    // Parse VRM extension from raw bytes
    // For GLB files, the JSON chunk starts after a 12-byte header + 8-byte chunk header
    let vrm_json = if data.starts_with(b"glTF") {
        // GLB format: parse JSON chunk
        let json_length =
            u32::from_le_bytes([data[12], data[13], data[14], data[15]]) as usize;
        let json_bytes = &data[20..20 + json_length];
        let root: serde_json::Value = serde_json::from_slice(json_bytes)?;
        root.get("extensions")
            .and_then(|e| e.get("VRM"))
            .cloned()
            .ok_or_else(|| VrmError::MissingExtension("VRM".into()))?
    } else {
        // Plain glTF JSON
        let root: serde_json::Value = serde_json::from_slice(data)?;
        root.get("extensions")
            .and_then(|e| e.get("VRM"))
            .cloned()
            .ok_or_else(|| VrmError::MissingExtension("VRM".into()))?
    };

    // Apply MToon material properties from VRM extension
    let mut materials = materials;
    apply_vrm_mtoon_properties(&vrm_json, &mut materials);

    let humanoid_bones = HumanoidBones::from_vrm_json(&vrm_json, &node_transforms)?;
    let blend_shapes = BlendShapeGroup::from_vrm_json(&vrm_json)?;
    let spring_bone_groups = SpringBoneGroup::from_vrm_json(&vrm_json)?;
    let look_at = LookAtApplyer::from_vrm_json(&vrm_json).ok();

    Ok(VrmModel {
        meshes,
        materials,
        skins,
        humanoid_bones,
        blend_shapes,
        node_transforms,
        spring_bone_groups,
        look_at,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn load_nonexistent_file_returns_error() {
        let result = load("/nonexistent/path/model.vrm");
        assert!(result.is_err());
    }
}
