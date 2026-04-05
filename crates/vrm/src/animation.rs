use std::collections::HashMap;
use std::io::BufReader;
use std::path::Path;

use glam::Quat;

use crate::bone::HumanoidBoneName;

/// A single animation channel: keyframe times and rotation values for one bone.
///
/// Rotations are stored as **world-space delta rotations**:
/// `delta = mixamo_world_rest_inv * mixamo_world_anim`
///
/// This correctly handles the bone orientation difference between Mixamo and VRM.
pub struct BoneChannel {
    pub bone: HumanoidBoneName,
    /// Keyframe timestamps in seconds.
    pub times: Vec<f32>,
    /// World-space delta quaternion rotations at each keyframe.
    pub rotations: Vec<Quat>,
}

/// A loaded animation clip containing per-bone rotation keyframes.
pub struct AnimationClip {
    pub name: String,
    /// Total duration in seconds.
    pub duration: f32,
    /// Per-bone rotation channels (world-space delta rotations).
    pub channels: Vec<BoneChannel>,
}

impl AnimationClip {
    /// Load an animation clip from a glTF/GLB or FBX file.
    ///
    /// Dispatches to `load_glb()` or `load_fbx()` based on file extension.
    /// Both paths produce world-space delta rotations retargeted for VRM.
    pub fn load(path: impl AsRef<Path>) -> anyhow::Result<Self> {
        let path = path.as_ref();
        match path
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
        {
            Some(ext) if ext == "fbx" => load_fbx(path),
            Some(ext) if ext == "glb" || ext == "gltf" => load_glb(path),
            Some(ext) => anyhow::bail!("Unsupported animation format: .{ext}"),
            None => anyhow::bail!("No file extension for animation: {}", path.display()),
        }
    }

    /// Load an animation clip from in-memory GLB/glTF bytes.
    ///
    /// This is the `include_bytes!` / Android-asset friendly variant of [`Self::load`].
    /// Uses `gltf::import_slice` internally; only GLB format is supported.
    pub fn load_from_bytes(data: &[u8], name: &str) -> anyhow::Result<Self> {
        load_glb_bytes(data, name)
    }
}

/// FBX time units per second (standard FBX TimeMode).
const FBX_TICKS_PER_SECOND: f64 = 46_186_158_000.0;

/// Load animation from a GLB/glTF file.
fn load_glb(path: &Path) -> anyhow::Result<AnimationClip> {
    let (document, buffers, _images) = gltf::import(path)?;

    let node_count = document.nodes().len();
    let mut node_names: Vec<String> = vec![String::new(); node_count];
    let mut node_rest_rot: Vec<Quat> = vec![Quat::IDENTITY; node_count];
    let mut node_parent: Vec<Option<usize>> = vec![None; node_count];

    for node in document.nodes() {
        let idx = node.index();
        node_names[idx] = node.name().unwrap_or("").to_string();
        let (_, rot, _) = node.transform().decomposed();
        node_rest_rot[idx] = Quat::from_xyzw(rot[0], rot[1], rot[2], rot[3]);
        for child in node.children() {
            node_parent[child.index()] = Some(idx);
        }
    }

    let anim = document
        .animations()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No animations found in {}", path.display()))?;

    let mut node_channels: HashMap<usize, (Vec<f32>, Vec<Quat>)> = HashMap::new();
    let mut duration: f32 = 0.0;

    for channel in anim.channels() {
        if channel.target().property() != gltf::animation::Property::Rotation {
            continue;
        }
        let node_idx = channel.target().node().index();
        let reader = channel.reader(|buf| buffers.get(buf.index()).map(|d| d.0.as_slice()));

        let times: Vec<f32> = match reader.read_inputs() {
            Some(iter) => iter.collect(),
            None => continue,
        };
        let rotations: Vec<Quat> = match reader.read_outputs() {
            Some(gltf::animation::util::ReadOutputs::Rotations(rots)) => rots
                .into_f32()
                .map(|[x, y, z, w]| Quat::from_xyzw(x, y, z, w))
                .collect(),
            _ => continue,
        };

        if let Some(&last) = times.last() {
            duration = duration.max(last);
        }
        node_channels.insert(node_idx, (times, rotations));
    }

    retarget_to_vrm(
        &node_names,
        &node_rest_rot,
        &node_parent,
        &node_channels,
        duration,
        anim.name().unwrap_or("idle"),
        true, // GLB from Blender needs 180° Y conjugation
    )
}

/// Load animation from in-memory GLB/glTF bytes (for `include_bytes!` / Android assets).
fn load_glb_bytes(data: &[u8], name: &str) -> anyhow::Result<AnimationClip> {
    let (document, buffers, _images) = gltf::import_slice(data)?;

    let node_count = document.nodes().len();
    let mut node_names: Vec<String> = vec![String::new(); node_count];
    let mut node_rest_rot: Vec<Quat> = vec![Quat::IDENTITY; node_count];
    let mut node_parent: Vec<Option<usize>> = vec![None; node_count];

    for node in document.nodes() {
        let idx = node.index();
        node_names[idx] = node.name().unwrap_or("").to_string();
        let (_, rot, _) = node.transform().decomposed();
        node_rest_rot[idx] = Quat::from_xyzw(rot[0], rot[1], rot[2], rot[3]);
        for child in node.children() {
            node_parent[child.index()] = Some(idx);
        }
    }

    let anim = document
        .animations()
        .next()
        .ok_or_else(|| anyhow::anyhow!("No animations found in bytes (name='{}')", name))?;

    let mut node_channels: HashMap<usize, (Vec<f32>, Vec<Quat>)> = HashMap::new();
    let mut duration: f32 = 0.0;

    for channel in anim.channels() {
        if channel.target().property() != gltf::animation::Property::Rotation {
            continue;
        }
        let node_idx = channel.target().node().index();
        let reader = channel.reader(|buf| buffers.get(buf.index()).map(|d| d.0.as_slice()));

        let times: Vec<f32> = match reader.read_inputs() {
            Some(iter) => iter.collect(),
            None => continue,
        };
        let rotations: Vec<Quat> = match reader.read_outputs() {
            Some(gltf::animation::util::ReadOutputs::Rotations(rots)) => rots
                .into_f32()
                .map(|[x, y, z, w]| Quat::from_xyzw(x, y, z, w))
                .collect(),
            _ => continue,
        };

        if let Some(&last) = times.last() {
            duration = duration.max(last);
        }
        node_channels.insert(node_idx, (times, rotations));
    }

    retarget_to_vrm(
        &node_names,
        &node_rest_rot,
        &node_parent,
        &node_channels,
        duration,
        anim.name().unwrap_or(name),
        true, // GLB from Blender needs 180deg Y conjugation
    )
}

/// Load animation from an FBX binary file using fbxcel tree parser.
///
/// Parses Model nodes (bones), AnimationCurve nodes (keyframes), and
/// Connections to reconstruct the skeleton hierarchy and animation data.
/// Euler angles from FBX are converted to quaternions.
fn load_fbx(path: &Path) -> anyhow::Result<AnimationClip> {
    use fbxcel::low::v7400::AttributeValue;
    use fbxcel::tree::any::AnyTree;

    let file = std::fs::File::open(path)?;
    let reader = BufReader::new(file);

    let (tree, _fbx_version) = match AnyTree::from_seekable_reader(reader)? {
        AnyTree::V7400(ver, tree, _footer) => (tree, ver),
        _ => anyhow::bail!("Unsupported FBX version (expected 7.x)"),
    };
    let root = tree.root();

    // --- 1. Parse Objects section ---
    let objects = root
        .first_child_by_name("Objects")
        .ok_or_else(|| anyhow::anyhow!("No Objects section in FBX"))?;

    // Model nodes: id → (name, pre_rotation_euler, lcl_rotation_euler)
    let mut model_nodes: HashMap<i64, (String, [f64; 3], [f64; 3])> = HashMap::new();
    // AnimationCurve nodes: id → (times_sec, values)
    let mut anim_curves: HashMap<i64, (Vec<f32>, Vec<f32>)> = HashMap::new();
    // AnimationCurveNode: id → property_name (e.g. "Lcl Rotation")
    let mut curve_nodes: HashMap<i64, String> = HashMap::new();

    for child in objects.children() {
        let attrs = child.attributes();
        match child.name() {
            "Model" => {
                let id = match attrs.first() {
                    Some(AttributeValue::I64(id)) => *id,
                    _ => continue,
                };
                // Name is typically "bone_name\x00\x01Model" — extract before null
                let raw_name = match attrs.get(1) {
                    Some(AttributeValue::String(s)) => s.as_str(),
                    _ => "",
                };
                let name = raw_name.split('\0').next().unwrap_or(raw_name).to_string();

                // Extract PreRotation and Lcl Rotation from Properties70.
                // FBX local transform = PreRotation * LclRotation * PostRotation
                // Mixamo uses PreRotation heavily for bone orientation setup.
                let mut pre_euler = [0.0f64; 3];
                let mut lcl_euler = [0.0f64; 3];
                if let Some(props) = child.first_child_by_name("Properties70") {
                    for p in props.children_by_name("P") {
                        let p_attrs = p.attributes();
                        if let Some(AttributeValue::String(prop_name)) = p_attrs.first() {
                            if p_attrs.len() >= 7 {
                                match prop_name.as_str() {
                                    "PreRotation" => {
                                        pre_euler[0] = attr_to_f64(&p_attrs[4]);
                                        pre_euler[1] = attr_to_f64(&p_attrs[5]);
                                        pre_euler[2] = attr_to_f64(&p_attrs[6]);
                                    }
                                    "Lcl Rotation" => {
                                        lcl_euler[0] = attr_to_f64(&p_attrs[4]);
                                        lcl_euler[1] = attr_to_f64(&p_attrs[5]);
                                        lcl_euler[2] = attr_to_f64(&p_attrs[6]);
                                    }
                                    _ => {}
                                }
                            }
                        }
                    }
                }
                model_nodes.insert(id, (name, pre_euler, lcl_euler));
            }
            "AnimationCurve" => {
                let id = match attrs.first() {
                    Some(AttributeValue::I64(id)) => *id,
                    _ => continue,
                };
                // KeyTime and KeyValueFloat are child nodes
                let mut times: Vec<f32> = Vec::new();
                let mut values: Vec<f32> = Vec::new();

                if let Some(kt) = child.first_child_by_name("KeyTime") {
                    if let Some(arr) = kt.attributes().first() {
                        if let Some(i64_arr) = arr.get_arr_i64() {
                            times = i64_arr
                                .iter()
                                .map(|t| (*t as f64 / FBX_TICKS_PER_SECOND) as f32)
                                .collect();
                        }
                    }
                }
                if let Some(kv) = child.first_child_by_name("KeyValueFloat") {
                    if let Some(arr) = kv.attributes().first() {
                        if let Some(f64_arr) = arr.get_arr_f64() {
                            values = f64_arr.iter().map(|v| *v as f32).collect();
                        } else if let Some(f32_arr) = arr.get_arr_f32() {
                            values = f32_arr.to_vec();
                        }
                    }
                }

                if !times.is_empty() && times.len() == values.len() {
                    anim_curves.insert(id, (times, values));
                }
            }
            "AnimationCurveNode" => {
                let id = match attrs.first() {
                    Some(AttributeValue::I64(id)) => *id,
                    _ => continue,
                };
                let raw_name = match attrs.get(1) {
                    Some(AttributeValue::String(s)) => s.split('\0').next().unwrap_or(s),
                    _ => "",
                };
                curve_nodes.insert(id, raw_name.to_string());
            }
            _ => {}
        }
    }

    // --- 2. Parse Connections section ---
    let connections = root
        .first_child_by_name("Connections")
        .ok_or_else(|| anyhow::anyhow!("No Connections section in FBX"))?;

    // curve_id → curve_node_id (with axis label d|X, d|Y, d|Z)
    let mut curve_to_curve_node: HashMap<i64, (i64, String)> = HashMap::new();
    // curve_node_id → model_id (with property label "Lcl Rotation" etc.)
    let mut curve_node_to_model: HashMap<i64, (i64, String)> = HashMap::new();
    // model_id → parent_model_id (bone hierarchy)
    let mut model_parent: HashMap<i64, i64> = HashMap::new();

    for c in connections.children_by_name("C") {
        let attrs = c.attributes();
        if attrs.len() < 3 {
            continue;
        }
        let conn_type = match &attrs[0] {
            AttributeValue::String(s) => s.as_str(),
            _ => continue,
        };
        let child_id = match &attrs[1] {
            AttributeValue::I64(id) => *id,
            _ => continue,
        };
        let parent_id = match &attrs[2] {
            AttributeValue::I64(id) => *id,
            _ => continue,
        };
        let label = attrs
            .get(3)
            .and_then(|a| a.get_string())
            .unwrap_or("")
            .to_string();

        match conn_type {
            "OP" => {
                // Object-Property connection
                if curve_nodes.contains_key(&child_id) && model_nodes.contains_key(&parent_id) {
                    curve_node_to_model.insert(child_id, (parent_id, label));
                } else if anim_curves.contains_key(&child_id)
                    && curve_nodes.contains_key(&parent_id)
                {
                    curve_to_curve_node.insert(child_id, (parent_id, label));
                }
            }
            "OO" => {
                // Object-Object connection
                if anim_curves.contains_key(&child_id) && curve_nodes.contains_key(&parent_id) {
                    curve_to_curve_node.insert(child_id, (parent_id, label));
                } else if model_nodes.contains_key(&child_id)
                    && model_nodes.contains_key(&parent_id)
                {
                    model_parent.insert(child_id, parent_id);
                }
            }
            _ => {}
        }
    }

    // --- 3. Build skeleton node arrays ---
    // Map model IDs to sequential indices
    let model_ids: Vec<i64> = model_nodes.keys().copied().collect();
    let id_to_idx: HashMap<i64, usize> = model_ids
        .iter()
        .enumerate()
        .map(|(i, id)| (*id, i))
        .collect();

    let node_count = model_ids.len();
    let mut node_names: Vec<String> = vec![String::new(); node_count];
    let mut node_rest_rot: Vec<Quat> = vec![Quat::IDENTITY; node_count];
    let mut node_parent: Vec<Option<usize>> = vec![None; node_count];

    // Store PreRotation per node index for composing with animation frames
    let mut node_pre_rot: Vec<Quat> = vec![Quat::IDENTITY; node_count];

    for (i, model_id) in model_ids.iter().enumerate() {
        let (name, pre_euler, lcl_euler) = &model_nodes[model_id];
        node_names[i] = name.clone();
        // FBX local rotation = PreRotation * LclRotation
        let pre_rot = euler_xyz_to_quat(pre_euler[0], pre_euler[1], pre_euler[2]);
        let lcl_rot = euler_xyz_to_quat(lcl_euler[0], lcl_euler[1], lcl_euler[2]);
        node_pre_rot[i] = pre_rot;
        node_rest_rot[i] = (pre_rot * lcl_rot).normalize();
        if let Some(&parent_id) = model_parent.get(model_id) {
            node_parent[i] = id_to_idx.get(&parent_id).copied();
        }
    }

    // --- 4. Build per-node animation channels ---
    // For each model bone with "Lcl Rotation" curve node, gather X/Y/Z curves
    let mut node_channels: HashMap<usize, (Vec<f32>, Vec<Quat>)> = HashMap::new();
    let mut duration: f32 = 0.0;

    // Group: curve_node_id → {axis → (times, values)}
    #[allow(clippy::type_complexity)]
    let mut cn_curves: HashMap<i64, HashMap<String, (Vec<f32>, Vec<f32>)>> = HashMap::new();
    for (curve_id, (cn_id, axis_label)) in &curve_to_curve_node {
        if let Some(curve_data) = anim_curves.get(curve_id) {
            let axis = if axis_label.contains('X') || axis_label.contains('x') {
                "X"
            } else if axis_label.contains('Y') || axis_label.contains('y') {
                "Y"
            } else if axis_label.contains('Z') || axis_label.contains('z') {
                "Z"
            } else {
                continue;
            };
            cn_curves
                .entry(*cn_id)
                .or_default()
                .insert(axis.to_string(), curve_data.clone());
        }
    }

    // For each curve node linked to a model via "Lcl Rotation"
    for (cn_id, (model_id, prop_label)) in &curve_node_to_model {
        if !prop_label.contains("Rotation") {
            continue;
        }
        let node_idx = match id_to_idx.get(model_id) {
            Some(&idx) => idx,
            None => continue,
        };
        let axes = match cn_curves.get(cn_id) {
            Some(a) => a,
            None => continue,
        };

        let x_curve = axes.get("X");
        let y_curve = axes.get("Y");
        let z_curve = axes.get("Z");

        // Determine number of keyframes (take max across axes)
        let num_keys = [x_curve, y_curve, z_curve]
            .iter()
            .filter_map(|c| c.map(|(t, _)| t.len()))
            .max()
            .unwrap_or(0);

        if num_keys == 0 {
            continue;
        }

        let mut times: Vec<f32> = Vec::with_capacity(num_keys);
        let mut rotations: Vec<Quat> = Vec::with_capacity(num_keys);

        // Get rest pose Lcl Rotation Euler and PreRotation for this bone
        let rest_lcl_euler = model_nodes
            .get(model_id)
            .map(|(_, _, lcl)| *lcl)
            .unwrap_or([0.0; 3]);
        let pre_rot = node_pre_rot[node_idx];

        for k in 0..num_keys {
            // Get animated Lcl Rotation Euler at this keyframe (fall back to rest Lcl)
            let ex = x_curve
                .and_then(|(_, v)| v.get(k))
                .copied()
                .unwrap_or(rest_lcl_euler[0] as f32);
            let ey = y_curve
                .and_then(|(_, v)| v.get(k))
                .copied()
                .unwrap_or(rest_lcl_euler[1] as f32);
            let ez = z_curve
                .and_then(|(_, v)| v.get(k))
                .copied()
                .unwrap_or(rest_lcl_euler[2] as f32);

            // FBX animated local = PreRotation * animated_LclRotation
            let lcl_anim = euler_xyz_to_quat(ex as f64, ey as f64, ez as f64);
            rotations.push((pre_rot * lcl_anim).normalize());

            // Use X axis time as primary (all axes should have same times)
            let t = x_curve
                .and_then(|(t, _)| t.get(k))
                .or_else(|| y_curve.and_then(|(t, _)| t.get(k)))
                .or_else(|| z_curve.and_then(|(t, _)| t.get(k)))
                .copied()
                .unwrap_or(0.0);
            times.push(t);
            duration = duration.max(t);
        }

        node_channels.insert(node_idx, (times, rotations));
    }

    let anim_name = path
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("fbx_anim");

    log::info!(
        "FBX parsed: {} models, {} anim curves, {} rotation channels",
        model_nodes.len(),
        anim_curves.len(),
        node_channels.len()
    );

    // Mixamo FBX also faces -Z (same as GLB), so 180° Y conjugation is needed
    retarget_to_vrm(
        &node_names,
        &node_rest_rot,
        &node_parent,
        &node_channels,
        duration,
        anim_name,
        true,
    )
}

/// Extract an f64 value from an FBX attribute (handles both F64 and F32).
fn attr_to_f64(attr: &fbxcel::low::v7400::AttributeValue) -> f64 {
    match attr {
        fbxcel::low::v7400::AttributeValue::F64(v) => *v,
        fbxcel::low::v7400::AttributeValue::F32(v) => *v as f64,
        fbxcel::low::v7400::AttributeValue::I64(v) => *v as f64,
        fbxcel::low::v7400::AttributeValue::I32(v) => *v as f64,
        _ => 0.0,
    }
}

/// Convert FBX Euler angles (degrees, XYZ rotation order) to a quaternion.
fn euler_xyz_to_quat(x_deg: f64, y_deg: f64, z_deg: f64) -> Quat {
    let x = (x_deg as f32).to_radians();
    let y = (y_deg as f32).to_radians();
    let z = (z_deg as f32).to_radians();
    // FBX default rotation order: Euler XYZ
    // Applied as: Rz * Ry * Rx (intrinsic XYZ = extrinsic ZYX)
    let qx = Quat::from_rotation_x(x);
    let qy = Quat::from_rotation_y(y);
    let qz = Quat::from_rotation_z(z);
    (qz * qy * qx).normalize()
}

/// Shared retargeting pipeline: converts skeleton + animation data to VRM local deltas.
///
/// `conjugate_y180`: if true, applies 180° Y conjugation to world deltas
/// (needed for GLB from Blender where model faces -Z instead of VRM's +Z).
fn retarget_to_vrm(
    node_names: &[String],
    node_rest_rot: &[Quat],
    node_parent: &[Option<usize>],
    node_channels: &HashMap<usize, (Vec<f32>, Vec<Quat>)>,
    duration: f32,
    anim_name: &str,
    conjugate_y180: bool,
) -> anyhow::Result<AnimationClip> {
    let world_rest = compute_world_rotations(node_rest_rot, node_parent);

    // Identify which node indices map to VRM bones
    let mut bone_node_map: Vec<(HumanoidBoneName, usize)> = Vec::new();
    for (idx, name) in node_names.iter().enumerate() {
        if let Some(bone) = mixamo_to_vrm(name) {
            if node_channels.contains_key(&idx) {
                bone_node_map.push((bone, idx));
            }
        }
    }

    let num_frames = node_channels
        .values()
        .map(|(t, _)| t.len())
        .max()
        .unwrap_or(0);

    if num_frames == 0 {
        anyhow::bail!("No keyframes found in animation");
    }

    let mut channels: Vec<BoneChannel> = bone_node_map
        .iter()
        .map(|(bone, _)| BoneChannel {
            bone: *bone,
            times: Vec::with_capacity(num_frames),
            rotations: Vec::with_capacity(num_frames),
        })
        .collect();

    let vrm_bone_parent = build_vrm_bone_parent_map();
    let bone_to_ch: HashMap<HumanoidBoneName, usize> = bone_node_map
        .iter()
        .enumerate()
        .map(|(i, (bone, _))| (*bone, i))
        .collect();

    for frame_idx in 0..num_frames {
        let mut frame_local: Vec<Quat> = node_rest_rot.to_vec();
        for (&node_idx, (_, rots)) in node_channels {
            if frame_idx < rots.len() {
                frame_local[node_idx] = rots[frame_idx];
            }
        }

        let frame_world = compute_world_rotations(&frame_local, node_parent);

        let world_deltas: Vec<Quat> = bone_node_map
            .iter()
            .map(|(_, node_idx)| {
                let world_rest_inv = world_rest[*node_idx].inverse();
                let world_anim = frame_world[*node_idx];
                let raw = (world_anim * world_rest_inv).normalize();
                if conjugate_y180 {
                    // Conjugate by 180° Y: (x,y,z,w) → (-x,y,-z,w)
                    Quat::from_xyzw(-raw.x, raw.y, -raw.z, raw.w)
                } else {
                    raw
                }
            })
            .collect();

        for (ch_idx, (bone, node_idx)) in bone_node_map.iter().enumerate() {
            let world_delta = world_deltas[ch_idx];

            let local_delta = if let Some(parent_bone) = vrm_bone_parent.get(bone) {
                if let Some(&parent_ch_idx) = bone_to_ch.get(parent_bone) {
                    let parent_world_delta = world_deltas[parent_ch_idx];
                    (parent_world_delta.inverse() * world_delta).normalize()
                } else {
                    world_delta
                }
            } else {
                world_delta
            };

            if let Some((times, _)) = node_channels.get(node_idx) {
                if frame_idx < times.len() {
                    channels[ch_idx].times.push(times[frame_idx]);
                }
            }
            channels[ch_idx].rotations.push(local_delta);
        }
    }

    channels.retain(|ch| !ch.times.is_empty());

    log::info!(
        "Loaded animation '{}': {:.2}s, {} bone channels (retargeted local delta)",
        anim_name,
        duration,
        channels.len()
    );

    Ok(AnimationClip {
        name: anim_name.to_string(),
        duration,
        channels,
    })
}

/// Compute world-space rotations for all nodes via FK.
fn compute_world_rotations(local_rots: &[Quat], parents: &[Option<usize>]) -> Vec<Quat> {
    let n = local_rots.len();
    let mut world = vec![Quat::IDENTITY; n];
    let mut computed = vec![false; n];

    // Topological sort: process parents before children
    // Simple iterative approach: keep processing until all done
    let mut remaining = n;
    while remaining > 0 {
        let mut progress = false;
        for i in 0..n {
            if computed[i] {
                continue;
            }
            match parents[i] {
                None => {
                    // Root node
                    world[i] = local_rots[i];
                    computed[i] = true;
                    remaining -= 1;
                    progress = true;
                }
                Some(parent) if computed[parent] => {
                    world[i] = (world[parent] * local_rots[i]).normalize();
                    computed[i] = true;
                    remaining -= 1;
                    progress = true;
                }
                _ => {}
            }
        }
        if !progress {
            break; // Avoid infinite loop on broken hierarchies
        }
    }

    world
}

/// Build a map of VRM humanoid bone → parent bone.
///
/// This hardcodes the standard VRM humanoid skeleton hierarchy.
fn build_vrm_bone_parent_map() -> HashMap<HumanoidBoneName, HumanoidBoneName> {
    use HumanoidBoneName::*;
    let pairs: &[(HumanoidBoneName, HumanoidBoneName)] = &[
        // Spine chain
        (Spine, Hips),
        (Chest, Spine),
        (UpperChest, Chest),
        (Neck, UpperChest),
        (Head, Neck),
        // Left arm
        (LeftShoulder, UpperChest),
        (LeftUpperArm, LeftShoulder),
        (LeftLowerArm, LeftUpperArm),
        (LeftHand, LeftLowerArm),
        // Right arm
        (RightShoulder, UpperChest),
        (RightUpperArm, RightShoulder),
        (RightLowerArm, RightUpperArm),
        (RightHand, RightLowerArm),
        // Left leg
        (LeftUpperLeg, Hips),
        (LeftLowerLeg, LeftUpperLeg),
        (LeftFoot, LeftLowerLeg),
        (LeftToes, LeftFoot),
        // Right leg
        (RightUpperLeg, Hips),
        (RightLowerLeg, RightUpperLeg),
        (RightFoot, RightLowerLeg),
        (RightToes, RightFoot),
        // Left fingers
        (LeftThumbProximal, LeftHand),
        (LeftThumbIntermediate, LeftThumbProximal),
        (LeftThumbDistal, LeftThumbIntermediate),
        (LeftIndexProximal, LeftHand),
        (LeftIndexIntermediate, LeftIndexProximal),
        (LeftIndexDistal, LeftIndexIntermediate),
        (LeftMiddleProximal, LeftHand),
        (LeftMiddleIntermediate, LeftMiddleProximal),
        (LeftMiddleDistal, LeftMiddleIntermediate),
        (LeftRingProximal, LeftHand),
        (LeftRingIntermediate, LeftRingProximal),
        (LeftRingDistal, LeftRingIntermediate),
        (LeftLittleProximal, LeftHand),
        (LeftLittleIntermediate, LeftLittleProximal),
        (LeftLittleDistal, LeftLittleIntermediate),
        // Right fingers
        (RightThumbProximal, RightHand),
        (RightThumbIntermediate, RightThumbProximal),
        (RightThumbDistal, RightThumbIntermediate),
        (RightIndexProximal, RightHand),
        (RightIndexIntermediate, RightIndexProximal),
        (RightIndexDistal, RightIndexIntermediate),
        (RightMiddleProximal, RightHand),
        (RightMiddleIntermediate, RightMiddleProximal),
        (RightMiddleDistal, RightMiddleIntermediate),
        (RightRingProximal, RightHand),
        (RightRingIntermediate, RightRingProximal),
        (RightRingDistal, RightRingIntermediate),
        (RightLittleProximal, RightHand),
        (RightLittleIntermediate, RightLittleProximal),
        (RightLittleDistal, RightLittleIntermediate),
        // Eyes
        (LeftEye, Head),
        (RightEye, Head),
        (Jaw, Head),
    ];
    pairs.iter().cloned().collect()
}

/// Map a Mixamo bone name (e.g. "mixamorig:Hips") to a VRM HumanoidBoneName.
fn mixamo_to_vrm(name: &str) -> Option<HumanoidBoneName> {
    let bone = name.strip_prefix("mixamorig:").unwrap_or(name);

    match bone {
        "Hips" => Some(HumanoidBoneName::Hips),
        "Spine" => Some(HumanoidBoneName::Spine),
        "Spine1" => Some(HumanoidBoneName::Chest),
        "Spine2" => Some(HumanoidBoneName::UpperChest),
        "Neck" => Some(HumanoidBoneName::Neck),
        "Head" => Some(HumanoidBoneName::Head),
        "LeftShoulder" => Some(HumanoidBoneName::LeftShoulder),
        "LeftArm" => Some(HumanoidBoneName::LeftUpperArm),
        "LeftForeArm" => Some(HumanoidBoneName::LeftLowerArm),
        "LeftHand" => Some(HumanoidBoneName::LeftHand),
        "RightShoulder" => Some(HumanoidBoneName::RightShoulder),
        "RightArm" => Some(HumanoidBoneName::RightUpperArm),
        "RightForeArm" => Some(HumanoidBoneName::RightLowerArm),
        "RightHand" => Some(HumanoidBoneName::RightHand),
        "LeftUpLeg" => Some(HumanoidBoneName::LeftUpperLeg),
        "LeftLeg" => Some(HumanoidBoneName::LeftLowerLeg),
        "LeftFoot" => Some(HumanoidBoneName::LeftFoot),
        "LeftToeBase" => Some(HumanoidBoneName::LeftToes),
        "RightUpLeg" => Some(HumanoidBoneName::RightUpperLeg),
        "RightLeg" => Some(HumanoidBoneName::RightLowerLeg),
        "RightFoot" => Some(HumanoidBoneName::RightFoot),
        "RightToeBase" => Some(HumanoidBoneName::RightToes),
        "LeftHandThumb1" => Some(HumanoidBoneName::LeftThumbProximal),
        "LeftHandThumb2" => Some(HumanoidBoneName::LeftThumbIntermediate),
        "LeftHandThumb3" => Some(HumanoidBoneName::LeftThumbDistal),
        "LeftHandIndex1" => Some(HumanoidBoneName::LeftIndexProximal),
        "LeftHandIndex2" => Some(HumanoidBoneName::LeftIndexIntermediate),
        "LeftHandIndex3" => Some(HumanoidBoneName::LeftIndexDistal),
        "LeftHandMiddle1" => Some(HumanoidBoneName::LeftMiddleProximal),
        "LeftHandMiddle2" => Some(HumanoidBoneName::LeftMiddleIntermediate),
        "LeftHandMiddle3" => Some(HumanoidBoneName::LeftMiddleDistal),
        "LeftHandRing1" => Some(HumanoidBoneName::LeftRingProximal),
        "LeftHandRing2" => Some(HumanoidBoneName::LeftRingIntermediate),
        "LeftHandRing3" => Some(HumanoidBoneName::LeftRingDistal),
        "LeftHandPinky1" => Some(HumanoidBoneName::LeftLittleProximal),
        "LeftHandPinky2" => Some(HumanoidBoneName::LeftLittleIntermediate),
        "LeftHandPinky3" => Some(HumanoidBoneName::LeftLittleDistal),
        "RightHandThumb1" => Some(HumanoidBoneName::RightThumbProximal),
        "RightHandThumb2" => Some(HumanoidBoneName::RightThumbIntermediate),
        "RightHandThumb3" => Some(HumanoidBoneName::RightThumbDistal),
        "RightHandIndex1" => Some(HumanoidBoneName::RightIndexProximal),
        "RightHandIndex2" => Some(HumanoidBoneName::RightIndexIntermediate),
        "RightHandIndex3" => Some(HumanoidBoneName::RightIndexDistal),
        "RightHandMiddle1" => Some(HumanoidBoneName::RightMiddleProximal),
        "RightHandMiddle2" => Some(HumanoidBoneName::RightMiddleIntermediate),
        "RightHandMiddle3" => Some(HumanoidBoneName::RightMiddleDistal),
        "RightHandRing1" => Some(HumanoidBoneName::RightRingProximal),
        "RightHandRing2" => Some(HumanoidBoneName::RightRingIntermediate),
        "RightHandRing3" => Some(HumanoidBoneName::RightRingDistal),
        "RightHandPinky1" => Some(HumanoidBoneName::RightLittleProximal),
        "RightHandPinky2" => Some(HumanoidBoneName::RightLittleIntermediate),
        "RightHandPinky3" => Some(HumanoidBoneName::RightLittleDistal),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixamo_mapping_covers_major_bones() {
        assert_eq!(
            mixamo_to_vrm("mixamorig:Hips"),
            Some(HumanoidBoneName::Hips)
        );
        assert_eq!(
            mixamo_to_vrm("mixamorig:Spine1"),
            Some(HumanoidBoneName::Chest)
        );
        assert_eq!(
            mixamo_to_vrm("mixamorig:LeftArm"),
            Some(HumanoidBoneName::LeftUpperArm)
        );
        assert_eq!(
            mixamo_to_vrm("mixamorig:RightForeArm"),
            Some(HumanoidBoneName::RightLowerArm)
        );
        assert_eq!(
            mixamo_to_vrm("mixamorig:LeftUpLeg"),
            Some(HumanoidBoneName::LeftUpperLeg)
        );
        assert_eq!(mixamo_to_vrm("mixamorig:HeadTop_End"), None);
        assert_eq!(mixamo_to_vrm("mixamorig:LeftHandThumb4"), None);
    }

    #[test]
    fn mixamo_mapping_finger_bones() {
        assert_eq!(
            mixamo_to_vrm("mixamorig:LeftHandThumb1"),
            Some(HumanoidBoneName::LeftThumbProximal)
        );
        assert_eq!(
            mixamo_to_vrm("mixamorig:RightHandPinky3"),
            Some(HumanoidBoneName::RightLittleDistal)
        );
    }

    #[test]
    fn compute_world_rotations_identity_chain() {
        let local = vec![Quat::IDENTITY, Quat::IDENTITY, Quat::IDENTITY];
        let parents = vec![None, Some(0), Some(1)];
        let world = compute_world_rotations(&local, &parents);
        for w in &world {
            assert!(w.dot(Quat::IDENTITY).abs() > 0.999);
        }
    }

    #[test]
    fn compute_world_rotations_accumulates() {
        let rot90y = Quat::from_rotation_y(std::f32::consts::FRAC_PI_2);
        let local = vec![rot90y, rot90y];
        let parents = vec![None, Some(0)];
        let world = compute_world_rotations(&local, &parents);
        // Child should have 180° Y rotation
        let expected = Quat::from_rotation_y(std::f32::consts::PI);
        assert!(world[1].dot(expected).abs() > 0.99);
    }

    #[test]
    fn compare_glb_vs_fbx_frame0() {
        use std::path::Path;

        let glb_path = Path::new("../../assets/animations/Running.glb");
        let fbx_path = Path::new("../../assets/animations/Running.fbx");
        if !glb_path.exists() || !fbx_path.exists() {
            eprintln!("Skipping compare test: animation files not found");
            return;
        }

        let glb_clip = load_glb(glb_path).expect("GLB load failed");
        let fbx_clip = load_fbx(fbx_path).expect("FBX load failed");

        // Build bone→frame0 rotation maps
        let glb_map: HashMap<HumanoidBoneName, Quat> = glb_clip
            .channels
            .iter()
            .filter_map(|ch| ch.rotations.first().map(|r| (ch.bone, *r)))
            .collect();
        let fbx_map: HashMap<HumanoidBoneName, Quat> = fbx_clip
            .channels
            .iter()
            .filter_map(|ch| ch.rotations.first().map(|r| (ch.bone, *r)))
            .collect();

        // Compare key bones
        let key_bones = [
            HumanoidBoneName::Hips,
            HumanoidBoneName::Spine,
            HumanoidBoneName::Chest,
            HumanoidBoneName::UpperChest,
            HumanoidBoneName::Neck,
            HumanoidBoneName::Head,
            HumanoidBoneName::LeftShoulder,
            HumanoidBoneName::LeftUpperArm,
            HumanoidBoneName::LeftLowerArm,
            HumanoidBoneName::LeftHand,
            HumanoidBoneName::RightUpperArm,
            HumanoidBoneName::RightLowerArm,
            HumanoidBoneName::LeftUpperLeg,
            HumanoidBoneName::LeftLowerLeg,
            HumanoidBoneName::LeftFoot,
            HumanoidBoneName::RightUpperLeg,
            HumanoidBoneName::RightLowerLeg,
            HumanoidBoneName::RightFoot,
        ];

        eprintln!("\n=== GLB vs FBX frame0 delta comparison ===");
        eprintln!(
            "{:<20} {:>8} {:>40} {:>40}",
            "Bone", "dot", "GLB (x,y,z,w)", "FBX (x,y,z,w)"
        );
        let mut max_diff = 0.0f32;
        for bone in &key_bones {
            let glb_q = glb_map.get(bone);
            let fbx_q = fbx_map.get(bone);
            match (glb_q, fbx_q) {
                (Some(g), Some(f)) => {
                    let dot = g.dot(*f).abs();
                    let diff = 1.0 - dot;
                    if diff > max_diff {
                        max_diff = diff;
                    }
                    let mark = if diff > 0.01 { " <<<" } else { "" };
                    eprintln!(
                        "{:<20} {:>8.5} ({:>8.5},{:>8.5},{:>8.5},{:>8.5}) ({:>8.5},{:>8.5},{:>8.5},{:>8.5}){}",
                        format!("{:?}", bone), dot,
                        g.x, g.y, g.z, g.w,
                        f.x, f.y, f.z, f.w,
                        mark
                    );
                }
                (Some(_), None) => eprintln!("{:<20} GLB only", format!("{:?}", bone)),
                (None, Some(_)) => eprintln!("{:<20} FBX only", format!("{:?}", bone)),
                _ => {}
            }
        }
        eprintln!("Max diff (1-dot): {:.6}", max_diff);
    }
}
