use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context, Result};
use renderer::context::RenderContext;
use renderer::debug_overlay::DebugOverlay;
use renderer::scene::{MeshMaterialInput, Scene};
use renderer::vertex::Vertex;
use tracker::holistic::HolisticTracker;

use nokhwa::pixel_format::RgbFormat;
use nokhwa::utils::{CameraFormat, CameraIndex, FrameFormat, RequestedFormat, RequestedFormatType};

use crate::auto_blink::AutoBlink;
use crate::tracker_thread::TrackerThread;
use crate::user_prefs::UserPrefs;
use winit::window::Window;

use crate::rig_config::RigConfig;
use crate::state::{AppState, RigState};

/// Default paths for ONNX models and VRM avatar.
const DEFAULT_VRM_PATH: &str = "assets/models/default_avatar.vrm";
const FACE_MODEL_PATH: &str = "assets/models/face_landmark.onnx";
const POSE_MODEL_PATH: &str = "assets/models/pose_landmark.onnx";
const HAND_MODEL_PATH: &str = "assets/models/hand_landmark.onnx";
const DEFAULT_ANIMATION_PATH: &str = "assets/animations/idle.glb";

/// Returns the base path for assets.
/// In a macOS .app bundle, assets are in Contents/Resources/assets/.
/// Otherwise, falls back to the current directory.
fn assets_base_path() -> PathBuf {
    if let Ok(exe) = std::env::current_exe() {
        // In .app bundle: Contents/MacOS/pixi-ai-bin -> Contents/Resources/assets
        if let Some(macos_dir) = exe.parent() {
            let resources_assets = macos_dir.join("../Resources/assets");
            if resources_assets.exists() {
                return resources_assets
                    .canonicalize()
                    .unwrap_or(resources_assets);
            }
        }
    }
    PathBuf::from("assets")
}

/// Check that all required model files exist and return a helpful error if not.
fn check_model_files() -> Result<()> {
    let base = assets_base_path();
    let required = [
        (base.join("models/default_avatar.vrm"), "VRM avatar"),
        (base.join("models/face_landmark.onnx"), "Face landmark ONNX model"),
        (base.join("models/pose_landmark.onnx"), "Pose landmark ONNX model"),
        (base.join("models/hand_landmark.onnx"), "Hand landmark ONNX model"),
    ];

    let missing: Vec<_> = required
        .iter()
        .filter(|(path, _)| !path.exists())
        .collect();

    if !missing.is_empty() {
        let list = missing
            .iter()
            .map(|(path, desc)| format!("  - {} ({desc})", path.display()))
            .collect::<Vec<_>>()
            .join("\n");

        anyhow::bail!(
            "Required model files not found:\n{list}\n\n\
             To download them, run:\n  \
             sh scripts/setup.sh download-models\n\n\
             Or download manually — see README.md for details."
        );
    }
    Ok(())
}

/// Check if a file path has a video extension.
fn is_video_file(path: &str) -> bool {
    let ext = std::path::Path::new(path)
        .extension()
        .and_then(|e| e.to_str())
        .map(|e| e.to_lowercase())
        .unwrap_or_default();
    matches!(ext.as_str(), "mp4" | "m4v" | "mov")
}

/// Initialize all application resources.
///
/// 1. wgpu rendering context
/// 2. VRM model loading
/// 3. GPU scene creation
/// 4. ML tracker initialization
pub async fn init_all(window: Arc<Window>) -> Result<AppState> {
    // 0. Verify model files exist
    check_model_files()?;

    let assets_base = assets_base_path();

    // 1. wgpu initialization
    let mut render_ctx = RenderContext::new(window).await?;

    // 2. Load VRM model
    let vrm_model = vrm::loader::load(
        assets_base
            .join("models/default_avatar.vrm")
            .to_str()
            .unwrap_or(DEFAULT_VRM_PATH),
    )
    .context("Failed to load VRM avatar. Run: sh scripts/setup.sh download-models")?;

    // Log VRM model stats for debugging skinning pipeline
    {
        let total_verts: usize = vrm_model.meshes.iter().map(|m| m.vertices.len()).sum();
        let skinned_verts: usize = vrm_model
            .meshes
            .iter()
            .flat_map(|m| &m.vertices)
            .filter(|v| v.joint_weights.iter().any(|w| *w > 0.0))
            .count();
        pipeline_logger::bone(log::Level::Info, "VRM model loaded")
            .field("meshes", vrm_model.meshes.len())
            .field("total_verts", total_verts)
            .field("skinned_verts", skinned_verts)
            .field("skin_joints", vrm_model.skins.len())
            .field("materials", vrm_model.materials.len())
            .field("node_transforms", vrm_model.node_transforms.len())
            .emit();
    }

    // 3. Create GPU scene from VRM model meshes
    let vertices_list: Vec<(&[Vertex], &[u32])> = vrm_model
        .meshes
        .iter()
        .map(|m| (m.vertices.as_slice(), m.indices.as_slice()))
        .collect();

    let max_joints = vrm_model.skins.len().max(1);

    // Extract per-mesh morph target position deltas for GPU upload
    let mesh_morph_targets: Vec<Vec<Vec<[f32; 3]>>> = vrm_model
        .meshes
        .iter()
        .map(|m| {
            m.morph_targets
                .iter()
                .map(|t| t.position_deltas.clone())
                .collect()
        })
        .collect();

    {
        let total_targets: usize = mesh_morph_targets.iter().map(|t| t.len()).sum();
        pipeline_logger::bone(log::Level::Info, "morph targets extracted")
            .field(
                "meshes_with_targets",
                mesh_morph_targets.iter().filter(|t| !t.is_empty()).count(),
            )
            .field("total_targets", total_targets)
            .emit();

        // Log per-mesh morph target info
        for (i, targets) in mesh_morph_targets.iter().enumerate() {
            if !targets.is_empty() {
                let verts = vrm_model.meshes[i].vertices.len();
                let max_delta: f32 = targets
                    .iter()
                    .flat_map(|t| t.iter())
                    .flat_map(|d| d.iter())
                    .map(|v| v.abs())
                    .fold(0.0f32, f32::max);
                pipeline_logger::bone(log::Level::Info, "mesh morph targets")
                    .field("mesh_index", i)
                    .field("num_targets", targets.len())
                    .field("num_vertices", verts)
                    .field("max_delta", format!("{:.6}", max_delta))
                    .emit();
            }
        }

        // Log blend shape bindings
        let bindings = vrm_model.blend_shapes.debug_bindings();
        for (preset, binds) in &bindings {
            pipeline_logger::bone(log::Level::Info, "blend shape binding")
                .field("preset", preset.as_str())
                .field("binds", format!("{:?}", binds))
                .emit();
        }
    }

    // Build per-mesh material inputs from VRM materials
    let mesh_materials: Vec<MeshMaterialInput> = vrm_model
        .meshes
        .iter()
        .map(
            |mesh| match mesh.material_index.and_then(|i| vrm_model.materials.get(i)) {
                Some(mat) => MeshMaterialInput {
                    base_color: mat.base_color,
                    shade_color: mat.shade_color,
                    rim_color: mat.rim_color,
                    shade_shift: mat.shade_shift,
                    shade_toony: mat.shade_toony,
                    rim_power: mat.rim_power,
                    rim_lift: mat.rim_lift,
                    base_color_texture: mat.base_color_texture.clone(),
                },
                None => MeshMaterialInput::default(),
            },
        )
        .collect();

    let prefs = UserPrefs::load();
    log::info!("User prefs loaded: {:?}", prefs);

    let mut scene = Scene::new(
        &render_ctx.device,
        &render_ctx.queue,
        &render_ctx.config,
        &vertices_list,
        &mesh_materials,
        &mesh_morph_targets,
        max_joints,
        &prefs.stage_lighting,
    );

    // Apply background config
    scene.set_clear_color(prefs.background.to_wgpu_color());
    let mut video_session: Option<Box<dyn video_decoder::VideoSession>> = None;
    if let Some(bg_path) = &prefs.background.image_path {
        if is_video_file(bg_path) {
            // Open video decoder session for video backgrounds
            #[cfg(target_os = "macos")]
            let native_handle = video_decoder::NativeHandle::Metal {
                texture: std::ptr::null_mut(),
                device: std::ptr::null_mut(),
            };
            #[cfg(not(target_os = "macos"))]
            let native_handle = video_decoder::NativeHandle::Wgpu {
                queue: std::ptr::null(),
                texture_id: 0,
            };
            let output = video_decoder::OutputTarget {
                native_handle,
                format: video_decoder::PixelFormat::Rgba8Srgb,
                width: 1280,
                height: 720,
                color_space: video_decoder::ColorSpace::default(),
            };
            match video_decoder::open(bg_path, output, video_decoder::SessionConfig::default()) {
                Ok(session) => {
                    let info = session.info();
                    log::info!(
                        "Video background opened: {bg_path} ({}x{}, {:.1}fps, {:?})",
                        info.width,
                        info.height,
                        info.fps,
                        info.backend
                    );
                    // Create GPU texture for video frames
                    if let Err(e) = scene.set_background_video(
                        &render_ctx.device,
                        &render_ctx.queue,
                        render_ctx.config.format,
                        info.width,
                        info.height,
                    ) {
                        log::warn!("Failed to create video background texture: {e}");
                    }
                    video_session = Some(session);
                }
                Err(e) => log::warn!("Failed to open video background '{bg_path}': {e}"),
            }
        } else {
            match scene.set_background_image_from_path(
                &render_ctx.device,
                &render_ctx.queue,
                render_ctx.config.format,
                Some(bg_path),
            ) {
                Ok(()) => log::info!("Background image loaded: {bg_path}"),
                Err(e) => log::warn!("Failed to load background image '{bg_path}': {e}"),
            }
        }
    }

    // 4. Initialize ML tracker on a background thread (face-only mode for debugging)
    let face_path = assets_base.join("models/face_landmark.onnx");
    let pose_path = assets_base.join("models/pose_landmark.onnx");
    let hand_path = assets_base.join("models/hand_landmark.onnx");
    let tracker = HolisticTracker::new(
        face_path.to_str().unwrap_or(FACE_MODEL_PATH),
        pose_path.to_str().unwrap_or(POSE_MODEL_PATH),
        hand_path.to_str().unwrap_or(HAND_MODEL_PATH),
    )
    .context("Failed to initialize ML tracker. Run: sh scripts/setup.sh download-models")?;
    let tracker_thread = TrackerThread::new_with_mode(tracker, true);

    // 5. Initialize webcam via nokhwa
    let camera = match init_camera() {
        Ok(cam) => {
            pipeline_logger::camera(log::Level::Info, "webcam initialized")
                .field("format", format!("{:?}", cam.camera_format()))
                .emit();
            Some(cam)
        }
        Err(e) => {
            pipeline_logger::camera(log::Level::Warn, "webcam init failed, using dummy frames")
                .field("error", format!("{e}"))
                .emit();
            None
        }
    };

    // 6. Initialize debug overlay
    let debug_overlay = DebugOverlay::new(
        &render_ctx.device,
        &render_ctx.queue,
        render_ctx.config.format,
    );

    let default_anim_path = assets_base.join("animations/idle.glb");
    let default_anim_str = default_anim_path
        .to_str()
        .unwrap_or(DEFAULT_ANIMATION_PATH)
        .to_owned();
    let anim_path = prefs
        .animation_path
        .as_deref()
        .unwrap_or(&default_anim_str);
    let idle_animation = load_idle_animation(&vrm_model, anim_path);

    // Restore mascot mode if it was active in previous session
    let mut mascot = crate::mascot::MascotState::new();
    mascot.always_on_top = prefs.always_on_top;
    if prefs.mascot_mode {
        mascot.enter(&render_ctx.window, prefs.always_on_top);
        render_ctx.set_transparent(true);
        scene.set_clear_alpha(0.0);
    }

    // 7. Initialize Lua-ImGui overlay
    let lua_imgui = match lua_imgui::LuaImgui::new(
        &render_ctx.device,
        &render_ctx.queue,
        render_ctx.config.format,
        &render_ctx.window,
    ) {
        Ok(mut li) => {
            let script_path = assets_base.join("scripts/ui.lua");
            if script_path.exists() {
                if let Err(e) = li.load_script(&script_path) {
                    log::warn!("Failed to load UI script: {e}");
                }
            }
            log::info!("Lua-ImGui initialized");
            Some(li)
        }
        Err(e) => {
            log::warn!("Failed to initialize Lua-ImGui: {e}");
            None
        }
    };

    Ok(AppState {
        render_ctx,
        scene,
        debug_overlay,
        vrm_model,
        tracker_thread,
        camera,
        rig: RigState::default(),
        rig_config: RigConfig::default(),
        last_frame_time: Instant::now(),
        rig_dirty: true,
        last_tracking_result: None,
        camera_distance: prefs.camera_distance,
        last_camera_frame: None,
        blink_mode: prefs.blink_mode,
        auto_blink: AutoBlink::new(),
        stage_lighting: prefs.stage_lighting.clone(),
        #[cfg(target_os = "macos")]
        vcam: None,
        vcam_enabled: true,
        vcam_last_send: Instant::now(),
        idle_animation,
        tracking_enabled: true,
        animation_path: prefs.animation_path,
        background: prefs.background,
        video_session,
        fps_counter: 0,
        fps_decode_counter: 0,
        fps_timer: std::time::Instant::now(),
        lua_imgui,
        mascot,
        last_cursor_pos: winit::dpi::PhysicalPosition::new(0.0, 0.0),
        mascot_alpha_map: Vec::new(),
        mascot_alpha_width: 0,
        mascot_alpha_height: 0,
    })
}

/// Try to load the idle animation clip. Returns None if the file is missing.
///
/// Extracts the VRM bind pose rotations and passes them to the animation player
/// so delta rotations from Mixamo can be correctly applied.
fn load_idle_animation(
    vrm_model: &vrm::model::VrmModel,
    path: &str,
) -> Option<vrm::animation_player::AnimationPlayer> {
    use std::collections::HashMap;
    use std::path::Path;
    use vrm::bone::HumanoidBoneName;

    if !Path::new(path).exists() {
        log::info!("No animation found at {path}, skipping");
        return None;
    }
    match vrm::animation::AnimationClip::load(path) {
        Ok(clip) => {
            log::info!(
                "Idle animation loaded: '{}' ({:.2}s, {} channels)",
                clip.name,
                clip.duration,
                clip.channels.len()
            );
            let mut player = vrm::animation_player::AnimationPlayer::new(clip);

            // Extract VRM bind pose: bone name → bind rotation from node_transforms
            let mut bind_pose = HashMap::new();
            let all_bones = [
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
                HumanoidBoneName::RightShoulder,
                HumanoidBoneName::RightUpperArm,
                HumanoidBoneName::RightLowerArm,
                HumanoidBoneName::RightHand,
                HumanoidBoneName::LeftUpperLeg,
                HumanoidBoneName::LeftLowerLeg,
                HumanoidBoneName::LeftFoot,
                HumanoidBoneName::LeftToes,
                HumanoidBoneName::RightUpperLeg,
                HumanoidBoneName::RightLowerLeg,
                HumanoidBoneName::RightFoot,
                HumanoidBoneName::RightToes,
            ];
            for bone_name in &all_bones {
                if let Some(bone) = vrm_model.humanoid_bones.get(*bone_name) {
                    if let Some(nt) = vrm_model.node_transforms.get(bone.node_index) {
                        bind_pose.insert(*bone_name, nt.rotation);
                    }
                }
            }
            player.set_vrm_bind_pose(bind_pose);

            Some(player)
        }
        Err(e) => {
            log::warn!("Failed to load idle animation: {e}");
            None
        }
    }
}

/// Try to initialize the default webcam (index 0) at 640x480.
fn init_camera() -> Result<nokhwa::Camera> {
    // On macOS, request camera permission via AVFoundation before creating the camera.
    // This blocks until the user grants/denies permission.
    let (tx, rx) = std::sync::mpsc::channel();
    nokhwa::nokhwa_initialize(move |granted| {
        let _ = tx.send(granted);
    });
    let granted = rx
        .recv_timeout(std::time::Duration::from_secs(30))
        .unwrap_or(false);
    if !granted {
        anyhow::bail!("Camera permission denied by user");
    }

    let index = CameraIndex::Index(0);
    let format = CameraFormat::new_from(640, 480, FrameFormat::YUYV, 30);
    let requested = RequestedFormat::new::<RgbFormat>(RequestedFormatType::Closest(format));
    let mut camera = nokhwa::Camera::new(index, requested).context("Failed to create camera")?;
    camera
        .open_stream()
        .context("Failed to open camera stream")?;
    log::info!("Webcam initialized: {:?}", camera.camera_format());
    Ok(camera)
}
