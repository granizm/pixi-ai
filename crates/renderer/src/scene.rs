use crate::camera::CameraUniform;
use crate::context::RenderContext;
use crate::depth::DepthTexture;
use crate::light::StageLighting;
use crate::mesh::GpuMesh;
use crate::morph::{MorphBindGroupLayout, PerMeshMorph};
use crate::pipeline::create_render_pipeline;
use crate::skin::SkinData;
use crate::texture::GpuTexture;
use crate::vertex::Vertex;
use glam::Mat4;
use wgpu::util::DeviceExt;

/// Per-mesh material data on the GPU.
struct GpuMaterial {
    bind_group: wgpu::BindGroup,
}

/// Material uniform data matching the shader's MaterialUniform struct.
/// Includes MToon toon shading parameters.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct MaterialUniform {
    base_color: [f32; 4],
    shade_color: [f32; 4],
    rim_color: [f32; 4],
    /// Packed: [shade_shift, shade_toony, rim_power, rim_lift]
    mtoon_params: [f32; 4],
}

pub struct Scene {
    meshes: Vec<GpuMesh>,
    materials: Vec<GpuMaterial>,
    skin: SkinData,
    per_mesh_morphs: Vec<PerMeshMorph>,
    depth: DepthTexture,
    pipeline: wgpu::RenderPipeline,
    camera_buffer: wgpu::Buffer,
    lights_buffer: wgpu::Buffer,
    camera_bind_group: wgpu::BindGroup,
    // Kept alive for potential future dynamic material creation.
    #[allow(dead_code)]
    material_bind_group_layout: wgpu::BindGroupLayout,
    /// Background clear color (configurable).
    clear_color: wgpu::Color,
    /// Background image rendering resources (fullscreen quad).
    bg_image: Option<BgImage>,
    /// Video background rendering resources (decoded video frames).
    bg_video: Option<BgVideo>,
    /// Staging resources for reading back rendered frames (virtual camera).
    frame_capture_texture: Option<wgpu::Texture>,
    frame_capture_depth: Option<DepthTexture>,
    /// Double-buffered staging buffers for async GPU readback.
    /// Index alternates each frame to overlap copy and readback.
    frame_capture_buffers: [Option<wgpu::Buffer>; 2],
    frame_capture_buf_idx: usize,
    /// Signals that the previous buffer's map_async callback has fired.
    frame_capture_map_ready: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// Whether the previous buffer has a pending readback.
    frame_capture_pending: bool,
    frame_capture_width: u32,
    frame_capture_height: u32,
}

/// GPU resources for rendering a fullscreen background image (static or animated GIF).
struct BgImage {
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    texture: wgpu::Texture,
    tex_width: u32,
    tex_height: u32,
    /// Animation frames (RGBA bytes) and per-frame delay. Empty for static images.
    frames: Vec<BgFrame>,
    /// Current animation frame index.
    current_frame: usize,
    /// Accumulated time since last frame switch.
    frame_elapsed: std::time::Duration,
}

/// A single frame of an animated background.
struct BgFrame {
    rgba: Vec<u8>,
    delay: std::time::Duration,
}

/// GPU resources for rendering a decoded video frame as a fullscreen background.
#[allow(dead_code)]
struct BgVideo {
    bind_group: wgpu::BindGroup,
    pipeline: wgpu::RenderPipeline,
    texture: wgpu::Texture,
    width: u32,
    height: u32,
}

/// Input data for one mesh's material (base color + MToon params + optional texture image).
pub struct MeshMaterialInput {
    pub base_color: [f32; 4],
    pub shade_color: [f32; 4],
    pub rim_color: [f32; 4],
    pub shade_shift: f32,
    pub shade_toony: f32,
    pub rim_power: f32,
    pub rim_lift: f32,
    pub base_color_texture: Option<image::DynamicImage>,
}

impl Default for MeshMaterialInput {
    fn default() -> Self {
        Self {
            base_color: [1.0, 1.0, 1.0, 1.0],
            shade_color: [0.5, 0.5, 0.5, 1.0],
            rim_color: [0.0, 0.0, 0.0, 1.0],
            shade_shift: -0.1,
            shade_toony: 0.5,
            rim_power: 1.0,
            rim_lift: 0.0,
            base_color_texture: None,
        }
    }
}

impl Scene {
    /// Create a new scene.
    ///
    /// `mesh_morph_targets[i]` contains the morph target position deltas for mesh `i`.
    /// Each inner `Vec<[f32; 3]>` has one `[dx, dy, dz]` per vertex for that morph target.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        config: &wgpu::SurfaceConfiguration,
        vertices_list: &[(&[Vertex], &[u32])],
        mesh_materials: &[MeshMaterialInput],
        mesh_morph_targets: &[Vec<Vec<[f32; 3]>>],
        max_joints: usize,
        stage_lighting: &StageLighting,
    ) -> Self {
        let meshes: Vec<GpuMesh> = vertices_list
            .iter()
            .map(|(verts, indices)| GpuMesh::from_vertices_indices(device, verts, indices))
            .collect();

        let skin = SkinData::new(device, max_joints.max(1));
        let morph_layout = MorphBindGroupLayout::new(device);
        let depth = DepthTexture::new(device, config.width, config.height);

        // Create per-mesh morph target data
        let per_mesh_morphs: Vec<PerMeshMorph> = vertices_list
            .iter()
            .enumerate()
            .map(|(i, (verts, _))| {
                let targets = mesh_morph_targets.get(i);
                let empty_targets = Vec::new();
                let target_deltas: &[Vec<[f32; 3]>] = match targets {
                    Some(t) => t.as_slice(),
                    None => &empty_targets,
                };
                PerMeshMorph::new(device, &morph_layout, verts.len(), target_deltas)
            })
            .collect();

        // Camera uniform
        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera_buffer"),
            contents: bytemuck::bytes_of(&CameraUniform::default()),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // Lights uniform (shares bind group 0 with camera)
        let lights_uniform = stage_lighting.to_uniform();
        let lights_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("lights_buffer"),
            contents: bytemuck::bytes_of(&lights_uniform),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let camera_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("camera_lights_bind_group_layout"),
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });

        let camera_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("camera_lights_bind_group"),
            layout: &camera_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: lights_buffer.as_entire_binding(),
                },
            ],
        });

        // Material bind group layout (shared by all meshes)
        let material_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("material_bind_group_layout"),
                entries: &[
                    // MaterialUniform
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    // Base color texture
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Texture {
                            sample_type: wgpu::TextureSampleType::Float { filterable: true },
                            view_dimension: wgpu::TextureViewDimension::D2,
                            multisampled: false,
                        },
                        count: None,
                    },
                    // Sampler
                    wgpu::BindGroupLayoutEntry {
                        binding: 2,
                        visibility: wgpu::ShaderStages::FRAGMENT,
                        ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                        count: None,
                    },
                ],
            });

        // Create default white texture as fallback
        let default_texture = GpuTexture::default_white(device, queue);

        // Create per-mesh material bind groups
        let materials: Vec<GpuMaterial> = (0..meshes.len())
            .map(|i| {
                let mat_input = mesh_materials.get(i);
                let defaults = MeshMaterialInput::default();
                let base_color = mat_input
                    .map(|m| m.base_color)
                    .unwrap_or(defaults.base_color);
                let shade_color = mat_input
                    .map(|m| m.shade_color)
                    .unwrap_or(defaults.shade_color);
                let rim_color = mat_input.map(|m| m.rim_color).unwrap_or(defaults.rim_color);
                let shade_shift = mat_input
                    .map(|m| m.shade_shift)
                    .unwrap_or(defaults.shade_shift);
                let shade_toony = mat_input
                    .map(|m| m.shade_toony)
                    .unwrap_or(defaults.shade_toony);
                let rim_power = mat_input.map(|m| m.rim_power).unwrap_or(defaults.rim_power);
                let rim_lift = mat_input.map(|m| m.rim_lift).unwrap_or(defaults.rim_lift);

                let material_uniform = MaterialUniform {
                    base_color,
                    shade_color,
                    rim_color,
                    mtoon_params: [shade_shift, shade_toony, rim_power, rim_lift],
                };
                let material_buffer =
                    device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&format!("material_buffer_{i}")),
                        contents: bytemuck::bytes_of(&material_uniform),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });

                // Use the mesh's texture if available, otherwise fall back to default white
                let gpu_texture = mat_input
                    .and_then(|m| m.base_color_texture.as_ref())
                    .map(|img| GpuTexture::from_image(device, queue, img));

                let (tex_view, tex_sampler) = match &gpu_texture {
                    Some(t) => (&t.view, &t.sampler),
                    None => (&default_texture.view, &default_texture.sampler),
                };

                let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some(&format!("material_bind_group_{i}")),
                    layout: &material_bind_group_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: material_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(tex_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::Sampler(tex_sampler),
                        },
                    ],
                });

                GpuMaterial { bind_group }
            })
            .collect();

        let shader_src = include_str!("../../../assets/shaders/skinning.wgsl");
        let pipeline = create_render_pipeline(
            device,
            config.format,
            shader_src,
            &[
                &camera_bind_group_layout,
                skin.bind_group_layout(),
                morph_layout.layout(),
                &material_bind_group_layout,
            ],
            Some(crate::depth::DEPTH_FORMAT),
        );

        Self {
            meshes,
            materials,
            skin,
            per_mesh_morphs,
            depth,
            pipeline,
            camera_buffer,
            lights_buffer,
            camera_bind_group,
            material_bind_group_layout,
            clear_color: wgpu::Color {
                r: 0.12,
                g: 0.12,
                b: 0.15,
                a: 1.0,
            },
            bg_image: None,
            bg_video: None,
            frame_capture_texture: None,
            frame_capture_depth: None,
            frame_capture_buffers: [None, None],
            frame_capture_buf_idx: 0,
            frame_capture_map_ready: std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false)),
            frame_capture_pending: false,
            frame_capture_width: 0,
            frame_capture_height: 0,
        }
    }

    /// Update GPU buffers before rendering.
    ///
    /// `per_mesh_morph_weights[i]` contains the morph weights for mesh `i`.
    pub fn prepare(
        &self,
        queue: &wgpu::Queue,
        joint_matrices: &[Mat4],
        per_mesh_morph_weights: &[Vec<f32>],
        camera_uniform: &CameraUniform,
        stage_lighting: &StageLighting,
    ) {
        queue.write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(camera_uniform));
        if !joint_matrices.is_empty() {
            self.skin.update(queue, joint_matrices);
        }
        for (i, morph) in self.per_mesh_morphs.iter().enumerate() {
            if let Some(weights) = per_mesh_morph_weights.get(i) {
                morph.update(queue, weights);
            }
        }
        let lights_uniform = stage_lighting.to_uniform();
        queue.write_buffer(&self.lights_buffer, 0, bytemuck::bytes_of(&lights_uniform));
    }

    /// Render the 3D scene to a texture view using raw device/queue references.
    /// This is the core implementation used by both desktop and Android paths.
    pub fn render_to_view_raw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
    ) {
        self.render_to_view_with_depth_raw(device, queue, view, &self.depth.view);
    }

    /// Render the 3D scene to a texture view (does not acquire or present the surface).
    pub fn render_to_view(&self, ctx: &RenderContext, view: &wgpu::TextureView) {
        self.render_to_view_raw(&ctx.device, &ctx.queue, view);
    }

    /// Render the 3D scene to a texture view with a specified depth buffer (raw device/queue).
    fn render_to_view_with_depth_raw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        let mut encoder = device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        // Background: video takes priority over static image
        let has_background = if let Some(bg_video) = &self.bg_video {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bg_video_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&bg_video.pipeline);
            pass.set_bind_group(0, &bg_video.bind_group, &[]);
            pass.draw(0..3, 0..1);
            true
        } else if let Some(bg) = &self.bg_image {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("bg_image_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(self.clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });
            pass.set_pipeline(&bg.pipeline);
            pass.set_bind_group(0, &bg.bind_group, &[]);
            pass.draw(0..3, 0..1);
            true
        } else {
            false
        };
        // Main 3D scene pass
        {
            // When the clear color is transparent (alpha < 1.0, i.e. mascot mode),
            // always clear to prevent stale/ghost pixels from previous frames.
            let clear_or_load = if has_background && self.clear_color.a >= 1.0 {
                wgpu::LoadOp::Load // preserve background image (opaque mode only)
            } else {
                wgpu::LoadOp::Clear(self.clear_color)
            };
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: clear_or_load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &self.camera_bind_group, &[]);
            pass.set_bind_group(1, self.skin.bind_group(), &[]);
            for (i, mesh) in self.meshes.iter().enumerate() {
                if let Some(morph) = self.per_mesh_morphs.get(i) {
                    pass.set_bind_group(2, morph.bind_group(), &[]);
                }
                if let Some(mat) = self.materials.get(i) {
                    pass.set_bind_group(3, &mat.bind_group, &[]);
                }
                mesh.draw(&mut pass);
            }
        }
        queue.submit(std::iter::once(encoder.finish()));
    }

    /// Render the 3D scene to a texture view with a specified depth buffer.
    fn render_to_view_with_depth(
        &self,
        ctx: &RenderContext,
        view: &wgpu::TextureView,
        depth_view: &wgpu::TextureView,
    ) {
        self.render_to_view_with_depth_raw(&ctx.device, &ctx.queue, view, depth_view);
    }

    /// Render the 3D scene to the frame capture texture (uses its own depth buffer).
    pub fn render_to_capture(&self, ctx: &RenderContext) {
        if let (Some(view), Some(depth)) = (self.frame_capture_view(), &self.frame_capture_depth) {
            self.render_to_view_with_depth(ctx, &view, &depth.view);
        }
    }

    /// Acquire surface, render, and present (convenience wrapper for non-overlay usage).
    pub fn render(&self, ctx: &RenderContext) -> anyhow::Result<()> {
        let output = ctx.surface.get_current_texture()?;
        let view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.render_to_view(ctx, &view);
        output.present();
        Ok(())
    }

    pub fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.depth.resize(device, width, height);
    }

    /// Set the background clear color.
    pub fn set_clear_color(&mut self, color: wgpu::Color) {
        self.clear_color = color;
    }

    /// Set the alpha component of the background clear color.
    /// Use 0.0 for transparent (mascot mode) and 1.0 for opaque (normal mode).
    pub fn set_clear_alpha(&mut self, alpha: f64) {
        self.clear_color.a = alpha;
    }

    /// Set a background from an image file path.
    /// Supports static images (PNG/JPEG) and animated GIFs.
    /// Pass None to remove the background image.
    pub fn set_background_image_from_path(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        path: Option<&str>,
    ) -> anyhow::Result<()> {
        match path {
            Some(p) => {
                self.bg_image = Some(BgImage::load(device, queue, surface_format, p)?);
                Ok(())
            }
            None => {
                self.bg_image = None;
                Ok(())
            }
        }
    }

    /// Advance the background animation by the given delta time.
    /// Call this every frame. No-op for static images.
    pub fn tick_background(&mut self, queue: &wgpu::Queue, dt: std::time::Duration) {
        if let Some(bg) = &mut self.bg_image {
            bg.tick(queue, dt);
        }
    }

    /// Create a video background texture for receiving decoded frames.
    pub fn set_background_video(
        &mut self,
        device: &wgpu::Device,
        _queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> anyhow::Result<()> {
        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("bg_video_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let (bind_group, pipeline) =
            BgImage::create_gpu_resources(device, &texture, surface_format);
        self.bg_video = Some(BgVideo {
            bind_group,
            pipeline,
            texture,
            width,
            height,
        });
        Ok(())
    }

    /// Remove the video background.
    pub fn remove_background_video(&mut self) {
        self.bg_video = None;
    }

    /// Update the video background texture with new RGBA frame data.
    pub fn update_video_frame(&self, queue: &wgpu::Queue, rgba: &[u8], width: u32, height: u32) {
        if let Some(bg) = &self.bg_video {
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &bg.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }

    /// Ensure the frame capture texture and double staging buffers exist at the given dimensions.
    pub fn ensure_frame_capture(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        if self.frame_capture_width == width
            && self.frame_capture_height == height
            && self.frame_capture_texture.is_some()
        {
            return;
        }

        let texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("frame_capture_texture"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Bgra8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });

        // Bytes per row must be aligned to 256 (wgpu requirement)
        let bytes_per_row = (width * 4 + 255) & !255;
        let buf_size = (bytes_per_row * height) as u64;
        let make_buffer = |label| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: buf_size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            })
        };

        self.frame_capture_texture = Some(texture);
        self.frame_capture_buffers = [
            Some(make_buffer("frame_capture_buf_0")),
            Some(make_buffer("frame_capture_buf_1")),
        ];
        self.frame_capture_buf_idx = 0;
        self.frame_capture_map_ready
            .store(false, std::sync::atomic::Ordering::Release);
        self.frame_capture_pending = false;
        self.frame_capture_depth = Some(DepthTexture::new(device, width, height));
        self.frame_capture_width = width;
        self.frame_capture_height = height;
    }

    /// Get the frame capture texture view for rendering.
    pub fn frame_capture_view(&self) -> Option<wgpu::TextureView> {
        self.frame_capture_texture
            .as_ref()
            .map(|t| t.create_view(&wgpu::TextureViewDescriptor::default()))
    }

    /// Copy the rendered frame capture texture into the current staging buffer
    /// and initiate an async map request. Returns the previous frame's BGRA data if available.
    ///
    /// Double-buffer flow: copy into buffer[idx], read back buffer[1-idx] from previous frame.
    /// Uses non-blocking poll — if the previous frame's mapping isn't ready, skips the readback.
    pub fn capture_frame_async(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
    ) -> Option<Vec<u8>> {
        let texture = self.frame_capture_texture.as_ref()?;
        let width = self.frame_capture_width;
        let height = self.frame_capture_height;
        let aligned_bpr = (width * 4 + 255) & !255;
        let cur = self.frame_capture_buf_idx;
        let prev = 1 - cur;

        // 1. Non-blocking poll to process GPU callbacks (map_async completion)
        device.poll(wgpu::Maintain::Poll);

        // 2. Try to read back the previous buffer if mapping is complete
        let result = if self.frame_capture_pending
            && self
                .frame_capture_map_ready
                .load(std::sync::atomic::Ordering::Acquire)
        {
            if let Some(buf) = &self.frame_capture_buffers[prev] {
                let data = buf.slice(..).get_mapped_range();
                let mut pixels = Vec::with_capacity((width * height * 4) as usize);
                for row in 0..height {
                    let start = (row * aligned_bpr) as usize;
                    let end = start + (width * 4) as usize;
                    pixels.extend_from_slice(&data[start..end]);
                }
                drop(data);
                buf.unmap();
                self.frame_capture_map_ready
                    .store(false, std::sync::atomic::Ordering::Release);
                Some(pixels)
            } else {
                None
            }
        } else {
            None
        };

        // 3. Copy texture → current staging buffer
        // Only issue a new copy+map if the previous mapping has been consumed (or first frame).
        // If the previous buffer is still mapped, skip this frame to avoid wgpu validation error.
        let can_copy = !self.frame_capture_pending
            || self
                .frame_capture_map_ready
                .load(std::sync::atomic::Ordering::Acquire)
            || result.is_some(); // result.is_some() means we just consumed the previous mapping

        if can_copy {
            if let Some(buf) = &self.frame_capture_buffers[cur] {
                let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("vcam_copy"),
                });
                encoder.copy_texture_to_buffer(
                    wgpu::TexelCopyTextureInfo {
                        texture,
                        mip_level: 0,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyBufferInfo {
                        buffer: buf,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: 0,
                            bytes_per_row: Some(aligned_bpr),
                            rows_per_image: Some(height),
                        },
                    },
                    wgpu::Extent3d {
                        width,
                        height,
                        depth_or_array_layers: 1,
                    },
                );
                queue.submit(std::iter::once(encoder.finish()));

                // 4. Start async map on the current buffer with completion signal
                let ready = self.frame_capture_map_ready.clone();
                buf.slice(..).map_async(wgpu::MapMode::Read, move |_| {
                    ready.store(true, std::sync::atomic::Ordering::Release);
                });

                // 5. Swap buffers
                self.frame_capture_buf_idx = prev;
                self.frame_capture_pending = true;
            }
        }

        result
    }
}

impl BgImage {
    /// Load a background image from a file path.
    /// Detects GIF by extension and decodes all frames for animation.
    fn load(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_format: wgpu::TextureFormat,
        path: &str,
    ) -> anyhow::Result<Self> {
        use std::path::Path;
        let ext = Path::new(path)
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_lowercase())
            .unwrap_or_default();

        let (first_rgba, w, h, frames) = if ext == "gif" {
            Self::decode_gif(path)?
        } else {
            let img = image::open(path)?;
            let rgba = img.to_rgba8();
            let (w, h) = rgba.dimensions();
            (rgba.into_raw(), w, h, Vec::new())
        };

        let texture = device.create_texture_with_data(
            queue,
            &wgpu::TextureDescriptor {
                label: Some("bg_image_texture"),
                size: wgpu::Extent3d {
                    width: w,
                    height: h,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::Rgba8UnormSrgb,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            &first_rgba,
        );

        let (bind_group, pipeline) = Self::create_gpu_resources(device, &texture, surface_format);

        Ok(Self {
            bind_group,
            pipeline,
            texture,
            tex_width: w,
            tex_height: h,
            frames,
            current_frame: 0,
            frame_elapsed: std::time::Duration::ZERO,
        })
    }

    /// Decode an animated GIF into frames. Returns (first_frame_rgba, w, h, frames).
    fn decode_gif(path: &str) -> anyhow::Result<(Vec<u8>, u32, u32, Vec<BgFrame>)> {
        use image::codecs::gif::GifDecoder;
        use image::AnimationDecoder;
        use std::io::BufReader;

        let file = std::fs::File::open(path)?;
        let decoder = GifDecoder::new(BufReader::new(file))?;
        let raw_frames: Vec<image::Frame> = decoder.into_frames().collect::<Result<Vec<_>, _>>()?;

        if raw_frames.is_empty() {
            anyhow::bail!("GIF has no frames: {path}");
        }

        let first = &raw_frames[0];
        let w = first.buffer().width();
        let h = first.buffer().height();

        let frames: Vec<BgFrame> = raw_frames
            .iter()
            .map(|f| {
                let (numer, denom) = f.delay().numer_denom_ms();
                let delay_ms = if denom == 0 { numer } else { numer / denom };
                // GIF frames with 0 or very short delay default to ~100ms
                let delay = std::time::Duration::from_millis(delay_ms.max(20) as u64);
                BgFrame {
                    rgba: f.buffer().as_raw().clone(),
                    delay,
                }
            })
            .collect();

        let first_rgba = frames[0].rgba.clone();
        log::info!("GIF decoded: {}x{}, {} frames", w, h, frames.len());

        Ok((first_rgba, w, h, frames))
    }

    /// Create bind group and pipeline for the background texture.
    pub(crate) fn create_gpu_resources(
        device: &wgpu::Device,
        texture: &wgpu::Texture,
        surface_format: wgpu::TextureFormat,
    ) -> (wgpu::BindGroup, wgpu::RenderPipeline) {
        let tex_view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("bg_image_bgl"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bg_image_bg"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&tex_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&sampler),
                },
            ],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("bg_image_shader"),
            source: wgpu::ShaderSource::Wgsl(BG_IMAGE_WGSL.into()),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("bg_image_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("bg_image_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview: None,
            cache: None,
        });

        (bind_group, pipeline)
    }

    /// Advance animation by dt. Uploads the next frame's pixel data to the GPU texture.
    fn tick(&mut self, queue: &wgpu::Queue, dt: std::time::Duration) {
        if self.frames.len() <= 1 {
            return;
        }
        self.frame_elapsed += dt;
        let delay = self.frames[self.current_frame].delay;
        if self.frame_elapsed >= delay {
            self.frame_elapsed -= delay;
            self.current_frame = (self.current_frame + 1) % self.frames.len();
            // Upload new frame to GPU
            queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: &self.texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                &self.frames[self.current_frame].rgba,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(self.tex_width * 4),
                    rows_per_image: Some(self.tex_height),
                },
                wgpu::Extent3d {
                    width: self.tex_width,
                    height: self.tex_height,
                    depth_or_array_layers: 1,
                },
            );
        }
    }
}

/// Fullscreen triangle shader for background image rendering.
/// Uses vertex_index to generate a fullscreen triangle without a vertex buffer.
const BG_IMAGE_WGSL: &str = r#"
@group(0) @binding(0) var bg_tex: texture_2d<f32>;
@group(0) @binding(1) var bg_sampler: sampler;

struct VsOutput {
    @builtin(position) position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) idx: u32) -> VsOutput {
    // Fullscreen triangle: 3 vertices cover the entire screen
    let x = f32(i32(idx & 1u)) * 4.0 - 1.0;
    let y = f32(i32(idx >> 1u)) * 4.0 - 1.0;
    var out: VsOutput;
    out.position = vec4<f32>(x, y, 0.0, 1.0);
    // UV: flip Y for texture coordinates
    out.uv = vec2<f32>((x + 1.0) * 0.5, (1.0 - y) * 0.5);
    return out;
}

@fragment
fn fs_main(in: VsOutput) -> @location(0) vec4<f32> {
    return textureSample(bg_tex, bg_sampler, in.uv);
}
"#;
