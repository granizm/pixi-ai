use wgpu::util::DeviceExt;

use crate::context::RenderContext;
use crate::texture::GpuTexture;

/// Vertex for 2D overlay rendering.
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct OverlayVertex {
    position: [f32; 2],
    color: [f32; 4],
    uv: [f32; 2],
}

/// Preview rectangle in NDC coordinates (bottom-right corner, 30% of screen).
const PREVIEW_RIGHT: f32 = 1.0;
const PREVIEW_LEFT: f32 = 0.4;
const PREVIEW_TOP: f32 = -0.4;
const PREVIEW_BOTTOM: f32 = -1.0;

/// Landmark dot half-size in NDC.
const DOT_SIZE: f32 = 0.006;
/// Line half-width in NDC.
const LINE_WIDTH: f32 = 0.002;

/// Colors for different landmark types.
const COLOR_POSE_LINE: [f32; 4] = [0.0, 0.81, 0.97, 1.0]; // #00cff7
const COLOR_POSE_DOT: [f32; 4] = [1.0, 0.01, 0.39, 1.0]; // #ff0364
const COLOR_LEFT_HAND_LINE: [f32; 4] = [0.92, 0.06, 0.39, 1.0]; // #eb1064
const COLOR_LEFT_HAND_DOT: [f32; 4] = [0.0, 0.81, 0.97, 1.0]; // #00cff7
const COLOR_RIGHT_HAND_LINE: [f32; 4] = [0.13, 0.76, 0.89, 1.0]; // #22c3e3
const COLOR_RIGHT_HAND_DOT: [f32; 4] = [1.0, 0.01, 0.39, 1.0]; // #ff0364
const COLOR_FACE_DOT: [f32; 4] = [0.0, 1.0, 0.0, 1.0]; // green

/// MediaPipe pose connections (pairs of landmark indices).
const POSE_CONNECTIONS: &[(usize, usize)] = &[
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
];

/// MediaPipe hand connections.
const HAND_CONNECTIONS: &[(usize, usize)] = &[
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (0, 17),
];

/// Key face landmark indices to draw (eyes, nose, mouth outline — not full tesselation).
#[allow(dead_code)]
const FACE_KEY_INDICES: &[usize] = &[
    // Left eye
    33, 133, 160, 159, 158, 144, 145, 153, // Right eye
    362, 263, 387, 386, 385, 373, 374, 380, // Nose
    1, 2, 98, 327, // Mouth outer
    61, 291, 0, 17, 78, 308, 13, 14, // Eyebrows
    70, 63, 105, 66, 107, 300, 293, 334, 296, 336, // Iris
    468, 473,
];

/// Input data for a single frame's debug overlay.
#[derive(Default)]
pub struct OverlayInput {
    /// Camera frame image (RGB8, 640x480).
    pub camera_frame: Option<image::DynamicImage>,
    /// Pose 2D landmarks (normalized 0–1), 33 points.
    pub pose_2d: Option<Vec<glam::Vec2>>,
    /// Left hand 3D landmarks (normalized 0–1), 21 points.
    pub left_hand: Option<Vec<glam::Vec3>>,
    /// Right hand 3D landmarks (normalized 0–1), 21 points.
    pub right_hand: Option<Vec<glam::Vec3>>,
    /// Face landmarks (normalized 0–1), 468–478 points.
    pub face: Option<Vec<glam::Vec3>>,
    /// HUD text lines to display (top-left corner).
    pub hud_lines: Vec<String>,
}

pub struct DebugOverlay {
    pipeline: wgpu::RenderPipeline,
    bind_group_layout: wgpu::BindGroupLayout,
    /// 1x1 white fallback texture for solid-color drawing (kept alive for bind group).
    #[allow(dead_code)]
    white_texture: GpuTexture,
    /// Camera frame texture (updated each frame).
    camera_texture: Option<GpuTexture>,
    camera_bind_group: Option<wgpu::BindGroup>,
    white_bind_group: wgpu::BindGroup,
}

impl DebugOverlay {
    pub fn new(device: &wgpu::Device, queue: &wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let shader_src = include_str!("../../../assets/shaders/debug_overlay.wgsl");
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("debug_overlay_shader"),
            source: wgpu::ShaderSource::Wgsl(shader_src.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("overlay_bind_group_layout"),
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

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay_pipeline_layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let vertex_layout = wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<OverlayVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                // position
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                // color
                wgpu::VertexAttribute {
                    offset: 8,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // uv
                wgpu::VertexAttribute {
                    offset: 24,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x2,
                },
            ],
        };

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("overlay_pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[vertex_layout],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
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

        let white_texture = GpuTexture::default_white(device, queue);
        let white_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay_white_bind_group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&white_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&white_texture.sampler),
                },
            ],
        });

        Self {
            pipeline,
            bind_group_layout,
            white_texture,
            camera_texture: None,
            camera_bind_group: None,
            white_bind_group,
        }
    }

    /// Update the camera frame texture from a new image.
    pub fn update_camera_frame(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        image: &image::DynamicImage,
    ) {
        let gpu_tex = GpuTexture::from_image(device, queue, image);
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("overlay_camera_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&gpu_tex.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&gpu_tex.sampler),
                },
            ],
        });
        self.camera_texture = Some(gpu_tex);
        self.camera_bind_group = Some(bind_group);
    }

    /// Render the debug overlay onto the given texture view.
    pub fn render(
        &self,
        ctx: &RenderContext,
        view: &wgpu::TextureView,
        input: &OverlayInput,
    ) -> anyhow::Result<()> {
        self.render_raw(&ctx.device, &ctx.queue, view, input)
    }

    /// Render the debug overlay using raw device/queue/view references.
    ///
    /// This variant is useful on platforms (e.g. Android) where a full
    /// [`RenderContext`] — which owns a `wgpu::Surface` — is not available.
    pub fn render_raw(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        view: &wgpu::TextureView,
        input: &OverlayInput,
    ) -> anyhow::Result<()> {
        let mut vertices: Vec<OverlayVertex> = Vec::with_capacity(2048);

        // 1. Camera preview quad (textured)
        let cam_vertices = self.build_camera_quad();

        // 2. HUD text (top-left)
        self.build_hud_text(&mut vertices, &input.hud_lines);

        // 3. Landmark dots and connections (solid color)
        self.build_pose_landmarks(&mut vertices, &input.pose_2d);
        self.build_hand_landmarks(
            &mut vertices,
            &input.left_hand,
            COLOR_LEFT_HAND_LINE,
            COLOR_LEFT_HAND_DOT,
        );
        self.build_hand_landmarks(
            &mut vertices,
            &input.right_hand,
            COLOR_RIGHT_HAND_LINE,
            COLOR_RIGHT_HAND_DOT,
        );
        self.build_face_landmarks(&mut vertices, &input.face);

        let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("overlay_encoder"),
        });

        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("overlay_pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load, // Preserve existing 3D content
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                ..Default::default()
            });

            pass.set_pipeline(&self.pipeline);

            // Draw camera preview quad (with camera texture)
            if !cam_vertices.is_empty() {
                let cam_bg = self
                    .camera_bind_group
                    .as_ref()
                    .unwrap_or(&self.white_bind_group);
                pass.set_bind_group(0, cam_bg, &[]);
                let cam_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("overlay_cam_vb"),
                    contents: bytemuck::cast_slice(&cam_vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                pass.set_vertex_buffer(0, cam_buffer.slice(..));
                pass.draw(0..cam_vertices.len() as u32, 0..1);
            }

            // Draw landmark dots/lines (with white texture → vertex color only)
            if !vertices.is_empty() {
                pass.set_bind_group(0, &self.white_bind_group, &[]);
                let lm_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("overlay_lm_vb"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX,
                });
                pass.set_vertex_buffer(0, lm_buffer.slice(..));
                pass.draw(0..vertices.len() as u32, 0..1);
            }
        }

        queue.submit(std::iter::once(encoder.finish()));
        Ok(())
    }

    /// Build the camera preview quad (6 vertices, textured).
    fn build_camera_quad(&self) -> Vec<OverlayVertex> {
        if self.camera_bind_group.is_none() {
            return Vec::new();
        }
        let white = [1.0, 1.0, 1.0, 0.9]; // Slightly transparent
        vec![
            // Triangle 1
            OverlayVertex {
                position: [PREVIEW_LEFT, PREVIEW_TOP],
                color: white,
                uv: [0.0, 0.0],
            },
            OverlayVertex {
                position: [PREVIEW_RIGHT, PREVIEW_TOP],
                color: white,
                uv: [1.0, 0.0],
            },
            OverlayVertex {
                position: [PREVIEW_LEFT, PREVIEW_BOTTOM],
                color: white,
                uv: [0.0, 1.0],
            },
            // Triangle 2
            OverlayVertex {
                position: [PREVIEW_RIGHT, PREVIEW_TOP],
                color: white,
                uv: [1.0, 0.0],
            },
            OverlayVertex {
                position: [PREVIEW_RIGHT, PREVIEW_BOTTOM],
                color: white,
                uv: [1.0, 1.0],
            },
            OverlayVertex {
                position: [PREVIEW_LEFT, PREVIEW_BOTTOM],
                color: white,
                uv: [0.0, 1.0],
            },
        ]
    }

    /// Render HUD text lines in the top-left corner using bitmap font.
    fn build_hud_text(&self, verts: &mut Vec<OverlayVertex>, lines: &[String]) {
        use crate::bitmap_font;

        let start_x: f32 = -0.98;
        let start_y: f32 = 0.95;
        let bg_color = [0.0, 0.0, 0.0, 0.5];
        let text_color = [1.0, 1.0, 1.0, 0.9];
        let uv = [0.5, 0.5];

        // Draw semi-transparent background
        if !lines.is_empty() {
            let max_chars = lines.iter().map(|l| l.len()).max().unwrap_or(0);
            let bg_w = max_chars as f32 * bitmap_font::CHAR_SPACING + 0.02;
            let bg_h = lines.len() as f32 * bitmap_font::LINE_HEIGHT + 0.015;
            let bg_l = start_x - 0.005;
            let bg_t = start_y + 0.01;
            let bg_r = bg_l + bg_w;
            let bg_b = bg_t - bg_h;

            verts.push(OverlayVertex {
                position: [bg_l, bg_t],
                color: bg_color,
                uv,
            });
            verts.push(OverlayVertex {
                position: [bg_r, bg_t],
                color: bg_color,
                uv,
            });
            verts.push(OverlayVertex {
                position: [bg_l, bg_b],
                color: bg_color,
                uv,
            });
            verts.push(OverlayVertex {
                position: [bg_r, bg_t],
                color: bg_color,
                uv,
            });
            verts.push(OverlayVertex {
                position: [bg_r, bg_b],
                color: bg_color,
                uv,
            });
            verts.push(OverlayVertex {
                position: [bg_l, bg_b],
                color: bg_color,
                uv,
            });
        }

        for (line_idx, line) in lines.iter().enumerate() {
            let y = start_y - line_idx as f32 * bitmap_font::LINE_HEIGHT;
            for (ch_idx, ch) in line.chars().enumerate() {
                let Some(glyph) = bitmap_font::glyph(ch) else {
                    continue;
                };
                let cx = start_x + ch_idx as f32 * bitmap_font::CHAR_SPACING;
                for (row, &bits) in glyph.iter().enumerate() {
                    for col in 0..5 {
                        if bits & (1 << (4 - col)) != 0 {
                            let px = cx + col as f32 * bitmap_font::PIXEL_SIZE;
                            let py = y - row as f32 * bitmap_font::PIXEL_SIZE;
                            let s = bitmap_font::PIXEL_SIZE * 0.45;
                            verts.push(OverlayVertex {
                                position: [px - s, py + s],
                                color: text_color,
                                uv,
                            });
                            verts.push(OverlayVertex {
                                position: [px + s, py + s],
                                color: text_color,
                                uv,
                            });
                            verts.push(OverlayVertex {
                                position: [px - s, py - s],
                                color: text_color,
                                uv,
                            });
                            verts.push(OverlayVertex {
                                position: [px + s, py + s],
                                color: text_color,
                                uv,
                            });
                            verts.push(OverlayVertex {
                                position: [px + s, py - s],
                                color: text_color,
                                uv,
                            });
                            verts.push(OverlayVertex {
                                position: [px - s, py - s],
                                color: text_color,
                                uv,
                            });
                        }
                    }
                }
            }
        }
    }

    fn build_pose_landmarks(
        &self,
        verts: &mut Vec<OverlayVertex>,
        pose_2d: &Option<Vec<glam::Vec2>>,
    ) {
        let Some(landmarks) = pose_2d else { return };
        if landmarks.len() < 33 {
            return;
        }

        // Draw connections
        for &(a, b) in POSE_CONNECTIONS {
            if a < landmarks.len() && b < landmarks.len() {
                let p1 = landmark2d_to_ndc(landmarks[a].x, landmarks[a].y);
                let p2 = landmark2d_to_ndc(landmarks[b].x, landmarks[b].y);
                push_line(verts, p1, p2, LINE_WIDTH, COLOR_POSE_LINE);
            }
        }

        // Draw dots
        for lm in landmarks.iter().take(33) {
            let pos = landmark2d_to_ndc(lm.x, lm.y);
            push_dot(verts, pos, DOT_SIZE, COLOR_POSE_DOT);
        }
    }

    fn build_hand_landmarks(
        &self,
        verts: &mut Vec<OverlayVertex>,
        hand: &Option<Vec<glam::Vec3>>,
        line_color: [f32; 4],
        dot_color: [f32; 4],
    ) {
        let Some(landmarks) = hand else { return };
        if landmarks.len() < 21 {
            return;
        }

        // Hand landmarks are in 3D but we use x,y for 2D overlay
        for &(a, b) in HAND_CONNECTIONS {
            if a < landmarks.len() && b < landmarks.len() {
                let p1 = landmark3d_to_ndc(landmarks[a]);
                let p2 = landmark3d_to_ndc(landmarks[b]);
                push_line(verts, p1, p2, LINE_WIDTH, line_color);
            }
        }

        for lm in landmarks.iter().take(21) {
            let pos = landmark3d_to_ndc(*lm);
            push_dot(verts, pos, DOT_SIZE * 0.8, dot_color);
        }
    }

    fn build_face_landmarks(&self, verts: &mut Vec<OverlayVertex>, face: &Option<Vec<glam::Vec3>>) {
        let Some(landmarks) = face else { return };

        // Draw all face landmarks for debugging (468+ points)
        for lm in landmarks.iter() {
            let pos = landmark3d_to_ndc(*lm);
            push_dot(verts, pos, DOT_SIZE * 0.4, COLOR_FACE_DOT);
        }
    }
}

/// Map a 2D landmark (normalized 0–1) to NDC within the preview rectangle.
fn landmark2d_to_ndc(x: f32, y: f32) -> [f32; 2] {
    let ndc_x = PREVIEW_LEFT + x * (PREVIEW_RIGHT - PREVIEW_LEFT);
    let ndc_y = PREVIEW_TOP - y * (PREVIEW_TOP - PREVIEW_BOTTOM);
    [ndc_x, ndc_y]
}

/// Map a 3D landmark (using x,y components, normalized 0–1) to NDC within the preview rectangle.
fn landmark3d_to_ndc(lm: glam::Vec3) -> [f32; 2] {
    landmark2d_to_ndc(lm.x, lm.y)
}

/// Push a small quad (dot) at the given NDC position.
fn push_dot(verts: &mut Vec<OverlayVertex>, center: [f32; 2], half_size: f32, color: [f32; 4]) {
    let uv = [0.5, 0.5]; // Will sample white from 1x1 texture
    let l = center[0] - half_size;
    let r = center[0] + half_size;
    let t = center[1] + half_size;
    let b = center[1] - half_size;

    verts.push(OverlayVertex {
        position: [l, t],
        color,
        uv,
    });
    verts.push(OverlayVertex {
        position: [r, t],
        color,
        uv,
    });
    verts.push(OverlayVertex {
        position: [l, b],
        color,
        uv,
    });
    verts.push(OverlayVertex {
        position: [r, t],
        color,
        uv,
    });
    verts.push(OverlayVertex {
        position: [r, b],
        color,
        uv,
    });
    verts.push(OverlayVertex {
        position: [l, b],
        color,
        uv,
    });
}

/// Push a thin quad (line) between two NDC points.
fn push_line(
    verts: &mut Vec<OverlayVertex>,
    p1: [f32; 2],
    p2: [f32; 2],
    half_width: f32,
    color: [f32; 4],
) {
    let dx = p2[0] - p1[0];
    let dy = p2[1] - p1[1];
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1e-6 {
        return;
    }
    // Perpendicular direction
    let nx = -dy / len * half_width;
    let ny = dx / len * half_width;

    let uv = [0.5, 0.5];
    let a = [p1[0] + nx, p1[1] + ny];
    let b = [p1[0] - nx, p1[1] - ny];
    let c = [p2[0] + nx, p2[1] + ny];
    let d = [p2[0] - nx, p2[1] - ny];

    verts.push(OverlayVertex {
        position: a,
        color,
        uv,
    });
    verts.push(OverlayVertex {
        position: c,
        color,
        uv,
    });
    verts.push(OverlayVertex {
        position: b,
        color,
        uv,
    });
    verts.push(OverlayVertex {
        position: c,
        color,
        uv,
    });
    verts.push(OverlayVertex {
        position: d,
        color,
        uv,
    });
    verts.push(OverlayVertex {
        position: b,
        color,
        uv,
    });
}
