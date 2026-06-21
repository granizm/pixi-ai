#[cfg(feature = "desktop")]
use std::sync::Arc;

pub struct RenderContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: wgpu::Surface<'static>,
    pub config: wgpu::SurfaceConfiguration,
    #[cfg(feature = "desktop")]
    pub window: Arc<winit::window::Window>,
    adapter: wgpu::Adapter,
}

impl RenderContext {
    /// Create from pre-initialized wgpu components (for Android / embedded use).
    #[cfg(not(feature = "desktop"))]
    pub fn from_parts(
        device: wgpu::Device,
        queue: wgpu::Queue,
        surface: wgpu::Surface<'static>,
        config: wgpu::SurfaceConfiguration,
        adapter: wgpu::Adapter,
    ) -> Self {
        Self {
            device,
            queue,
            surface,
            config,
            adapter,
        }
    }

    #[cfg(feature = "desktop")]
    pub async fn new(window: Arc<winit::window::Window>) -> anyhow::Result<Self> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor::default());
        let surface = instance.create_surface(window.clone())?;
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .ok_or_else(|| anyhow::anyhow!("No adapter"))?;
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor::default(), None)
            .await?;
        let size = window.inner_size();
        let config = surface
            .get_default_config(&adapter, size.width.max(1), size.height.max(1))
            .ok_or_else(|| anyhow::anyhow!("No surface config"))?;
        surface.configure(&device, &config);
        Ok(Self {
            device,
            queue,
            surface,
            config,
            window,
            adapter,
        })
    }

    /// Enable or disable transparent compositing on the surface.
    ///
    /// When `transparent` is true, the surface alpha mode is set to
    /// `PostMultiplied` or `PreMultiplied` (whichever is supported),
    /// allowing the desktop to show through transparent pixels.
    pub fn set_transparent(&mut self, transparent: bool) {
        if transparent {
            let caps = self.surface.get_capabilities(&self.adapter);
            self.config.alpha_mode = if caps
                .alpha_modes
                .contains(&wgpu::CompositeAlphaMode::PostMultiplied)
            {
                wgpu::CompositeAlphaMode::PostMultiplied
            } else if caps
                .alpha_modes
                .contains(&wgpu::CompositeAlphaMode::PreMultiplied)
            {
                wgpu::CompositeAlphaMode::PreMultiplied
            } else {
                log::warn!("No transparent alpha mode available, using Auto");
                wgpu::CompositeAlphaMode::Auto
            };
        } else {
            self.config.alpha_mode = wgpu::CompositeAlphaMode::Auto;
        }
        self.surface.configure(&self.device, &self.config);
        // Force GPU to finish pending work before the new config takes effect.
        // This prevents ghost artifacts when switching to/from transparent mode.
        self.device.poll(wgpu::Maintain::Wait);
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
        }
    }
}
