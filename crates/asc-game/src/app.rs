//! Window, event loop and the frame/tick loop.

use std::sync::Arc;

use asc_render::Renderer;
use winit::application::ApplicationHandler;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::window::{Window, WindowId};

pub fn run() -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app)?;
    if let Some(err) = app.error {
        return Err(err);
    }
    Ok(())
}

#[derive(Default)]
struct App {
    state: Option<State>,
    error: Option<anyhow::Error>,
}

struct State {
    window: Arc<Window>,
    renderer: Renderer,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Ascension")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0));
        let result = event_loop.create_window(attrs).map_err(anyhow::Error::from).and_then(|window| {
            let window = Arc::new(window);
            Ok(State { renderer: Renderer::new(window.clone())?, window })
        });
        match result {
            Ok(state) => self.state = Some(state),
            Err(err) => {
                self.error = Some(err);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let Some(state) = self.state.as_mut() else { return };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.renderer.resize(size.width, size.height),
            WindowEvent::RedrawRequested => state.renderer.render(),
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}
