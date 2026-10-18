//! Window, event loop and the frame/tick loop.

use std::sync::Arc;
use std::time::{Duration, Instant};

use asc_render::{Camera, FrameParams, FrameStats, Frustum, Renderer, UiFrame};
use asc_world::{EditKind, EditOp, Material, RayHit, World, raycast};
use glam::{Vec2, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};

use crate::hud::{self, HudInfo};
use crate::input::Input;
use crate::player::{self, Mode, MoveInput, Player};
use crate::sim::Sim;
use crate::streaming::Streamer;

const SEED: i32 = 1337;
const TICK: Duration = Duration::from_millis(50);
const MAX_TICKS_PER_FRAME: u32 = 5;
const FLY_REACH: f32 = 64.0;
const WALK_REACH: f32 = 5.0;
const EDIT_REPEAT: Duration = Duration::from_millis(110);

#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Fly a fixed route, dig once, log stats and quit. For unattended performance checks.
    pub autopilot: bool,
    /// Walk and jump across the terrain, logging collision clearance. For unattended checks.
    pub walk_test: bool,
    pub time_of_day: Option<f32>,
    pub exit_after: Option<f32>,
    pub view_radius: Option<i32>,
    pub vsync: bool,
    pub hide_hud: bool,
    /// Start pose: x, y, z, yaw, pitch (radians).
    pub camera: Option<[f32; 5]>,
}

impl Options {
    pub fn from_args() -> anyhow::Result<Self> {
        let mut opts = Self::default();
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = |name: &str| args.next().ok_or_else(|| anyhow::anyhow!("{name} needs a value"));
            match arg.as_str() {
                "--autopilot" => opts.autopilot = true,
                "--walk-test" => opts.walk_test = true,
                "--vsync" => opts.vsync = true,
                "--hide-hud" => opts.hide_hud = true,
                "--time" => opts.time_of_day = Some(value("--time")?.parse()?),
                "--exit-after" => opts.exit_after = Some(value("--exit-after")?.parse()?),
                "--view-radius" => opts.view_radius = Some(value("--view-radius")?.parse()?),
                "--camera" => {
                    let parts: Vec<f32> =
                        value("--camera")?.split(',').map(str::parse).collect::<Result<_, _>>()?;
                    let pose: [f32; 5] =
                        parts.try_into().map_err(|_| anyhow::anyhow!("--camera wants x,y,z,yaw,pitch"))?;
                    opts.camera = Some(pose);
                }
                other => anyhow::bail!("unknown argument {other}"),
            }
        }
        Ok(opts)
    }
}

pub fn run(options: Options) -> anyhow::Result<()> {
    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App { options, state: None, error: None };
    event_loop.run_app(&mut app)?;
    if let Some(err) = app.error {
        return Err(err);
    }
    Ok(())
}

struct App {
    options: Options,
    state: Option<State>,
    error: Option<anyhow::Error>,
}

struct State {
    window: Arc<Window>,
    renderer: Renderer,
    world: World,
    streamer: Streamer,
    sim: Sim,
    player: Player,
    spawn: Vec3,
    /// Derived from the player every frame.
    camera: Camera,
    input: Input,
    egui_ctx: egui::Context,
    egui_state: egui_winit::State,
    options: Options,

    captured: bool,
    show_hud: bool,
    vsync: bool,
    fly_speed: f32,
    brush_radius: f32,
    last_edit: Instant,
    hit: Option<RayHit>,

    started: Instant,
    last_frame: Instant,
    tick_accumulator: Duration,
    frame_ms: f32,
    /// Longest frame since the last log line, to catch hitches averages hide.
    worst_frame_ms: f32,
    /// Worst CPU time spent updating, and inside the renderer (encode + submit + present).
    worst_update_ms: f32,
    worst_render_ms: f32,
    /// Worst acquire / submit / present inside the renderer.
    worst_gpu_wait: [f32; 3],
    last_stats: FrameStats,
    last_log: Instant,
    autopilot_dug: bool,
    worst_physics_ms: f32,
    /// Walk test: lowest clearance between the capsule and the ground, and grounded step counts.
    min_clearance: f32,
    grounded_steps: (u32, u32),
    walk_test_anchor: Vec3,
    walk_test_check: Instant,
    walk_test_turn: f32,
    walk_test_blocked: u32,
}

impl State {
    fn new(window: Arc<Window>, options: Options) -> anyhow::Result<Self> {
        let mut renderer = Renderer::new(window.clone())?;
        let renderer_vsync = options.vsync;
        let show_hud = !options.hide_hud;
        if options.vsync {
            renderer.set_vsync(true);
        }
        let world = World::new(SEED);
        let streamer = Streamer::new(world.worldgen.clone(), options.view_radius.unwrap_or(10));
        let mut sim = Sim::default();
        if let Some(t) = options.time_of_day {
            sim.time_of_day = t.rem_euclid(1.0);
        }
        let spawn = find_ground(&world, 0.0, 0.0);
        let mut player = Player::new(spawn);
        player.pitch = -0.15;
        if options.autopilot {
            player.mode = Mode::Fly;
            player.teleport(spawn + Vec3::Y * 25.0);
            player.pitch = -0.3;
        }
        if let Some([x, y, z, yaw, pitch]) = options.camera {
            player.mode = Mode::Fly;
            player.teleport(Vec3::new(x, y, z) - Vec3::Y * player::EYE_HEIGHT);
            (player.yaw, player.pitch) = (yaw, pitch);
        }
        let camera =
            Camera { position: player.eye(), yaw: player.yaw, pitch: player.pitch, ..Default::default() };
        log::info!("player starts at {} ({:?})", player.feet, player.mode);
        let egui_ctx = egui::Context::default();
        let egui_state = egui_winit::State::new(
            egui_ctx.clone(),
            egui::ViewportId::ROOT,
            &window,
            Some(window.scale_factor() as f32),
            None,
            Some(renderer.device().limits().max_texture_dimension_2d as usize),
        );
        let now = Instant::now();
        Ok(Self {
            window,
            renderer,
            world,
            streamer,
            sim,
            player,
            spawn,
            camera,
            input: Input::default(),
            egui_ctx,
            egui_state,
            options,
            captured: false,
            show_hud,
            vsync: renderer_vsync,
            fly_speed: 14.0,
            brush_radius: 3.0,
            last_edit: now,
            hit: None,
            started: now,
            last_frame: now,
            tick_accumulator: Duration::ZERO,
            frame_ms: 0.0,
            worst_frame_ms: 0.0,
            worst_update_ms: 0.0,
            worst_render_ms: 0.0,
            worst_gpu_wait: [0.0; 3],
            last_stats: FrameStats::default(),
            last_log: now,
            autopilot_dug: false,
            worst_physics_ms: 0.0,
            min_clearance: f32::MAX,
            grounded_steps: (0, 0),
            walk_test_anchor: Vec3::ZERO,
            walk_test_check: now,
            walk_test_turn: 0.0,
            walk_test_blocked: 0,
        })
    }

    fn set_captured(&mut self, captured: bool) {
        if captured == self.captured {
            return;
        }
        if captured {
            // Windows can confine but not lock the cursor; raw motion events drive the camera either way.
            let grabbed = self
                .window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| self.window.set_cursor_grab(CursorGrabMode::Confined));
            if let Err(err) = grabbed {
                log::warn!("cursor grab failed: {err}");
            }
        } else {
            let _ = self.window.set_cursor_grab(CursorGrabMode::None);
        }
        self.window.set_cursor_visible(!captured);
        self.captured = captured;
    }

    fn frame(&mut self, event_loop: &ActiveEventLoop) {
        let now = Instant::now();
        let dt = (now - self.last_frame).min(Duration::from_millis(100));
        self.last_frame = now;
        self.frame_ms += (dt.as_secs_f32() * 1e3 - self.frame_ms) * 0.05;
        self.worst_frame_ms = self.worst_frame_ms.max(dt.as_secs_f32() * 1e3);

        // Fixed-rate simulation, decoupled from rendering.
        self.tick_accumulator += dt;
        let mut ticks = 0;
        while self.tick_accumulator >= TICK && ticks < MAX_TICKS_PER_FRAME {
            self.sim.tick();
            self.tick_accumulator -= TICK;
            ticks += 1;
        }
        if ticks == MAX_TICKS_PER_FRAME {
            self.tick_accumulator = Duration::ZERO;
        }

        self.handle_keys(event_loop);
        if self.options.autopilot {
            self.autopilot(dt.as_secs_f32());
        } else {
            self.move_player(dt.as_secs_f32());
        }
        self.camera = Camera {
            position: self.player.eye(),
            yaw: self.player.yaw,
            pitch: self.player.pitch,
            ..self.camera
        };

        let reach = if self.player.mode == Mode::Fly { FLY_REACH } else { WALK_REACH };
        self.hit = raycast(|p| self.world.density_at(p), self.camera.position, self.camera.forward(), reach);
        if self.captured && self.player.mode == Mode::Fly {
            self.edit_terrain();
        }

        let (w, h) = self.renderer.gpu().size();
        let frustum = Frustum::from_view_proj(self.camera.view_proj(w as f32 / h as f32));
        self.streamer.update(&self.world, self.camera.position, &frustum, &mut self.renderer);

        let raw_input = self.egui_state.take_egui_input(&self.window);
        let info = self.hud_info();
        let output = self.egui_ctx.run_ui(raw_input, |ui| hud::draw(ui.ctx(), &info));
        self.egui_state.handle_platform_output(&self.window, output.platform_output);
        let primitives = self.egui_ctx.tessellate(output.shapes, output.pixels_per_point);

        let params = FrameParams {
            camera: self.camera,
            sun_dir: self.sim.sun_dir(),
            view_distance: self.streamer.view_radius as f32 * 32.0,
            brush: self
                .hit
                .filter(|_| self.captured || self.options.autopilot)
                .map(|h| (h.point, self.brush_radius)),
        };
        let ui = UiFrame {
            primitives: &primitives,
            textures: &output.textures_delta,
            pixels_per_point: output.pixels_per_point,
        };
        let render_start = Instant::now();
        self.worst_update_ms = self.worst_update_ms.max((render_start - now).as_secs_f32() * 1e3);
        self.last_stats = self.renderer.render(&params, Some(ui));
        self.worst_render_ms = self.worst_render_ms.max(render_start.elapsed().as_secs_f32() * 1e3);
        let st = &self.last_stats;
        for (w, v) in self.worst_gpu_wait.iter_mut().zip([st.acquire_ms, st.submit_ms, st.present_ms]) {
            *w = w.max(v);
        }

        if self.last_log.elapsed() >= Duration::from_secs(2) {
            self.last_log = Instant::now();
            let worst = std::mem::take(&mut self.worst_frame_ms);
            let (retarget_ms, upload_ms) = self.streamer.take_worst();
            let update_ms = std::mem::take(&mut self.worst_update_ms);
            let render_ms = std::mem::take(&mut self.worst_render_ms);
            let [acq, sub, pre] = std::mem::take(&mut self.worst_gpu_wait);
            let s = self.streamer.stats();
            let physics_ms = std::mem::take(&mut self.worst_physics_ms);
            if self.options.walk_test {
                let (g, n) = std::mem::take(&mut self.grounded_steps);
                log::info!(
                    "walk test: feet {:.1} | grounded {:.0}% | min clearance {:.3} m | blocked turns {} | worst physics {:.2} ms",
                    self.player.feet,
                    100.0 * g as f32 / n.max(1) as f32,
                    std::mem::replace(&mut self.min_clearance, f32::MAX),
                    self.walk_test_blocked,
                    physics_ms,
                );
            }
            log::info!(
                "fps {:.0} ({:.2} ms, worst {:.1}) | chunks drawn {} / gpu {} / loaded {} / queued {} | tris {}k | build {:.2} ms mesh {:.2} ms | edit {} | worst retarget {:.1} upload {:.1} update {:.1} render {:.1} (acquire {:.1} submit {:.1} present {:.1}) ms",
                1000.0 / self.frame_ms.max(0.01),
                self.frame_ms,
                worst,
                self.last_stats.chunks_drawn,
                self.renderer.chunk_count(),
                s.loaded,
                s.queued + s.in_flight,
                self.last_stats.triangles / 1000,
                s.avg_build_ms,
                s.avg_mesh_ms,
                s.last_edit_ms.map_or("-".into(), |ms| format!("{ms:.1} ms")),
                retarget_ms,
                upload_ms,
                update_ms,
                render_ms,
                acq,
                sub,
                pre,
            );
        }
        if let Some(limit) = self.options.exit_after
            && self.started.elapsed().as_secs_f32() >= limit
        {
            event_loop.exit();
        }
        self.input.end_frame();
    }

    fn handle_keys(&mut self, event_loop: &ActiveEventLoop) {
        let input = &self.input;
        if input.pressed(KeyCode::Escape) {
            if self.captured {
                self.set_captured(false);
            } else {
                event_loop.exit();
            }
        }
        if self.input.pressed(KeyCode::KeyF) {
            self.player.mode = match self.player.mode {
                Mode::Walk => Mode::Fly,
                Mode::Fly => Mode::Walk,
            };
            self.player.velocity = Vec3::ZERO;
        }
        if self.input.pressed(KeyCode::F3) {
            self.show_hud = !self.show_hud;
        }
        if self.input.pressed(KeyCode::F5) {
            match self.renderer.reload_shaders() {
                Ok(()) => log::info!("shaders reloaded"),
                Err(err) => log::error!("shader reload failed, keeping the old ones: {err:#}"),
            }
        }
        if self.input.pressed(KeyCode::F6) {
            self.vsync = !self.vsync;
            self.renderer.set_vsync(self.vsync);
        }
        if self.input.pressed(KeyCode::F11) {
            let full = self.window.fullscreen().is_some();
            self.window.set_fullscreen((!full).then_some(Fullscreen::Borderless(None)));
        }
        if self.input.pressed(KeyCode::BracketLeft) {
            self.streamer.view_radius = (self.streamer.view_radius - 1).max(3);
        }
        if self.input.pressed(KeyCode::BracketRight) {
            self.streamer.view_radius = (self.streamer.view_radius + 1).min(24);
        }
        if self.input.held(KeyCode::KeyT) {
            let step = if self.input.held(KeyCode::ShiftLeft) { -1.0 } else { 1.0 };
            // An in-game hour per real second.
            self.sim.time_of_day =
                (self.sim.time_of_day + step * self.frame_ms / 1000.0 / 24.0).rem_euclid(1.0);
        }
        let wheel = self.input.wheel;
        if wheel != 0.0 {
            if self.input.held(KeyCode::AltLeft) {
                self.fly_speed = (self.fly_speed * 1.25f32.powf(wheel)).clamp(1.0, 500.0);
            } else {
                self.brush_radius = (self.brush_radius + wheel * 0.5).clamp(1.0, 8.0);
            }
        }
    }

    fn move_player(&mut self, dt: f32) {
        if self.captured {
            let sensitivity = 0.0022;
            self.player.yaw += self.input.mouse_delta.x * sensitivity;
            self.player.pitch =
                (self.player.pitch - self.input.mouse_delta.y * sensitivity).clamp(-1.55, 1.55);
        }
        let i = &self.input;
        let axis =
            |pos: KeyCode, neg: KeyCode| f32::from(u8::from(i.held(pos))) - f32::from(u8::from(i.held(neg)));
        let mut input = MoveInput {
            forward: axis(KeyCode::KeyW, KeyCode::KeyS),
            right: axis(KeyCode::KeyD, KeyCode::KeyA),
            up: axis(KeyCode::Space, KeyCode::ControlLeft),
            sprint: i.held(KeyCode::ShiftLeft),
            fly_speed: self.fly_speed,
        };
        if self.options.walk_test {
            input = self.walk_test_input();
        }
        let t = Instant::now();
        let world = &self.world;
        let steps = self.player.update(dt, &input, &|p| world.density_at(p));
        if steps > 0 {
            self.worst_physics_ms = self.worst_physics_ms.max(t.elapsed().as_secs_f32() * 1e3 / steps as f32);
        }
        if self.options.walk_test && steps > 0 && self.player.mode == Mode::Walk {
            let r = asc_world::Capsule::PLAYER.radius;
            let low = self.player.feet + Vec3::Y * r;
            let world = &self.world;
            let clearance = asc_world::distance_estimate(&|p| world.density_at(p), low) - r;
            self.min_clearance = self.min_clearance.min(clearance);
            self.grounded_steps.0 += u32::from(self.player.grounded);
            self.grounded_steps.1 += 1;
        }
        if self.player.feet.y < player::KILL_Y {
            log::warn!("fell out of the world at {}; respawning", self.player.feet);
            self.player.teleport(self.spawn);
        }
    }

    /// Walk test: sprint along a slowly turning heading, jumping every few seconds, and turning away
    /// when blocked (as a player would).
    fn walk_test_input(&mut self) -> MoveInput {
        let t = self.started.elapsed().as_secs_f32();
        if self.walk_test_check.elapsed() > Duration::from_secs(1) {
            let moved = (self.player.feet - self.walk_test_anchor).truncate().length();
            if t > 4.0 && moved < 1.0 {
                self.walk_test_turn += 1.6;
                self.walk_test_blocked += 1;
            }
            self.walk_test_anchor = self.player.feet;
            self.walk_test_check = Instant::now();
        }
        self.player.yaw = 0.2 + t * 0.05 + self.walk_test_turn;
        self.player.pitch = -0.1;
        MoveInput {
            forward: if t > 3.0 { 1.0 } else { 0.0 },
            up: if t > 3.0 && (t % 3.0) < 0.1 { 1.0 } else { 0.0 },
            sprint: true,
            ..Default::default()
        }
    }

    /// Scripted flight for unattended checks: settle, dig once in view, then cruise forward.
    fn autopilot(&mut self, dt: f32) {
        let t = self.started.elapsed().as_secs_f32();
        self.player.pitch = -0.3;
        if t > 6.0
            && !self.autopilot_dug
            && let Some(hit) = self.hit
        {
            self.apply_edit(hit, EditKind::Dig);
            self.autopilot_dug = true;
        }
        if t > 10.0 {
            let mut feet = self.player.feet + Vec3::new(1.0, 0.0, 0.2).normalize() * 40.0 * dt;
            let ground = self.world.worldgen.height(feet.x, feet.z);
            feet.y += ((ground + 30.0) - feet.y) * (dt * 1.5).min(1.0);
            self.player.teleport(feet);
        }
    }

    fn edit_terrain(&mut self) {
        let dig = self.input.button_held(MouseButton::Left);
        let fill = self.input.button_held(MouseButton::Right);
        let fresh =
            self.input.button_pressed(MouseButton::Left) || self.input.button_pressed(MouseButton::Right);
        if !(dig || fill) || (!fresh && self.last_edit.elapsed() < EDIT_REPEAT) {
            return;
        }
        let Some(hit) = self.hit else { return };
        self.apply_edit(hit, if dig { EditKind::Dig } else { EditKind::Fill });
    }

    fn apply_edit(&mut self, hit: RayHit, kind: EditKind) {
        let r = self.brush_radius;
        let center = match kind {
            // Dig slightly into the surface; fill slightly out of it so it builds towards the player.
            EditKind::Dig => hit.point - hit.normal * r * 0.3,
            EditKind::Fill => hit.point + hit.normal * r * 0.3,
        };
        if kind == EditKind::Fill && center.distance(self.camera.position) < r + 0.5 {
            return;
        }
        let dirty = self.world.edits.push(EditOp { center, radius: r, kind, material: Material::Rock });
        self.streamer.invalidate(&dirty);
        self.last_edit = Instant::now();
    }

    fn hud_info(&self) -> HudInfo {
        HudInfo {
            visible: self.show_hud,
            captured: self.captured,
            frame_ms: self.frame_ms,
            position: self.camera.position,
            frame: self.last_stats,
            stream: self.streamer.stats(),
            gpu_chunks: self.renderer.chunk_count(),
            ticks: self.sim.ticks,
            time_of_day: self.sim.time_of_day,
            brush_radius: self.brush_radius,
            fly_speed: self.fly_speed,
            view_radius: self.streamer.view_radius,
            vsync: self.vsync,
            edits: self.world.edits.len(),
            gpu_name: self.renderer.gpu().adapter_info.name.clone(),
            hot_reload: self.renderer.hot_reload_available(),
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("Ascension")
            .with_inner_size(winit::dpi::LogicalSize::new(1280.0, 720.0))
            .with_maximized(true);
        let result = event_loop
            .create_window(attrs)
            .map_err(anyhow::Error::from)
            .and_then(|window| State::new(Arc::new(window), self.options.clone()));
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
        let egui_consumed =
            !state.captured && state.egui_state.on_window_event(&state.window, &event).consumed;
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.renderer.resize(size.width, size.height),
            WindowEvent::RedrawRequested => state.frame(event_loop),
            WindowEvent::Focused(false) => {
                state.set_captured(false);
                state.input.clear();
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    state.input.key(code, event.state, event.repeat);
                }
            }
            WindowEvent::MouseInput { state: button_state, button, .. } => {
                if !state.captured && button_state == ElementState::Pressed && !egui_consumed {
                    // The click that captures the mouse shouldn't also dig.
                    state.set_captured(true);
                } else if state.captured {
                    state.input.button(button, button_state);
                }
            }
            WindowEvent::MouseWheel { delta, .. } if !egui_consumed => state.input.scroll(delta),
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if let (Some(state), DeviceEvent::MouseMotion { delta }) = (self.state.as_mut(), event)
            && state.captured
        {
            state.input.mouse_delta += Vec2::new(delta.0 as f32, delta.1 as f32);
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

/// Feet position standing on the ground in column (x, z).
fn find_ground(world: &World, x: f32, z: f32) -> Vec3 {
    let top = Vec3::new(x, 400.0, z);
    match raycast(|p| world.density_at(p), top, -Vec3::Y, 800.0) {
        Some(hit) => hit.point + Vec3::Y * 0.05,
        None => Vec3::new(x, world.worldgen.height(x, z) + 2.0, z),
    }
}
