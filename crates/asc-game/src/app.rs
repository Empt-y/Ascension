//! Window, event loop and the frame/tick loop.

use std::sync::Arc;
use std::time::{Duration, Instant};

use asc_render::{Camera, FrameParams, FrameStats, Frustum, Renderer, UiFrame};
use asc_sim::inventory::{InventoryOp, InventorySnapshot};
use asc_sim::protocol::{ClientMsg, ServerMsg};
use asc_sim::rules::find_ground;
use asc_sim::link::LocalLink;
use asc_sim::server::{self as srv, Server};
use asc_world::{EditKind, EditOp, Material, RayHit, World, raycast};
use glam::{Vec2, Vec3};
use winit::application::ApplicationHandler;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Window, WindowId};

use crate::assets::PartModels;
use crate::building::{BUILD_REACH, Building, Selection};
use crate::content::Content;
use crate::far::FarStreamer;
use crate::hud::{self, HudInfo};
use crate::input::Input;
use crate::inventory::{HOTBAR, Inventory, Stack};
use crate::inventory_ui;
use crate::inventory_ui::MenuChoice;
use crate::mining::{BITE_RADIUS, Mining};
use crate::player::{self, Mode, MoveInput, Player};
use crate::save;
use crate::sim::Sim;
use crate::streaming::Streamer;


/// The build tool in hand.
#[derive(Clone, Copy, Debug, PartialEq)]
enum BuildTool {
    Piece(Selection),
    Flatten,
}
const FLY_REACH: f32 = 64.0;
const WALK_REACH: f32 = 5.0;
/// How close the crosshair must be to an ore node's collider to mine it.
const NODE_TOLERANCE: f32 = 0.15;
const EDIT_REPEAT: Duration = Duration::from_millis(110);

#[derive(Clone, Debug, Default)]
pub struct Options {
    /// Fall from 30 m (dying), respawn, then walk back to the stash. For unattended checks.
    pub fall_test: bool,
    /// Dig a trench from the nearest shore inland below sea level and watch the sea fill it.
    pub fluid_test: bool,
    /// Start the nearest active volcano's eruption (doing this), and geysers, soon after starting.
    pub erupt: Option<asc_sim::volcanism::Kind>,
    /// Render the scene at this fraction of the window's resolution and upscale it (`--render-scale`).
    pub render_scale: Option<f32>,
    /// Run a stand-in factory on the server every tick (light: mostly waiting machines; heavy:
    /// enough work to overrun the tick).
    pub tick_stress: Option<asc_sim::stress::StressConfig>,
    /// Fly a fixed route, dig once, log stats and quit. For unattended performance checks.
    pub autopilot: bool,
    /// Walk and jump across the terrain, logging collision clearance. For unattended checks.
    pub walk_test: bool,
    /// Mine straight down for a while, then place some of it back, logging the inventory.
    pub mine_test: bool,
    pub time_of_day: Option<f32>,
    pub exit_after: Option<f32>,
    pub view_radius: Option<i32>,
    pub vsync: bool,
    pub hide_hud: bool,
    /// Start pose: x, y, z, yaw, pitch (radians).
    pub camera: Option<[f32; 5]>,
    /// Start on foot near the closest active volcano, facing it (off to the side of its breach).
    pub near_volcano: bool,
    /// World to load or create, under `saves_dir`.
    pub world: String,
    pub saves_dir: Option<std::path::PathBuf>,
    /// Start this world over (the old one is kept, renamed).
    pub new_world: bool,
    /// Seed for a newly created world.
    pub seed: Option<i32>,
    /// Start with the inventory screen open (for screenshots).
    pub show_inventory: bool,
    /// Draw the far-terrain clipmap (on by default).
    pub far: bool,
    pub fog_density: Option<f32>,
    /// Scene MSAA samples (1 or 4).
    pub msaa: Option<u32>,
    /// Lay out every building part near spawn (for screenshots).
    pub showcase: bool,
    /// Grow trees, grass, rocks... over the world (on by default).
    pub scatter: bool,
    /// Sun shadows (on by default).
    pub shadows: bool,
    /// Temporal anti-aliasing (on by default).
    pub taa: bool,
    /// Accessibility: no wind motion in plants.
    pub reduce_motion: bool,
    /// Turn the camera slowly in place at this many radians per second (for testing aliasing:
    /// a fraction of a pixel per frame, so anything that flickers between frames is aliasing).
    pub orbit: Option<f32>,
    /// Flatten a site, build a small house on it, walk inside, and log what happens.
    pub build_test: bool,
    /// Place a large scripted block of pieces near spawn (a building performance test).
    pub build_stress: bool,
    /// No sound.
    pub mute: bool,
    /// How loud everything is, 0..1 (1 when not given).
    pub volume: Option<f32>,
}

impl Options {
    pub fn from_args() -> anyhow::Result<Self> {
        let mut opts = Self {
            world: "world1".into(),
            far: true,
            scatter: true,
            shadows: true,
            taa: true,
            ..Self::default()
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            let mut value = |name: &str| args.next().ok_or_else(|| anyhow::anyhow!("{name} needs a value"));
            match arg.as_str() {
                "--autopilot" => opts.autopilot = true,
                "--walk-test" => opts.walk_test = true,
                "--mine-test" => opts.mine_test = true,
                "--new-world" => opts.new_world = true,
                "--show-inventory" => opts.show_inventory = true,
                "--far" => {
                    opts.far = match value("--far")?.as_str() {
                        "on" => true,
                        "off" => false,
                        other => anyhow::bail!("--far wants on or off, not {other}"),
                    }
                }
                "--fog" => opts.fog_density = Some(value("--fog")?.parse()?),
                "--msaa" => opts.msaa = Some(value("--msaa")?.parse()?),
                "--showcase" => opts.showcase = true,
                "--reduce-motion" => opts.reduce_motion = true,
                "--taa" => {
                    opts.taa = match value("--taa")?.as_str() {
                        "on" => true,
                        "off" => false,
                        other => anyhow::bail!("--taa wants on or off, not {other}"),
                    }
                }
                "--orbit" => opts.orbit = Some(value("--orbit")?.parse()?),
                "--render-scale" => {
                    let v = value("--render-scale")?;
                    let preset = asc_render::RENDER_SCALES.iter().find(|(_, name)| *name == v.as_str());
                    opts.render_scale = Some(match preset {
                        Some((scale, _)) => *scale,
                        None => v.parse().map_err(|_| {
                            anyhow::anyhow!("--render-scale wants native, quality, balanced, performance or 0.25-1")
                        })?,
                    });
                }
                "--shadows" => {
                    opts.shadows = match value("--shadows")?.as_str() {
                        "on" => true,
                        "off" => false,
                        other => anyhow::bail!("--shadows wants on or off, not {other}"),
                    }
                }
                "--scatter" => {
                    opts.scatter = match value("--scatter")?.as_str() {
                        "on" => true,
                        "off" => false,
                        other => anyhow::bail!("--scatter wants on or off, not {other}"),
                    }
                }
                "--build-test" => opts.build_test = true,
                "--build-stress" => opts.build_stress = true,
                "--world" => opts.world = value("--world")?,
                "--saves-dir" => opts.saves_dir = Some(value("--saves-dir")?.into()),
                "--seed" => opts.seed = Some(value("--seed")?.parse()?),
                "--vsync" => opts.vsync = true,
                "--hide-hud" => opts.hide_hud = true,
                "--time" => opts.time_of_day = Some(value("--time")?.parse()?),
                "--exit-after" => opts.exit_after = Some(value("--exit-after")?.parse()?),
                "--fall-test" => opts.fall_test = true,
                "--fluid-test" => opts.fluid_test = true,
                "--mute" => opts.mute = true,
                "--near-volcano" => opts.near_volcano = true,
                "--volume" => opts.volume = Some(value("--volume")?.parse()?),
                "--erupt" => opts.erupt = Some(asc_sim::volcanism::Kind::Natural),
                "--erupt-flow" => opts.erupt = Some(asc_sim::volcanism::Kind::Flows),
                "--erupt-blast" => opts.erupt = Some(asc_sim::volcanism::Kind::Blast),
                "--tick-stress" => {
                    opts.tick_stress = Some(match value("--tick-stress")?.as_str() {
                        "light" => asc_sim::stress::StressConfig::LIGHT,
                        "heavy" => asc_sim::stress::StressConfig::HEAVY,
                        other => anyhow::bail!("--tick-stress wants light or heavy, not {other}"),
                    });
                }
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
    /// The game itself, running on its own thread. The fields below that mirror it (`world`,
    /// `building.structures`, the scatter's mined nodes, `inventory`, `sim`) only change through
    /// its messages.
    server: LocalLink,
    /// The server's last ticks per second and slowest tick.
    server_stats: (f32, f32),
    ping: u32,
    /// The world as the server has it (seed plus edits), for meshing, collision and raycasts.
    world: World,
    streamer: Streamer,
    far: FarStreamer,
    sim: Sim,
    player: Player,
    spawn: Vec3,
    content: Content,
    parts: PartModels,
    nature: crate::assets::NatureModels,
    scatter: crate::scatter::Scatter,
    /// Edit ops and building pieces the scatter has already been re-filtered for.
    scatter_seen: (usize, usize),
    building: Building,
    build_tool: Option<BuildTool>,
    previews: Option<Vec<[egui::TextureHandle; 3]>>,
    /// Flatten tool preview: cell centre and target height.
    flatten_preview: Option<(glam::Vec2, f32)>,
    build_test_phase: u8,
    build_test_floor: f32,
    inventory: Inventory,
    mining: Mining,
    held: Option<Stack>,
    /// Inventory ops sent but not yet in a server snapshot, re-applied on top of each snapshot.
    pending_ops: Vec<(u32, InventoryOp)>,
    inventory_seq: u32,
    /// The player's health, air and temperature, as the server has them.
    vitals: asc_sim::vitals::Vitals,
    /// The red edge after being hurt, fading from 1.
    hurt_flash: f32,
    fall_test_stage: u8,
    fluid_test_stage: u8,
    fluid_test_end: Vec3,
    /// What the server last said about volcanoes near by.
    volcanism: asc_sim::volcanism::Report,
    erupt_sent: bool,
    /// Smoke, lava spray and glowing bombs from the volcanoes near by, and when they last moved.
    effects: crate::effects::Effects,
    effects_at: std::time::Instant,
    volcanic: crate::assets::VolcanicModels,
    /// Sound (`None` without an audio device, or with `--mute`), the weather it hears, and what
    /// it tracks between frames: distance walked since the last footstep, whether the player was
    /// on the ground and how fast they were falling, how deep in water.
    audio: Option<crate::audio::Audio>,
    weather: asc_world::weather::Weather,
    stride: f32,
    /// Arrived in flight (`--near-volcano`): walk once the server has the new position.
    walk_soon: bool,
    was_grounded: bool,
    falling: f32,
    was_submerged: f32,
    /// Where things left at death lie.
    stashes: std::collections::BTreeMap<u32, Vec3>,
    /// The simulated water, as the server sends it.
    fluids: crate::fluids::ClientFluids,
    inventory_open: bool,
    notice: Option<(String, Instant)>,
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
        let parts = PartModels::load(&mut renderer)?;
        let save_dir = options.saves_dir.clone().unwrap_or_else(save::default_saves_dir).join(&options.world);
        // Scripted tests and tools build and dig from wherever they stand.
        let validate = !(options.build_test
            || options.build_stress
            || options.autopilot
            || options.fall_test
            || options.fluid_test);
        let config = srv::Config {
            save_dir,
            seed: options.seed,
            new_world: options.new_world,
            validate,
            // Single player: the player is the host.
            cheats: true,
            stress: options.tick_stress,
        };
        let mut server = Server::open(config, asc_build::Lattice::new(&parts.spec))?;
        if let Some(t) = options.time_of_day {
            server.handle(ClientMsg::SetTime(t));
        }
        server.drain().for_each(drop);
        let welcome = server.welcome();
        let content = Content::load()?;
        let seed = welcome.seed;
        let mut world = World::new(seed);
        world.edits = asc_world::EditLog::from_ops(welcome.edits.iter().copied());
        let streamer = Streamer::new(world.worldgen.clone(), options.view_radius.unwrap_or(10));
        let far = FarStreamer::new(world.worldgen.clone(), &welcome.edits);
        let nature = crate::assets::NatureModels::load(&mut renderer)?;
        let ores = crate::assets::OreModels::load(&mut renderer)?;
        let volcanic = crate::assets::VolcanicModels::load(&mut renderer)?;
        let mut scatter = crate::scatter::Scatter::new(
            crate::scatter::Config::new(seed, &nature, Some(&ores)),
            world.worldgen.clone(),
        );
        scatter.set_removed_nodes(welcome.mined.iter().copied());
        scatter.set_fire(&welcome.fire);
        for scar in &welcome.scars {
            scatter.add_scar(*scar);
        }
        let lattice = asc_build::Lattice::new(&parts.spec);
        let structures = asc_build::Structures::from_pieces(lattice, welcome.pieces.iter().copied());
        log::info!("structures: {} pieces", structures.len());
        let building = Building::new(structures);
        renderer.set_far_enabled(options.far);
        if let Some(samples) = options.msaa {
            renderer.set_msaa(samples)?;
        }
        renderer.set_shadows(options.shadows);
        renderer.set_taa(options.taa);
        if let Some(scale) = options.render_scale {
            renderer.set_render_scale(scale);
        }
        renderer.set_wind(if options.reduce_motion { 0.0 } else { 1.0 });
        if let Some(d) = options.fog_density {
            renderer.fog_density = d;
        }
        let sim = Sim { ticks: welcome.ticks, time_of_day: welcome.time_of_day };
        let spawn = server.spawn;
        let server = asc_sim::link::spawn(server);
        let mut player = Player::new(welcome.feet);
        player.mode = welcome.mode;
        (player.yaw, player.pitch) = (welcome.yaw, welcome.pitch);
        let mut inventory = Inventory::default();
        let held = inventory.restore(&welcome.inventory);
        let vitals = welcome.vitals;
        let stashes: std::collections::BTreeMap<u32, Vec3> = welcome.stashes.iter().copied().collect();
        let mut fluids = crate::fluids::ClientFluids::new();
        for (coord, packed) in &welcome.fluids {
            fluids.apply(*coord, packed);
        }
        if options.autopilot {
            player.mode = Mode::Fly;
            player.teleport(spawn + Vec3::Y * 25.0);
            player.pitch = -0.3;
        }
        let mut walk_soon = false;
        if options.near_volcano {
            let here = Vec2::new(player.feet.x, player.feet.z);
            let active = world
                .worldgen
                .volcanoes_near(here, 20_000.0)
                .into_iter()
                .filter(|v| v.kind == asc_world::volcano::VolcanoKind::Active)
                .min_by(|a, b| a.center.distance(here).total_cmp(&b.center.distance(here)));
            match active {
                Some(v) => {
                    // Off to the side of the breach (where blasts and its lava go), outside the cone.
                    let side = if v.tilt == Vec2::ZERO { Vec2::Y } else { v.tilt.perp() };
                    let at = v.center + side * (v.radius * 1.6).max(900.0);
                    let ground = find_ground(&world, at.x, at.y);
                    // There in flight (the host may: a walk that far would be refused as too fast),
                    // walking from the next frame.
                    player.mode = Mode::Fly;
                    player.teleport(ground);
                    let to = v.center - at;
                    (player.yaw, player.pitch) = (to.y.atan2(to.x), 0.05);
                    walk_soon = true;
                    log::info!("near the volcano at {}: standing {:.0} m from its middle", v.center, to.length());
                }
                None => log::warn!("--near-volcano: no active volcano within 20 km"),
            }
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
            server,
            server_stats: (0.0, 0.0),
            ping: 0,
            world,
            streamer,
            far,
            sim,
            player,
            spawn,
            content,
            parts,
            nature,
            scatter,
            scatter_seen: (0, 0),
            building,
            build_tool: None,
            previews: None,
            flatten_preview: None,
            build_test_phase: 0,
            build_test_floor: 0.0,
            inventory,
            mining: Mining::default(),
            held,
            vitals,
            hurt_flash: 0.0,
            fall_test_stage: 0,
            fluid_test_stage: 0,
            fluid_test_end: Vec3::ZERO,
            volcanism: Default::default(),
            erupt_sent: false,
            effects: {
                let mut effects = crate::effects::Effects::default();
                effects.set_fire(&welcome.fire);
                effects
            },
            effects_at: std::time::Instant::now(),
            volcanic,
            audio: if options.mute {
                None
            } else {
                match crate::assets::root().and_then(|r| crate::audio::Audio::new(&r.join("sounds"), &r.join("sounds_library"))) {
                    Ok(mut a) => {
                        a.set_volume(options.volume.unwrap_or(1.0));
                        Some(a)
                    }
                    Err(err) => {
                        log::warn!("no sound: {err:#}");
                        None
                    }
                }
            },
            weather: asc_world::weather::Weather::new(seed),
            stride: 0.0,
            walk_soon,
            was_grounded: true,
            falling: 0.0,
            was_submerged: 0.0,
            fluids,
            stashes,
            pending_ops: Vec::new(),
            inventory_seq: welcome.inventory.seq,
            inventory_open: options.show_inventory,
            notice: None,
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

        self.receive();

        self.handle_keys();
        if self.options.autopilot {
            self.autopilot(dt.as_secs_f32());
        } else {
            self.move_player(dt.as_secs_f32());
        }
        if let Some(rate) = self.options.orbit {
            self.player.yaw += rate * dt.as_secs_f32();
        }
        self.body_sounds(dt.as_secs_f32());
        let p = &self.player;
        self.send(ClientMsg::Move { feet: p.feet, yaw: p.yaw, pitch: p.pitch, mode: p.mode });
        if self.walk_soon {
            self.walk_soon = false;
            self.player.mode = Mode::Walk;
        }
        self.camera = Camera {
            position: self.player.eye(),
            yaw: self.player.yaw,
            pitch: self.player.pitch,
            ..self.camera
        };

        let reach = if self.build_tool.is_some() {
            BUILD_REACH
        } else if self.player.mode == Mode::Fly {
            FLY_REACH
        } else {
            WALK_REACH
        };
        let (world, building, scatter) = (&self.world, &self.building, &self.scatter);
        self.hit = raycast(
            |p| world.density_at(p).min(building.sdf(p)).min(scatter.sdf(p)),
            self.camera.position,
            self.camera.forward(),
            reach,
        );
        if self.options.build_test {
            self.build_test_step();
        }
        if self.options.fall_test {
            self.fall_test_step();
        }
        if self.options.fluid_test {
            self.fluid_test_step();
        }
        if let Some(kind) = self.options.erupt
            && !self.erupt_sent
            && self.started.elapsed().as_secs_f32() > 2.0
        {
            self.erupt_sent = true;
            self.send(ClientMsg::Erupt(kind));
        }
        if self.build_tool.is_some() {
            self.mining.reset();
            self.use_build_tool();
        } else {
            if self.captured && self.player.mode == Mode::Fly {
                self.edit_terrain();
            }
            if self.player.mode == Mode::Walk && !self.inventory_open && !self.building.menu_open {
                self.mine_and_place(dt.as_secs_f32());
            } else {
                self.mining.reset();
            }
        }
        if !self.options.showcase {
            self.building.sync(&mut self.renderer, &self.parts);
            if let (Some(BuildTool::Flatten), Some((xz, h))) = (self.build_tool, self.flatten_preview) {
                // Show the levelled top as a foundation-shaped ghost.
                let model = self.parts.model(0, asc_build::PartKind::Foundation);
                let position = Vec3::new(xz.x, h - self.building.structures.lattice.foundation_height, xz.y);
                self.renderer.set_ghost(Some(asc_render::Ghost {
                    instance: asc_render::PartInstance::rigid(model, position, 0),
                    valid: true,
                }));
            }
        }

        let (w, h) = self.renderer.gpu().size();
        let frustum = Frustum::from_view_proj(self.camera.view_proj(w as f32 / h as f32));
        self.streamer.update(&self.world, &self.fluids, self.camera.position, &frustum, &mut self.renderer);
        self.fluids.update(&self.world, &mut self.renderer);
        if self.renderer.far_enabled() {
            let p = self.camera.position;
            self.far.update(glam::Vec2::new(p.x, p.z), &mut self.renderer);
        }
        if self.options.scatter {
            // Re-filter scatter where the ground was edited or something was built or dismantled.
            let ops = self.world.edits.ops();
            for op in &ops[self.scatter_seen.0.min(ops.len())..] {
                self.scatter.invalidate(op.center, op.reach() + 1.0);
            }
            let pieces = self.building.structures.len();
            if pieces != self.scatter_seen.1 {
                self.scatter.invalidate(self.camera.position, BUILD_REACH + 8.0);
            }
            self.scatter_seen = (ops.len(), pieces);
            self.scatter.update(
                self.camera.position,
                &self.world,
                &self.building.structures,
                &mut self.renderer,
            );
        }

        let raw_input = self.egui_state.take_egui_input(&self.window);
        let info = self.hud_info();
        if self.notice.as_ref().is_some_and(|(_, t)| t.elapsed() > Duration::from_secs(2)) {
            self.notice = None;
        }
        let progress = self.mining.progress(&self.content);
        let walking = self.player.mode == Mode::Walk;
        if self.building.menu_open && self.previews.is_none() {
            match crate::assets::load_previews(&self.egui_ctx, &self.parts.spec) {
                Ok(p) => self.previews = Some(p),
                Err(err) => log::warn!("build menu previews: {err:#}"),
            }
        }
        let menu = self.building.menu_open.then_some(()).and(self.previews.as_ref());
        let spec = &self.parts.spec;
        let status = self.build_status();
        let mut picked = None;
        let (content, inventory, held) = (&self.content, &self.inventory, self.held);
        self.hurt_flash = (self.hurt_flash - dt.as_secs_f32() * 1.5).max(0.0);
        let (vitals, hurt) = (self.vitals, self.hurt_flash);
        let mut clicked = None;
        let (open, captured, notice) = (self.inventory_open, self.captured, &self.notice);
        let output = self.egui_ctx.run_ui(raw_input, |ui| {
            let ctx = ui.ctx();
            hud::draw(ctx, &info);
            if captured {
                inventory_ui::crosshair(ctx, progress);
            }
            inventory_ui::hurt_overlay(ctx, hurt);
            if walking {
                inventory_ui::vitals(ctx, &vitals);
                inventory_ui::hotbar(ctx, content, inventory);
            }
            if let Some((text, _)) = notice {
                inventory_ui::notice(ctx, text);
            }
            if open {
                clicked = inventory_ui::window(ctx, content, inventory, held);
            }
            if let Some(previews) = menu {
                picked = inventory_ui::build_menu(ctx, spec, previews);
            }
            if let Some((text, ok)) = &status {
                inventory_ui::build_status(ctx, text, *ok);
            }
        });
        if let Some(choice) = picked {
            self.choose_build_tool(choice);
        }
        if let Some(op) = clicked {
            self.inventory_op(op);
        }
        self.egui_state.handle_platform_output(&self.window, output.platform_output);
        let primitives = self.egui_ctx.tessellate(output.shapes, output.pixels_per_point);

        let now = std::time::Instant::now();
        let dt = now.duration_since(self.effects_at).as_secs_f32().min(0.25);
        self.effects_at = now;
        self.effects.update(dt, &self.volcanism, &self.world.worldgen, self.camera.position);
        let bombs = self.effects.bombs(&self.volcanism, &self.volcanic.bombs, self.started.elapsed().as_secs_f32());
        self.building.set_moving(bombs);
        self.listen();
        let (ash, shake) = self.volcanic_air();
        let params = FrameParams {
            camera: Camera { position: self.camera.position + shake, ..self.camera },
            ash,
            particles: self.effects.particles(&self.volcanism),
            sun_dir: self.sim.sun_dir(),
            view_distance: self.streamer.view_radius as f32 * 32.0,
            underwater: self.fluids.fraction_at(&self.world, self.camera.position) > 0.5,
            lights: {
                let mut lights = self.fluids.lava_lights(self.camera.position);
                lights.extend(self.streamer.lava_lights(&self.fluids, self.camera.position, 160.0));
                // A blast's flash, fading as its shockwave spreads.
                lights.extend(self.volcanism.blasts.iter().filter(|b| b.radius < 1200.0).map(|b| asc_render::PointLight {
                    position: b.center + Vec3::Y * 200.0,
                    radius: 600.0 + b.radius,
                    color: Vec3::new(1.0, 0.6, 0.3) * 40.0 * (1.0 - b.radius / 1200.0),
                }));
                // Burning ground, near by.
                lights.extend(self.effects.fire_lights(&self.world.worldgen, self.camera.position, 6));
                // Volcanic bombs glow as they fly.
                lights.extend(self.volcanism.bombs.iter().map(|&(position, _)| asc_render::PointLight {
                    position,
                    radius: 30.0,
                    color: Vec3::new(1.0, 0.4, 0.1) * 6.0,
                }));
                lights
            },
            brush: self
                .hit
                .filter(|_| self.captured || self.options.autopilot || self.options.mine_test)
                .map(|h| {
                    let radius = if self.player.mode == Mode::Walk { BITE_RADIUS } else { self.brush_radius };
                    (h.point, radius)
                }),
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
            let fs = self.far.stats();
            let sc = self.scatter.stats();
            if sc.batches > 0 || sc.in_flight > 0 {
                log::info!(
                    "scatter: {} batches, {}k instances, {} in flight | {} drawn",
                    sc.batches,
                    sc.instances / 1000,
                    sc.in_flight,
                    self.last_stats.parts_drawn,
                );
            }
            if !self.building.structures.is_empty() {
                log::info!(
                    "parts: {} placed, {} instances drawn",
                    self.building.structures.len(),
                    self.last_stats.parts_drawn,
                );
            }
            log::info!(
                "far: {} levels drawn, {}k tris | worker {} jobs in flight, {:.0}k samples/s, {}k total",
                self.last_stats.far_levels,
                self.last_stats.far_triangles / 1000,
                fs.in_flight,
                fs.samples_per_sec / 1000.0,
                fs.samples / 1000,
            );
            if self.options.mine_test {
                log::info!(
                    "mine test: edits {} | inventory {}",
                    self.world.edits.len(),
                    self.inventory_summary()
                );
            }
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
            log::info!("server: {:.1} tps, slowest tick {:.2} ms", self.server_stats.0, self.server_stats.1);
            if let Some([fireflies, hot, bad, frames]) = self.renderer.taa_debug_counts() {
                log::info!("taa debug: fireflies {fireflies}, sun-bright inputs {hot}, nan/inf {bad} (pixels over {frames} frames)");
                if let Some(s) = self.renderer.taa_debug_sources() {
                    log::info!("taa debug sources: terrain {} foliage {} models {} far {} sky {} water {} other {}", s[1], s[2], s[3], s[4], s[5], s[6], s[0] + s[7]);
                }
            }
            if let Some(stages) = self.renderer.gpu_stages() {
                log::info!("gpu ms: {stages} | parts drawn {}", self.last_stats.parts_drawn);
            }
        }
        if let Some(limit) = self.options.exit_after
            && self.started.elapsed().as_secs_f32() >= limit
        {
            event_loop.exit();
        }
        self.input.end_frame();
    }

    fn handle_keys(&mut self) {
        let input = &self.input;
        if input.pressed(KeyCode::Escape) {
            if self.building.menu_open {
                self.toggle_build_menu();
            } else if self.build_tool.is_some() {
                self.set_build_tool(None);
            } else if self.inventory_open {
                self.toggle_inventory();
            } else if self.captured {
                self.set_captured(false);
            }
            // (Esc never quits: a screenshot tool's Esc reached the game after it had released the
            // mouse and closed it. Close the window to quit.)
        }
        if self.input.pressed(KeyCode::KeyE) && self.player.mode == Mode::Walk && !self.building.menu_open {
            self.toggle_inventory();
        }
        if self.input.pressed(KeyCode::KeyB) && !self.inventory_open {
            self.toggle_build_menu();
        }
        const DIGITS: [KeyCode; HOTBAR] = [
            KeyCode::Digit1,
            KeyCode::Digit2,
            KeyCode::Digit3,
            KeyCode::Digit4,
            KeyCode::Digit5,
            KeyCode::Digit6,
            KeyCode::Digit7,
            KeyCode::Digit8,
            KeyCode::Digit9,
        ];
        if let Some(i) = DIGITS.iter().position(|k| self.input.pressed(*k)) {
            self.inventory_op(InventoryOp::Select(i));
        }
        if self.input.pressed(KeyCode::KeyF) {
            self.player.mode = match self.player.mode {
                Mode::Walk => Mode::Fly,
                Mode::Fly => Mode::Walk,
            };
            self.player.velocity = Vec3::ZERO;
        }
        if self.input.pressed(KeyCode::F8)
            && let Some(audio) = &mut self.audio
        {
            if self.input.held(KeyCode::ShiftLeft) || self.input.held(KeyCode::ShiftRight) {
                let set = audio.next_step_set();
                self.show_notice(format!("Footsteps: {set}"));
            } else {
                let library = audio.toggle_library();
                self.show_notice(if library { "Sounds: library recordings".into() } else { "Sounds: generated".into() });
            }
        }
        if self.input.pressed(KeyCode::F9) {
            self.send(ClientMsg::Save);
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
        if self.input.pressed(KeyCode::F7) {
            // Cycle the upscaling presets.
            let now = self.renderer.render_scale();
            let presets = asc_render::RENDER_SCALES;
            let i = presets.iter().position(|(s, _)| (*s - now).abs() < 1e-3).map_or(0, |i| (i + 1) % presets.len());
            self.renderer.set_render_scale(presets[i].0);
            let (w, h) = self.renderer.render_size();
            self.show_notice(format!("Render scale: {} ({}%, {w}x{h})", presets[i].1, (presets[i].0 * 100.0).round()));
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
            let t = (self.sim.time_of_day + step * self.frame_ms / 1000.0 / 24.0).rem_euclid(1.0);
            self.send(ClientMsg::SetTime(t));
        }
        let wheel = self.input.wheel;
        if wheel != 0.0 && self.build_tool.is_some() {
            if let Some(BuildTool::Piece(sel)) = self.build_tool {
                let tiers = self.parts.spec.tiers.len();
                let tier = if wheel > 0.0 { (sel.tier + tiers - 1) % tiers } else { (sel.tier + 1) % tiers };
                self.set_build_tool(Some(BuildTool::Piece(Selection { tier, ..sel })));
            }
        } else if wheel != 0.0 && self.player.mode == Mode::Walk && !self.input.held(KeyCode::AltLeft) {
            // Wheel down moves right along the hotbar, as in most games.
            let step = if wheel > 0.0 { HOTBAR - 1 } else { 1 };
            self.inventory_op(InventoryOp::Select((self.inventory.selected + step) % HOTBAR));
        } else if wheel != 0.0 {
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
        if self.options.build_test && self.build_test_phase == 2 {
            // Walk straight in through the doorway, towards the back wall.
            input = MoveInput { forward: 1.0, ..Default::default() };
        }
        // How deep in water the body is: sampled at the knees, waist and chest.
        let feet = self.player.feet;
        self.player.submerged = [0.4, 0.9, 1.4]
            .iter()
            .map(|h| self.fluids.fraction_at(&self.world, feet + Vec3::Y * h).min(1.0))
            .sum::<f32>()
            / 3.0;
        let t = Instant::now();
        let world = &self.world;
        let building = &self.building;
        let scatter = &self.scatter;
        let steps =
            self.player.update(dt, &input, &|p| world.density_at(p).min(building.sdf(p)).min(scatter.sdf(p)));
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

    fn send(&mut self, msg: ClientMsg) {
        self.server.send(msg);
    }

    /// Apply the server's messages that have arrived.
    fn receive(&mut self) {
        while let Some(msg) = self.server.try_recv() {
            self.apply(msg);
        }
    }

    /// Wait until the server has handled everything sent so far, applying what it sends back
    /// (scripted tests that act on the results of their own requests).
    fn sync(&mut self) {
        self.ping += 1;
        let n = self.ping;
        self.send(ClientMsg::Ping(n));
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            match self.server.recv_timeout(Duration::from_millis(100)) {
                Some(ServerMsg::Pong(m)) if m == n => return,
                Some(msg) => self.apply(msg),
                None => {}
            }
        }
        log::error!("the server didn't answer within 10 s");
    }

    /// Ask the server to save, and wait for it to finish.
    fn save_and_wait(&mut self) {
        self.send(ClientMsg::Save);
        let deadline = Instant::now() + Duration::from_secs(10);
        while Instant::now() < deadline {
            match self.server.recv_timeout(Duration::from_millis(100)) {
                Some(ServerMsg::Saved { .. }) => return,
                Some(msg) => self.apply(msg),
                None => {}
            }
        }
        log::error!("the server didn't finish saving within 10 s");
    }

    /// Apply one server message to the client's copies of the game.
    fn apply(&mut self, msg: ServerMsg) {
        match msg {
            ServerMsg::Welcome(_) => {}
            ServerMsg::Edited(op) => {
                self.edit_sound(&op);
                let dirty = self.world.edits.push(op);
                self.streamer.invalidate(&dirty);
                self.fluids.edited(&dirty);
                self.far.edited(&op);
            }
            ServerMsg::PiecePlaced(piece) => self.building.placed(piece),
            ServerMsg::PieceRemoved(key) => self.building.removed(&key),
            ServerMsg::NodeMined(id) => self.scatter.remove_node(id),
            ServerMsg::Inventory(snapshot) => self.inventory_snapshot(&snapshot),
            ServerMsg::Time { ticks, time_of_day } => {
                self.sim.ticks = ticks;
                self.sim.time_of_day = time_of_day;
            }
            ServerMsg::Notice(text) => self.show_notice(text),
            ServerMsg::Saved { .. } | ServerMsg::Pong(_) => {}
            ServerMsg::Stats { tps, mspt } => self.server_stats = (tps, mspt),
            ServerMsg::Vitals(v) => self.vitals = v,
            ServerMsg::Hurt { amount, .. } => {
                self.hurt_flash = (self.hurt_flash + amount as f32 / 200.0).min(1.0);
                if let Some(audio) = &mut self.audio {
                    audio.play("hurt", (0.4 + amount as f32 / 300.0).min(1.0));
                }
            }
            ServerMsg::Died { cause } => {
                self.hurt_flash = 1.0;
                if let Some(audio) = &mut self.audio {
                    audio.play("hurt", 1.0);
                }
                self.show_notice(format!("You {}. Your things are where you fell.", cause.describe()));
                self.player.velocity = Vec3::ZERO;
            }
            ServerMsg::StashPlaced { id, at } => {
                self.stashes.insert(id, at);
                self.sync_stashes();
            }
            ServerMsg::Fluid { coord, data } => self.fluids.apply(coord, &data),
            ServerMsg::Volcanism(report) => {
                self.volcanic_sounds(&report);
                self.volcanism = report;
            }
            ServerMsg::Scar(scar) => self.scatter.add_scar(scar),
            ServerMsg::Fire(changed) => {
                self.scatter.set_fire(&changed);
                self.effects.set_fire(&changed);
            }
            ServerMsg::StashTaken { id } => {
                self.stashes.remove(&id);
                self.sync_stashes();
            }
            ServerMsg::Correct { feet, mode } => {
                log::warn!("server put the player back at {feet}");
                self.player.teleport(feet);
                self.player.mode = mode;
            }
        }
    }

    /// Fall test: at 3 s jump 30 m up (and die on landing); at 9 s go back to where the stash was
    /// left, which should pick it up; log each stage.
    fn fall_test_step(&mut self) {
        let t = self.started.elapsed().as_secs_f32();
        if t > 3.0 && self.fall_test_stage == 0 {
            self.fall_test_stage = 1;
            self.player.mode = Mode::Walk;
            let up = self.player.feet + Vec3::Y * 30.0;
            self.player.teleport(up);
            log::info!("fall test: dropped from {up:.1} with {} health", self.vitals.health);
        }
        if t > 9.0 && self.fall_test_stage == 1 {
            self.fall_test_stage = 2;
            log::info!("fall test: at {:.1}, {} health, stashes {:?}", self.player.feet, self.vitals.health, self.stashes);
            if let Some(at) = self.stashes.values().next() {
                self.player.teleport(*at + Vec3::Y * 0.2);
            }
        }
        if t > 12.0 && self.fall_test_stage == 2 {
            self.fall_test_stage = 3;
            log::info!("fall test: stashes left {:?} | inventory {}", self.stashes, self.inventory_summary());
        }
    }

    /// Fluid test: find a shore near spawn, dig a trench from the sea 16 m inland below sea level,
    /// hover above it, and log how full its far end is as the sea flows in.
    fn fluid_test_step(&mut self) {
        let t = self.started.elapsed().as_secs_f32();
        let sea = asc_world::SEA_LEVEL;
        if self.fluid_test_stage == 0 && t > 2.0 {
            self.fluid_test_stage = 1;
            let wg = self.world.worldgen.clone();
            let shore = (0..40000)
                .map(|i| ((i % 200) as f32 * 4.0 - 400.0, (i / 200) as f32 * 4.0 - 400.0))
                .find(|&(x, z)| wg.height(x, z) < sea - 2.0 && (wg.height(x + 14.0, z) - (sea + 1.0)).abs() < 1.0);
            let Some((x, z)) = shore else {
                log::warn!("fluid test: no shore near spawn");
                return;
            };
            for k in 0..12 {
                let center = Vec3::new(x + 2.0 + k as f32 * 1.4, sea - 1.4, z);
                self.send(ClientMsg::Edit(EditOp { center, radius: 1.6, kind: EditKind::Dig, material: Material::Air }));
            }
            self.fluid_test_end = Vec3::new(x + 16.0, sea - 1.4, z);
            self.player.mode = Mode::Fly;
            self.player.teleport(Vec3::new(x + 8.0, sea + 9.0, z - 11.0));
            (self.player.yaw, self.player.pitch) = (1.45, -0.6);
            log::info!("fluid test: trench from the shore at ({x:.0}, {z:.0}) to {:.1}", self.fluid_test_end);
        }
        if self.fluid_test_stage == 1 && (t * 2.0).fract() < 0.03 {
            let fill = self.fluids.fraction_at(&self.world, self.fluid_test_end);
            log::info!("fluid test t={t:.1}: trench end {:.2} full, {} fluid chunks here", fill, self.fluids.fluids.chunks.len());
        }
    }

    /// Footsteps on whatever is underfoot, a thump on landing, a splash on falling into water.
    fn body_sounds(&mut self, dt: f32) {
        let Some(audio) = &mut self.audio else { return };
        let p = &self.player;
        let feet = p.feet;
        if p.mode == Mode::Walk {
            let speed = Vec2::new(p.velocity.x, p.velocity.z).length();
            let wading = p.submerged > 0.08;
            if p.grounded && speed > 0.6 && p.submerged < 0.5 {
                self.stride += speed * dt;
                // A step every 1.6 m walking, a little longer running.
                if self.stride > 1.4 + 0.05 * speed {
                    self.stride = 0.0;
                    let under = feet - Vec3::Y * 0.25;
                    let name = if wading {
                        "step_water".to_owned()
                    } else if self.world.density_at(under) >= 0.0 && self.building.sdf(under) < 0.1 {
                        // On a building.
                        "step_hard".to_owned()
                    } else {
                        format!("step_{}", crate::audio::surface(self.world.sample_at(under).1))
                    };
                    let volume = 0.18 + 0.025 * speed;
                    audio.play_at(&name, feet, crate::audio::Range(2.0, 30.0), volume);
                    // The body's weight under it, softer when wading.
                    let weight = if wading { 0.25 } else { 0.55 };
                    audio.play_at("thump", feet, crate::audio::Range(2.0, 20.0), volume * weight);
                }
            }
            if p.grounded && !self.was_grounded && self.falling < -5.0 {
                let volume = ((-self.falling - 4.0) / 12.0).clamp(0.3, 1.0);
                audio.play_at("land", feet, crate::audio::Range(2.0, 30.0), volume);
                audio.play_at("thump", feet, crate::audio::Range(2.0, 20.0), volume * 0.8);
            }
            if p.submerged > 0.35 && self.was_submerged <= 0.35 && self.falling < -3.0 {
                audio.play_at("splash", feet, crate::audio::Range(2.0, 40.0), ((-self.falling) / 12.0).clamp(0.4, 1.0));
            }
        }
        self.was_grounded = p.grounded;
        self.falling = p.velocity.y;
        self.was_submerged = p.submerged;
    }

    /// Digging and placing make a sound by what was dug or placed (sampled before the edit lands).
    fn edit_sound(&mut self, op: &EditOp) {
        let Some(audio) = &mut self.audio else { return };
        // Only the small sphere edits a player makes (not cooling lava, levelling or a blast).
        if op.radius > 3.0 || op.center.distance(self.camera.position) > 40.0 {
            return;
        }
        let range = crate::audio::Range(2.0, 40.0);
        match op.kind {
            EditKind::Dig => {
                let material = self.world.sample_at(op.center).1;
                let name = if crate::audio::surface(material) == "hard" { "dig_hard" } else { "dig_soft" };
                audio.play_at(name, op.center, range, 0.8);
            }
            EditKind::Fill => audio.play_at("place", op.center, range, 0.7),
            _ => {}
        }
    }

    /// What a new report from the volcanoes sounds like: bombs landing, an eruption bursting out,
    /// a blast's boom as its shockwave reaches the listener.
    fn volcanic_sounds(&mut self, report: &asc_sim::volcanism::Report) {
        use asc_sim::volcanism::Phase;
        let Some(audio) = &mut self.audio else { return };
        let ear = self.camera.position;
        // A report comes every tick (0.05 s): no bomb flies further than this between them.
        for (at, size) in crate::audio::landed(&self.volcanism.bombs, &report.bombs, 6.0) {
            let volume = 0.45 + 0.13 * size as f32;
            audio.play_at("bomb_land", at, crate::audio::Range(8.0, 900.0), volume);
        }
        for a in &report.active {
            let before = self.volcanism.active.iter().find(|b| b.summit.distance(a.summit) < 50.0).map(|b| b.phase);
            if a.phase == Phase::Erupting && before.is_some_and(|p| p != Phase::Erupting) {
                audio.play_at("boom", a.summit, crate::audio::Range(300.0, 9000.0), 1.0);
            }
        }
        for b in &report.blasts {
            let before = self.volcanism.blasts.iter().find(|o| o.center.distance(b.center) < 1.0).map_or(0.0, |o| o.radius);
            let d = b.center.distance(ear);
            if d > before && d <= b.radius {
                // The shockwave is here: everywhere at once, loudest close in.
                audio.play("blast", (1.0 - d / 9000.0).clamp(0.35, 1.0));
            }
        }
    }

    /// Put the listener at the camera; every so often, set the ambience from what's round it.
    fn listen(&mut self) {
        use asc_sim::volcanism::Phase;
        use crate::audio::Ambience;
        let Some(audio) = &mut self.audio else { return };
        let at = self.camera.position;
        audio.listen(at, self.camera.yaw);
        if !audio.ambience_due() {
            return;
        }
        let wg = &self.world.worldgen;
        let mut scene = crate::audio::Scene::default();
        let ground = wg.height(at.x, at.z);
        // Under the ground (a cave, a tunnel): no wind or rain.
        let outside = at.y > ground - 2.0;
        let exposure = if outside { 0.35 + 0.65 * smoothstep(0.0, 40.0, at.y - ground) } else { 0.0 };
        let weather = self.weather.at(wg, at.x, at.z, ground, self.sim.ticks, self.sim.time_of_day);
        let wind = weather.wind.length();
        scene.levels.insert(Ambience::Wind, (None, (0.15 + wind / 14.0).min(1.0) * exposure * 0.4));
        if weather.fall == asc_world::weather::Fall::Rain && outside {
            scene.levels.insert(Ambience::Rain, (None, weather.precipitation * 0.9));
        }
        let (river, lake) = crate::audio::water_near(wg, at, 48.0);
        if let Some((p, strength)) = river {
            scene.levels.insert(Ambience::River, (Some(p), 0.2 + 0.25 * strength));
        }
        if let Some((p, _)) = lake {
            scene.levels.insert(Ambience::Lake, (Some(p), 0.3));
        }
        let mut lava = self.fluids.lava_lights(at);
        lava.extend(self.streamer.lava_lights(&self.fluids, at, 120.0));
        if let Some(l) = lava.iter().min_by(|a, b| a.position.distance(at).total_cmp(&b.position.distance(at))) {
            scene.levels.insert(Ambience::Lava, (Some(l.position), 0.9));
            // Now and then a bubble bursts or the crust hisses (one every two seconds or so).
            if audio.chance(0.12) {
                audio.play_at("lava_pop", l.position, crate::audio::Range(3.0, 60.0), 0.7);
            }
        }
        let fires = self.effects.fire_lights(wg, at, 6);
        if let Some(f) = fires.first() {
            scene.levels.insert(Ambience::Fire, (Some(f.position), 0.35 + 0.1 * fires.len() as f32));
        }
        if let Some(a) = self.volcanism.active.iter().min_by(|a, b| a.summit.distance(at).total_cmp(&b.summit.distance(at))) {
            let rumble = match a.phase {
                Phase::Rumbling => 0.4 + 0.6 * a.strength,
                Phase::Erupting => 1.0,
                Phase::Waning => 0.7 * a.strength,
                Phase::Quiet => 0.0,
            };
            scene.levels.insert(Ambience::Rumble, (Some(a.summit), rumble));
            if a.phase == Phase::Erupting {
                scene.levels.insert(Ambience::Roar, (Some(a.summit), 0.4 + 0.6 * a.strength));
            }
        }
        scene.underwater = self.fluids.fraction_at(&self.world, at) > 0.5;
        if scene.underwater {
            scene.levels.insert(Ambience::Underwater, (None, 0.7));
        }
        audio.set_scene(&scene);
    }

    /// Ash in the air round the camera (0..1: only inside a volcano's smoke), and how far to shake
    /// the camera (rumbling and erupting volcanoes near by).
    fn volcanic_air(&self) -> (f32, Vec3) {
        use asc_sim::volcanism::Phase;
        let at = self.camera.position;
        let ash = self.effects.smoke_at(at);
        let mut quake = 0.0f32;
        for a in &self.volcanism.active {
            let d = Vec2::new(a.summit.x - at.x, a.summit.z - at.z).length();
            let shaking = match a.phase {
                Phase::Rumbling => a.strength * 0.5,
                Phase::Erupting => a.strength,
                Phase::Waning => a.strength * 0.3,
                Phase::Quiet => 0.0,
            };
            quake = quake.max(shaking * (1.0 - smoothstep(a.radius * 0.5, a.radius * 4.0, d)));
        }
        // A blast's shockwave: a hard jolt as it passes.
        for b in &self.volcanism.blasts {
            let d = Vec2::new(b.center.x - at.x, b.center.z - at.z).length();
            quake = quake.max(6.0 * (1.0 - ((d - b.radius).abs() / 400.0).min(1.0)));
        }
        // A flow's thundering front.
        for f in &self.volcanism.flows {
            let front = Vec2::new(f.origin.x, f.origin.z) + f.dir * f.front;
            quake = quake.max(1.5 * (1.0 - smoothstep(100.0, 1500.0, front.distance(Vec2::new(at.x, at.z)))));
        }
        let t = self.started.elapsed().as_secs_f32();
        let quake = if self.options.reduce_motion { 0.0 } else { quake };
        let shake = Vec3::new((t * 37.0).sin() + (t * 23.0).sin() * 0.5, (t * 41.0).sin() * 0.6, (t * 29.0).cos()) * 0.06 * quake;
        (ash, shake)
    }

    /// Draw a small crate where each stash lies.
    fn sync_stashes(&mut self) {
        let model = self.parts.model(0, asc_build::PartKind::Foundation);
        let extras = self
            .stashes
            .values()
            .map(|at| asc_render::PartInstance { model, position: *at, yaw: 0.4, scale: 0.3, sway: 0.0 })
            .collect();
        self.building.set_extras(extras);
    }

    /// Do an inventory op now (so the screen responds at once) and ask the server to do it too.
    fn inventory_op(&mut self, op: InventoryOp) {
        self.inventory_seq += 1;
        self.inventory.apply(&self.content, op, &mut self.held);
        self.pending_ops.push((self.inventory_seq, op));
        self.send(ClientMsg::Inventory { seq: self.inventory_seq, op });
    }

    /// The server's inventory, with the ops it hasn't seen yet applied on top.
    fn inventory_snapshot(&mut self, snapshot: &InventorySnapshot) {
        self.held = self.inventory.restore(snapshot);
        self.pending_ops.retain(|(seq, _)| *seq > snapshot.seq);
        for (_, op) in self.pending_ops.clone() {
            self.inventory.apply(&self.content, op, &mut self.held);
        }
    }

    /// Ask the server to place a piece; whether it did (the reason if not, from its notice).
    fn place_piece(&mut self, piece: asc_build::Piece, gap: f32) -> Result<(), String> {
        self.send(ClientMsg::Place { piece, gap });
        self.sync();
        if self.building.structures.get(&piece.key) == Some(&piece) {
            Ok(())
        } else {
            Err(self.notice.as_ref().map_or_else(|| "refused".to_owned(), |(t, _)| t.clone()))
        }
    }

    fn toggle_build_menu(&mut self) {
        self.building.menu_open = !self.building.menu_open;
        if self.building.menu_open {
            self.set_captured(false);
        } else {
            self.set_captured(true);
        }
    }

    fn choose_build_tool(&mut self, choice: MenuChoice) {
        self.building.menu_open = false;
        self.set_captured(true);
        self.set_build_tool(Some(match choice {
            MenuChoice::Piece(tier, kind) => BuildTool::Piece(Selection { tier, kind }),
            MenuChoice::Flatten => BuildTool::Flatten,
        }));
    }

    fn set_build_tool(&mut self, tool: Option<BuildTool>) {
        self.build_tool = tool;
        self.building.selection = match tool {
            Some(BuildTool::Piece(sel)) => Some(sel),
            _ => None,
        };
        self.building.preview = None;
        self.flatten_preview = None;
        self.renderer.set_ghost(None);
    }

    /// Height of the real (edited) ground at (x, z), probing down from `above`.
    fn ground_height(world: &World, x: f32, z: f32, above: f32) -> f32 {
        let top = Vec3::new(x, above + 40.0, z);
        raycast(|p| world.density_at(p), top, -Vec3::Y, 120.0)
            .map_or(world.worldgen.height(x, z), |h| h.point.y)
    }

    /// Ghost preview, place, flatten and dismantle with the active build tool.
    fn use_build_tool(&mut self) {
        let camera = self.camera.position;
        let above = self.hit.map_or(camera.y, |h| h.point.y.max(camera.y));
        let world = &self.world;
        let terrain = |x: f32, z: f32| Self::ground_height(world, x, z, above);
        let clicked = self.captured && self.input.button_pressed(MouseButton::Left);
        match self.build_tool {
            Some(BuildTool::Piece(_)) => {
                self.building.update_preview(self.hit, camera, &terrain);
                // The gap rule needs the snap, so recheck with it.
                if let Some((snap, valid)) = &mut self.building.preview {
                    *valid = self.building.structures.check_snap(snap);
                }
                if clicked {
                    match self.building.request() {
                        Ok((piece, gap)) => self.send(ClientMsg::Place { piece, gap }),
                        Err(why) => self.show_notice(format!("Can't place: {why}")),
                    }
                }
            }
            Some(BuildTool::Flatten) => {
                let lattice = &self.building.structures.lattice;
                self.flatten_preview = self.hit.filter(|h| h.distance <= BUILD_REACH).map(|h| {
                    let cell = lattice.cell_of(h.point.x, h.point.z);
                    let center = (cell.as_vec2() + 0.5) * lattice.cell;
                    let height = (h.point.y / lattice.unit).round() * lattice.unit;
                    (center, height)
                });
                if clicked && let Some((center, height)) = self.flatten_preview {
                    self.flatten(center, height);
                }
            }
            None => {}
        }
        if self.captured && self.input.pressed(KeyCode::KeyX) {
            match self.building.dismantle_target(self.hit) {
                Ok(key) => self.send(ClientMsg::Dismantle { key }),
                Err(why) => self.show_notice(format!("Can't dismantle: {why}")),
            }
        }
        if self.captured && self.input.button_pressed(MouseButton::Right) {
            self.set_build_tool(None);
        }
    }

    /// Level one grid cell to `height`: cut above, fill below with soil.
    fn flatten(&mut self, center: glam::Vec2, height: f32) {
        self.send(ClientMsg::Flatten { center, height });
    }

    fn build_status(&self) -> Option<(String, bool)> {
        let spec = &self.parts.spec;
        match self.build_tool? {
            BuildTool::Piece(sel) => {
                let name = format!("{} {}", spec.tiers[sel.tier].name, spec.part(sel.kind).name);
                Some(match &self.building.preview {
                    Some((_, Ok(()))) => {
                        (format!("{name}: click to place   (wheel: tier, X: dismantle, RMB: stop)"), true)
                    }
                    Some((_, Err(why))) => (format!("{name}: {why}"), false),
                    None => (format!("{name}: aim at the ground or a piece"), false),
                })
            }
            BuildTool::Flatten => Some(match self.flatten_preview {
                Some((_, h)) => {
                    (format!("Flatten to {h:.1} m: click to level this cell   (RMB: stop)"), true)
                }
                None => ("Flatten: aim at the ground".to_owned(), false),
            }),
        }
    }

    /// Build test: flatten a 3x3 site, build a 4 m room with a doorway, walk in, and log.
    fn build_test_step(&mut self) {
        use asc_build::{PartKind, PieceKey, Side, Target};
        let t = self.started.elapsed().as_secs_f32();
        match self.build_test_phase {
            0 if t > 2.0 => {
                let lattice = self.building.structures.lattice.clone();
                let site = lattice.cell_of(self.spawn.x + 6.0, self.spawn.z);
                let ground =
                    self.world.worldgen.height((site.x as f32 + 1.5) * 2.0, (site.y as f32 + 1.5) * 2.0);
                let height = (ground / lattice.unit).round() * lattice.unit;
                for dz in 0..3 {
                    for dx in 0..3 {
                        let cell = site + glam::IVec2::new(dx, dz);
                        self.flatten((cell.as_vec2() + 0.5) * lattice.cell, height);
                    }
                }
                // Snapping reads the levelled ground.
                self.sync();
                let world = &self.world;
                let terrain = |x: f32, z: f32| Self::ground_height(world, x, z, height + 5.0);
                // Foundations snap to the terrain alone, so all nine can be worked out first.
                let mut snaps = Vec::new();
                for dz in 0..3 {
                    for dx in 0..3 {
                        let cell = site + glam::IVec2::new(dx, dz);
                        let p = Vec3::new((cell.x as f32 + 0.5) * 2.0, height, (cell.y as f32 + 0.5) * 2.0);
                        let snap = self.building.structures.snap(
                            PartKind::Foundation,
                            Target::Terrain(p),
                            Vec3::ZERO,
                            &terrain,
                        );
                        snaps.push((cell, snap));
                    }
                }
                let mut placed = 0;
                let mut failures = Vec::new();
                for (cell, snap) in snaps {
                    let piece = asc_build::Piece { key: snap.key, tier: 1, grounded: snap.grounded };
                    match self.place_piece(piece, snap.gap) {
                        Ok(()) => placed += 1,
                        Err(why) => failures.push(format!("foundation {cell}: {why} (gap {:.2})", snap.gap)),
                    }
                }
                let mut heights: Vec<(glam::IVec2, i32)> =
                    self.building.structures.iter().map(|p| (p.key.cell, p.key.y)).collect();
                heights.sort_by_key(|(c, _)| (c.y, c.x));
                log::info!("build test: flattened to {height:.2}; foundation (cell, base units) {heights:?}");
                let floor = self.building.structures.iter().map(|p| lattice.top(&p.key)).max().unwrap_or(0);
                // Two layers of walls round the outside, leaving a doorway in the south middle.
                for layer in 0..2 {
                    let y = floor + layer * lattice.wall_units;
                    for i in 0..3 {
                        let walls = [
                            (site + glam::IVec2::new(i, 0), Side::NegZ),
                            (site + glam::IVec2::new(i, 2), Side::PosZ),
                            (site + glam::IVec2::new(0, i), Side::NegX),
                            (site + glam::IVec2::new(2, i), Side::PosX),
                        ];
                        for (cell, side) in walls {
                            if side == Side::NegZ && i == 1 {
                                continue;
                            }
                            let key = PieceKey::wall_on_side(cell, y, side);
                            match self.place_piece(asc_build::Piece { key, tier: 2, grounded: false }, 0.0) {
                                Ok(()) => placed += 1,
                                Err(why) => failures.push(format!("wall {key:?}: {why}")),
                            }
                        }
                    }
                }
                // Roof: edge cells first (they rest on walls), then the middle (rests on them).
                let roof = floor + 2 * lattice.wall_units;
                let mut cells: Vec<glam::IVec2> =
                    (0..3).flat_map(|z| (0..3).map(move |x| glam::IVec2::new(x, z))).collect();
                cells.sort_by_key(|c| u8::from(*c == glam::IVec2::ONE));
                for c in cells {
                    let key = PieceKey::ceiling(site + c, roof);
                    match self.place_piece(asc_build::Piece { key, tier: 3, grounded: false }, 0.0) {
                        Ok(()) => placed += 1,
                        Err(why) => failures.push(format!("ceiling {key:?}: {why}")),
                    }
                }
                log::info!("build test: placed {placed} pieces, {} failures {failures:?}", failures.len());
                // Stand in the doorway, facing in.
                let door = (site + glam::IVec2::new(1, 0)).as_vec2() * 2.0 + glam::Vec2::new(1.0, 0.6);
                self.player.mode = Mode::Walk;
                self.player.teleport(Vec3::new(door.x, lattice.height(floor) + 0.3, door.y));
                (self.player.yaw, self.player.pitch) = (std::f32::consts::FRAC_PI_2, 0.05);
                self.build_test_floor = lattice.height(floor);
                self.build_test_phase = 1;
            }
            1 | 2 if (t * 2.0).fract() < 0.02 => {
                log::info!(
                    "build test t={t:.1}: feet {:.2} v {:.2} grounded {}",
                    self.player.feet,
                    self.player.velocity,
                    self.player.grounded
                );
                if self.build_test_phase == 1 && t > 4.0 {
                    self.build_test_phase = 2;
                }
            }
            1 if t > 4.0 => self.build_test_phase = 2,
            2 if t > 9.0 => {
                let feet = self.player.feet;
                let lattice = &self.building.structures.lattice;
                let under_roof = self.building.structures.sdf(feet + Vec3::Y * 3.5) < 0.6;
                log::info!(
                    "build test: feet {feet:.2} floor {:.2} grounded {} | inside, under the roof: {under_roof} | pieces {}",
                    self.build_test_floor,
                    self.player.grounded,
                    self.building.structures.len(),
                );
                let _ = lattice;
                // Finish in build mode with a wall ghost aimed at the floor, for screenshots.
                self.set_build_tool(Some(BuildTool::Piece(Selection { tier: 0, kind: PartKind::Wall })));
                self.player.teleport(self.player.feet - Vec3::Z * 2.5);
                (self.player.yaw, self.player.pitch) = (std::f32::consts::FRAC_PI_2, -0.75);
                self.build_test_phase = 3;
            }
            _ => {}
        }
    }

    fn toggle_inventory(&mut self) {
        self.inventory_open = !self.inventory_open;
        if self.inventory_open {
            self.set_captured(false);
        } else {
            if self.held.is_some() {
                self.inventory_op(InventoryOp::ReturnHeld);
            }
            self.set_captured(true);
        }
    }

    fn show_notice(&mut self, text: String) {
        log::info!("notice: {text}");
        self.notice = Some((text, Instant::now()));
    }

    fn mine_and_place(&mut self, dt: f32) {
        self.mining.tick_cooldown(dt);
        // Mining edits terrain only: a crosshair on a piece mines nothing.
        if self.hit.is_some_and(|h| self.building.structures.piece_at(h.point, 0.05).is_some()) {
            self.mining.reset();
            return;
        }
        let (mine, place) = if self.options.mine_test {
            self.mine_test_input()
        } else if self.captured {
            (self.input.button_held(MouseButton::Left), self.input.button_held(MouseButton::Right))
        } else {
            (false, false)
        };
        let node = self.hit.and_then(|h| self.scatter.node_at(h.point, NODE_TOLERANCE));
        if mine && let Some(node) = node {
            self.mine_node(dt, node);
            return;
        }
        if mine {
            if let Some(hit) = self.mining.mine(dt, self.hit, &self.world, &self.content) {
                self.send(ClientMsg::Bite { point: hit.point, normal: hit.normal });
            }
        } else {
            self.mining.reset();
        }
        if place && let Some(hit) = self.hit {
            let fresh = self.input.button_pressed(MouseButton::Right) || self.options.mine_test;
            if self.mining.try_place() {
                self.send(ClientMsg::Fill { point: hit.point, normal: hit.normal, fresh });
            }
        }
    }

    /// Hold to break an ore node; it yields a handful of its ore.
    fn mine_node(&mut self, dt: f32, node: crate::scatter::NodeHit) {
        use asc_sim::rules::{DEEP_NODE_HARDNESS, SURFACE_NODE_HARDNESS};
        let hardness = self.content.ores[node.ore].hardness
            * match node.kind {
                crate::scatter::NodeKind::Surface => SURFACE_NODE_HARDNESS,
                crate::scatter::NodeKind::Deep => DEEP_NODE_HARDNESS,
            };
        if self.mining.mine_node(dt, node, hardness) {
            self.send(ClientMsg::MineNode { id: node.id });
        }
    }

    /// Mine test: sweep the crosshair over the ground ahead, mining a shallow pit, then aim ahead and
    /// place five lumps of soil.
    fn mine_test_input(&mut self) -> (bool, bool) {
        let t = self.started.elapsed().as_secs_f32();
        if t < 25.0 {
            self.player.yaw = (t * 0.7).sin() * 0.9;
            self.player.pitch = -0.75;
        } else {
            self.player.yaw = 1.6;
            self.player.pitch = -0.55;
            if let Some(soil) = self.content.item_id("soil")
                && let Some(i) =
                    (0..HOTBAR).find(|i| self.inventory.slots[*i].is_some_and(|s| s.item == soil))
                && i != self.inventory.selected
            {
                self.inventory_op(InventoryOp::Select(i));
            }
        }
        (t > 3.0 && t < 25.0, (26.0..27.1).contains(&t))
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
            EditKind::Dig | EditKind::DigBox { .. } => hit.point - hit.normal * r * 0.3,
            EditKind::Fill | EditKind::FillBox { .. } => hit.point + hit.normal * r * 0.3,
        };
        if kind == EditKind::Fill && center.distance(self.camera.position) < r + 0.5 {
            return;
        }
        self.send(ClientMsg::Edit(EditOp { center, radius: r, kind, material: Material::Rock }));
        self.last_edit = Instant::now();
    }

    /// About 1000 pieces: a 10x10 foundation slab with walls on every cell's north and east edge and
    /// a ceiling over each storey, three storeys high, a little way from spawn.
    fn build_stress(&mut self) {
        use asc_build::{Edge, PieceKey};
        let lattice = self.building.structures.lattice.clone();
        let site = lattice.cell_of(self.spawn.x + 10.0, self.spawn.z - 10.0);
        let mut highest = f32::MIN;
        for dz in 0..=10 {
            for dx in 0..=10 {
                let (x, z) = ((site.x + dx) as f32 * lattice.cell, (site.y + dz) as f32 * lattice.cell);
                highest = highest.max(self.world.worldgen.height(x, z));
            }
        }
        let floor = lattice.units_at_or_above(highest + 0.5);
        let base = floor - lattice.foundation_units;
        let cells: Vec<glam::IVec2> =
            (0..10).flat_map(|z| (0..10).map(move |x| site + glam::IVec2::new(x, z))).collect();
        let mut failures = 0;
        let mut place = |state: &mut Self, key: PieceKey, tier: usize, grounded: bool| {
            if state.place_piece(asc_build::Piece { key, tier: tier as u8, grounded }, 0.0).is_err() {
                failures += 1;
            }
        };
        for &cell in &cells {
            place(self, PieceKey::foundation(cell, base), 1, true);
        }
        for storey in 0..3 {
            let y = floor + storey * 2 * lattice.wall_units;
            for layer in 0..2 {
                for &cell in &cells {
                    for edge in [Edge::North, Edge::East] {
                        let tier = (storey as usize + layer as usize) % 4;
                        place(
                            self,
                            PieceKey::wall(cell, y + layer * lattice.wall_units, edge),
                            tier,
                            false,
                        );
                    }
                }
            }
            for &cell in &cells {
                place(self, PieceKey::ceiling(cell, y + 2 * lattice.wall_units), 3, false);
            }
        }
        log::info!("build stress: {} pieces placed, {failures} refused", self.building.structures.len());
    }

    /// Every tier (rows) and part (columns) laid out in front of the spawn point.
    fn showcase(&mut self) {
        use asc_build::PartKind;
        let mut instances = Vec::new();
        let origin = self.spawn + Vec3::new(6.0, 0.0, -4.5);
        for tier in 0..self.parts.spec.tiers.len() {
            for (col, kind) in PartKind::ALL.into_iter().enumerate() {
                let x = origin.x + col as f32 * 3.0;
                let z = origin.z + tier as f32 * 3.0;
                let ground = find_ground(&self.world, x, z).y;
                let y = match kind {
                    PartKind::Ceiling => ground + 1.4,
                    _ => ground,
                };
                instances.push(asc_render::PartInstance::rigid(
                    self.parts.model(tier, kind),
                    Vec3::new(x, y, z),
                    3,
                ));
            }
        }
        // Nature models in a row per kind behind the parts, spaced by how big that kind is.
        let mut z = origin.z + self.parts.spec.tiers.len() as f32 * 3.0 + 4.0;
        for (kind, spacing) in
            [("tree", 9.0), ("shrub", 3.0), ("plant", 1.6), ("grass", 1.2), ("rock", 3.0), ("debris", 4.0)]
        {
            let all: Vec<_> = self.nature.models.iter().filter(|m| m.spec.kind == kind).collect();
            for row in all.chunks(8) {
                z += spacing * 0.5;
                for (i, model) in row.iter().enumerate() {
                    let x = origin.x + (i as f32 - row.len() as f32 / 2.0) * spacing;
                    let ground = find_ground(&self.world, x, z).y;
                    instances.push(asc_render::PartInstance {
                        model: model.id,
                        position: Vec3::new(x, ground, z),
                        yaw: i as f32 * 1.3,
                        scale: 1.0,
                        sway: 1.0,
                    });
                }
                z += spacing * 0.5;
            }
        }
        log::info!("showcase: {} parts and nature models", instances.len());
        self.renderer.set_parts(&instances);
    }

    fn inventory_summary(&self) -> String {
        let mut parts: Vec<String> = Vec::new();
        for (i, item) in self.content.items.iter().enumerate() {
            let id = crate::content::ItemId(i as u16);
            let count = self.inventory.count(id);
            let rem = self.inventory.remainders.get(&id).copied().unwrap_or(0.0);
            if count > 0 || rem > 0.0 {
                parts.push(format!("{} {} (+{rem:.2})", item.id, count));
            }
        }
        if parts.is_empty() { "empty".into() } else { parts.join(", ") }
    }

    fn hud_info(&self) -> HudInfo {
        HudInfo {
            visible: self.show_hud,
            mode: match self.player.mode {
                Mode::Walk => "walking",
                Mode::Fly => "flying",
            },
            grounded: self.player.grounded,
            speed: self.player.velocity.length(),
            target: self.mining.target_name(),
            physics_ms: self.worst_physics_ms,
            frame_ms: self.frame_ms,
            position: self.camera.position,
            frame: self.last_stats,
            stream: self.streamer.stats(),
            gpu_chunks: self.renderer.chunk_count(),
            ticks: self.sim.ticks,
            time_of_day: self.sim.time_of_day,
            tps: self.server_stats.0,
            mspt: self.server_stats.1,
            brush_radius: self.brush_radius,
            fly_speed: self.fly_speed,
            view_radius: self.streamer.view_radius,
            vsync: self.vsync,
            edits: self.world.edits.len(),
            gpu_name: self.renderer.gpu().adapter_info.name.clone(),
            far: self.far.stats(),
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
            Ok(mut state) => {
                log::info!("inventory: {} | player at {:.1}", state.inventory_summary(), state.player.feet);
                if state.options.showcase {
                    state.showcase();
                }
                if state.options.build_stress {
                    state.build_stress();
                }
                state.sync_stashes();
                self.state = Some(state);
            }
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
                if !state.captured
                    && button_state == ElementState::Pressed
                    && !egui_consumed
                    && !state.inventory_open
                    && !state.building.menu_open
                {
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

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_mut() {
            state.save_and_wait();
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = &self.state {
            state.window.request_redraw();
        }
    }
}

fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}
