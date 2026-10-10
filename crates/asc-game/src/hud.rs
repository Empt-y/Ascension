//! Debug HUD (F3) and crosshair.

use asc_render::FrameStats;
use glam::Vec3;

use crate::streaming::StreamStats;

pub struct HudInfo {
    pub visible: bool,
    pub captured: bool,
    pub frame_ms: f32,
    pub position: Vec3,
    pub frame: FrameStats,
    pub stream: StreamStats,
    pub gpu_chunks: usize,
    pub ticks: u64,
    pub time_of_day: f32,
    pub brush_radius: f32,
    pub fly_speed: f32,
    pub view_radius: i32,
    pub vsync: bool,
    pub edits: usize,
    pub gpu_name: String,
    pub hot_reload: bool,
}

fn clock(time_of_day: f32) -> String {
    let minutes = (time_of_day * 24.0 * 60.0) as u32;
    format!("{:02}:{:02}", minutes / 60, minutes % 60)
}

pub fn draw(ctx: &egui::Context, info: &HudInfo) {
    if info.captured {
        let center = ctx.content_rect().center();
        let painter =
            ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("crosshair")));
        painter.circle_stroke(center, 3.0, egui::Stroke::new(1.5, egui::Color32::from_white_alpha(200)));
    }
    if !info.visible {
        return;
    }
    egui::Window::new("Ascension")
        .anchor(egui::Align2::LEFT_TOP, [8.0, 8.0])
        .resizable(false)
        .collapsible(false)
        .title_bar(false)
        .frame(egui::Frame::window(&ctx.global_style()).fill(egui::Color32::from_black_alpha(160)))
        .show(ctx, |ui| {
            ui.style_mut().override_text_style = Some(egui::TextStyle::Monospace);
            let s = &info.stream;
            ui.label(format!(
                "{:.0} fps  {:.2} ms  vsync {}",
                1000.0 / info.frame_ms.max(0.01),
                info.frame_ms,
                if info.vsync { "on" } else { "off" }
            ));
            ui.label(&info.gpu_name);
            ui.separator();
            let p = info.position;
            ui.label(format!("pos {:.1} {:.1} {:.1}", p.x, p.y, p.z));
            ui.label(format!("time {}  tick {}", clock(info.time_of_day), info.ticks));
            ui.separator();
            ui.label(format!(
                "chunks drawn {}  shadow {}",
                info.frame.chunks_drawn, info.frame.shadow_chunks
            ));
            ui.label(format!(
                "on gpu {}  tracked {}  queued {}  building {}",
                info.gpu_chunks, s.loaded, s.queued, s.in_flight
            ));
            ui.label(format!("triangles {}k", info.frame.triangles / 1000));
            ui.label(format!("build {:.2} ms  mesh {:.2} ms", s.avg_build_ms, s.avg_mesh_ms));
            ui.label(format!(
                "edits {}  last edit visible in {}",
                info.edits,
                s.last_edit_ms.map_or("-".to_owned(), |ms| format!("{ms:.1} ms"))
            ));
            ui.label(format!(
                "view radius {} chunks  brush {:.1} m  speed {:.0} m/s",
                info.view_radius, info.brush_radius, info.fly_speed
            ));
            ui.separator();
            ui.label("click: capture mouse   Esc: release / quit");
            ui.label("WASD Space Ctrl: fly   Shift: fast   Alt+wheel: speed");
            ui.label("LMB dig   RMB fill   wheel: brush size");
            ui.label("T / Shift+T: time   [ ]: view radius   F3 HUD   F6 vsync   F11 fullscreen");
            if info.hot_reload {
                ui.label("F5: reload shaders");
            }
        });
}
