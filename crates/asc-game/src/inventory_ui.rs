//! Hotbar, inventory screen and mining progress ring (egui).

use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

use crate::content::Content;
use crate::inventory::{HOTBAR, Inventory, InventoryOp, SLOTS, Stack};

const SLOT: f32 = 46.0;
const GAP: f32 = 4.0;

fn item_color(content: &Content, stack: Stack) -> Color32 {
    let [r, g, b] = content.item(stack.item).color;
    // Item colours are linear; egui's Rgba is linear too and converts to sRGB for display.
    egui::Rgba::from_rgb(r, g, b).into()
}

fn paint_stack(painter: &egui::Painter, rect: Rect, content: &Content, stack: Stack) {
    let inner = rect.shrink(7.0);
    painter.rect_filled(inner, 5.0, item_color(content, stack));
    painter.rect_stroke(inner, 5.0, Stroke::new(1.0, Color32::from_black_alpha(120)), StrokeKind::Inside);
    if stack.count > 1 {
        let pos = rect.right_bottom() - Vec2::new(4.0, 2.0);
        let font = FontId::proportional(15.0);
        painter.text(pos + Vec2::splat(1.0), Align2::RIGHT_BOTTOM, stack.count, font.clone(), Color32::BLACK);
        painter.text(pos, Align2::RIGHT_BOTTOM, stack.count, font, Color32::WHITE);
    }
}

fn paint_slot_frame(painter: &egui::Painter, rect: Rect, highlighted: bool) {
    painter.rect_filled(rect, 6.0, Color32::from_black_alpha(150));
    let stroke = if highlighted {
        Stroke::new(2.5, Color32::from_rgb(255, 220, 120))
    } else {
        Stroke::new(1.0, Color32::from_white_alpha(40))
    };
    painter.rect_stroke(rect, 6.0, stroke, StrokeKind::Inside);
}

/// The hotbar along the bottom of the screen, with the selected item's name above it.
pub fn hotbar(ctx: &egui::Context, content: &Content, inventory: &Inventory) {
    let screen = ctx.content_rect();
    let width = HOTBAR as f32 * (SLOT + GAP) - GAP;
    let origin = Pos2::new(screen.center().x - width / 2.0, screen.bottom() - SLOT - 14.0);
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("hotbar")));
    for i in 0..HOTBAR {
        let rect = Rect::from_min_size(origin + Vec2::new(i as f32 * (SLOT + GAP), 0.0), Vec2::splat(SLOT));
        paint_slot_frame(&painter, rect, i == inventory.selected);
        if let Some(stack) = inventory.slots[i] {
            paint_stack(&painter, rect, content, stack);
        }
    }
    if let Some(stack) = inventory.selected_stack() {
        let name = &content.item(stack.item).name;
        let pos = Pos2::new(screen.center().x, origin.y - 8.0);
        painter.text(
            pos + Vec2::splat(1.0),
            Align2::CENTER_BOTTOM,
            name,
            FontId::proportional(16.0),
            Color32::BLACK,
        );
        painter.text(pos, Align2::CENTER_BOTTOM, name, FontId::proportional(16.0), Color32::WHITE);
    }
}

/// Health (and air, when it's running out) above the hotbar.
pub fn vitals(ctx: &egui::Context, v: &asc_sim::vitals::Vitals) {
    use asc_sim::vitals::{MAX_AIR, MAX_HEALTH};
    let screen = ctx.content_rect();
    let width = HOTBAR as f32 * (SLOT + GAP) - GAP;
    let left = screen.center().x - width / 2.0;
    let bottom = screen.bottom() - SLOT - 14.0 - 30.0;
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Background, egui::Id::new("vitals")));
    let bar = |y: f32, fraction: f32, color: Color32| {
        let rect = Rect::from_min_size(Pos2::new(left, y), Vec2::new(width * 0.5 - GAP, 9.0));
        painter.rect_filled(rect, 4.0, Color32::from_black_alpha(150));
        let fill = Rect::from_min_size(rect.min, Vec2::new(rect.width() * fraction.clamp(0.0, 1.0), rect.height()));
        painter.rect_filled(fill, 4.0, color);
    };
    bar(bottom, v.health as f32 / MAX_HEALTH as f32, Color32::from_rgb(210, 50, 45));
    if v.air < MAX_AIR {
        bar(bottom - 13.0, v.air as f32 / MAX_AIR as f32, Color32::from_rgb(80, 160, 230));
    }
}

/// A red edge on the screen after being hurt, fading out (`strength` 0..1).
pub fn hurt_overlay(ctx: &egui::Context, strength: f32) {
    if strength <= 0.0 {
        return;
    }
    let screen = ctx.content_rect();
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("hurt")));
    let alpha = (strength.min(1.0) * 140.0) as u8;
    let edge = 90.0;
    for (i, a) in [(0, 1.0f32), (1, 0.6), (2, 0.3)] {
        let w = edge * (i as f32 + 1.0) / 3.0;
        let c = Color32::from_rgba_unmultiplied(200, 0, 0, (alpha as f32 * a) as u8);
        painter.rect_stroke(screen.shrink(w * 0.5), 0.0, Stroke::new(w, c), StrokeKind::Inside);
    }
}

/// Crosshair with an arc showing how far through the current bite you are.
pub fn crosshair(ctx: &egui::Context, progress: f32) {
    let center = ctx.content_rect().center();
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("crosshair")));
    painter.circle_stroke(center, 3.0, Stroke::new(1.5, Color32::from_white_alpha(200)));
    if progress > 0.0 {
        let radius = 13.0;
        painter.circle_stroke(center, radius, Stroke::new(3.0, Color32::from_black_alpha(90)));
        let segments = 48;
        let points: Vec<Pos2> = (0..=((segments as f32 * progress) as usize).max(1))
            .map(|i| {
                let a = -std::f32::consts::FRAC_PI_2 + i as f32 / segments as f32 * std::f32::consts::TAU;
                center + Vec2::new(a.cos(), a.sin()) * radius
            })
            .collect();
        painter.add(egui::Shape::line(points, Stroke::new(3.0, Color32::from_rgb(120, 220, 255))));
    }
}

/// Short notice above the hotbar ("Inventory full", ...).
pub fn notice(ctx: &egui::Context, text: &str) {
    let screen = ctx.content_rect();
    let pos = Pos2::new(screen.center().x, screen.bottom() - SLOT - 46.0);
    let painter = ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("notice")));
    painter.text(
        pos,
        Align2::CENTER_BOTTOM,
        text,
        FontId::proportional(16.0),
        Color32::from_rgb(255, 200, 120),
    );
}

/// The full inventory screen; `held` is the stack on the cursor. Returns what was clicked (stacks
/// move Minecraft-style when the op is applied).
pub fn window(
    ctx: &egui::Context,
    content: &Content,
    inventory: &Inventory,
    held: Option<Stack>,
) -> Option<InventoryOp> {
    let mut op = None;
    egui::Window::new("Inventory")
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .collapsible(false)
        .resizable(false)
        .show(ctx, |ui| {
            let shift = ui.input(|i| i.modifiers.shift);
            let mut slot_ui = |ui: &mut egui::Ui, i: usize| {
                let (rect, response) = ui.allocate_exact_size(Vec2::splat(SLOT), Sense::click());
                let painter = ui.painter();
                paint_slot_frame(painter, rect, response.hovered());
                if let Some(stack) = inventory.slots[i] {
                    paint_stack(painter, rect, content, stack);
                }
                if response.clicked() {
                    op = Some(if shift { InventoryOp::QuickMove(i) } else { InventoryOp::Click(i) });
                } else if response.secondary_clicked() {
                    op = Some(InventoryOp::RightClick(i));
                }
                if let Some(stack) = inventory.slots[i]
                    && held.is_none()
                {
                    response.on_hover_text(&content.item(stack.item).name);
                }
            };
            ui.spacing_mut().item_spacing = Vec2::splat(GAP);
            for row in 0..3 {
                ui.horizontal(|ui| {
                    for col in 0..HOTBAR {
                        slot_ui(ui, HOTBAR + row * HOTBAR + col);
                    }
                });
            }
            ui.add_space(10.0);
            ui.horizontal(|ui| {
                for i in 0..HOTBAR {
                    slot_ui(ui, i);
                }
            });
            debug_assert_eq!(HOTBAR * 4, SLOTS);
            ui.add_space(4.0);
            ui.label("Click: pick up / put down   Right click: split   Shift+click: move   E: close");
        });

    if let (Some(stack), Some(pointer)) = (held, ctx.pointer_latest_pos()) {
        let painter =
            ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("held stack")));
        paint_stack(&painter, Rect::from_center_size(pointer, Vec2::splat(SLOT)), content, stack);
    }
    op
}

/// What was picked in the build menu.
#[derive(Clone, Copy, Debug)]
pub enum MenuChoice {
    Piece(usize, asc_build::PartKind),
    Flatten,
}

/// The build menu: one preview tile per (tier, part), plus terrain tools.
pub fn build_menu(
    ctx: &egui::Context,
    spec: &asc_build::BuildingSpec,
    previews: &[[egui::TextureHandle; 3]],
) -> Option<MenuChoice> {
    let mut chosen = None;
    egui::Window::new("Build")
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .collapsible(false)
        .resizable(false)
        .show(ctx, |ui| {
            egui::Grid::new("build grid").spacing([8.0, 8.0]).show(ui, |ui| {
                ui.label("");
                for kind in asc_build::PartKind::ALL {
                    ui.label(&spec.part(kind).name);
                }
                ui.end_row();
                for (tier, row) in previews.iter().enumerate() {
                    ui.label(&spec.tiers[tier].name);
                    for (slot, kind) in asc_build::PartKind::ALL.into_iter().enumerate() {
                        let image = egui::Image::new(&row[slot]).fit_to_exact_size(Vec2::new(150.0, 112.0));
                        let tip = format!("{} {}", spec.tiers[tier].name, spec.part(kind).name);
                        if ui.add(egui::Button::image(image)).on_hover_text(tip).clicked() {
                            chosen = Some(MenuChoice::Piece(tier, kind));
                        }
                    }
                    ui.end_row();
                }
            });
            ui.separator();
            if ui.button("Flatten ground (2 x 2 m)").on_hover_text("Level the cell under the crosshair for a foundation").clicked() {
                chosen = Some(MenuChoice::Flatten);
            }
            ui.label("Pick a piece. Then LMB places, X dismantles, wheel changes tier, RMB stops building. B or Esc closes this.");
        });
    chosen
}

/// One line about the piece being placed, above the hotbar.
pub fn build_status(ctx: &egui::Context, text: &str, ok: bool) {
    let screen = ctx.content_rect();
    let pos = Pos2::new(screen.center().x, screen.bottom() - SLOT - 70.0);
    let painter =
        ctx.layer_painter(egui::LayerId::new(egui::Order::Foreground, egui::Id::new("build status")));
    let color = if ok { Color32::from_rgb(150, 220, 255) } else { Color32::from_rgb(255, 150, 120) };
    painter.text(
        pos + Vec2::splat(1.0),
        Align2::CENTER_BOTTOM,
        text,
        FontId::proportional(17.0),
        Color32::BLACK,
    );
    painter.text(pos, Align2::CENTER_BOTTOM, text, FontId::proportional(17.0), color);
}
