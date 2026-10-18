//! Hotbar, inventory screen and mining progress ring (egui).

use egui::{Align2, Color32, FontId, Pos2, Rect, Sense, Stroke, StrokeKind, Vec2};

use crate::content::Content;
use crate::inventory::{HOTBAR, Inventory, SLOTS, Stack};

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

/// The full inventory screen. Clicks move stacks Minecraft-style; `held` is the stack on the cursor.
pub fn window(ctx: &egui::Context, content: &Content, inventory: &mut Inventory, held: &mut Option<Stack>) {
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
                    if shift {
                        inventory.quick_move(content, i);
                    } else {
                        inventory.click(content, i, held);
                    }
                } else if response.secondary_clicked() {
                    inventory.right_click(content, i, held);
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

    if let (Some(stack), Some(pointer)) = (*held, ctx.pointer_latest_pos()) {
        let painter =
            ctx.layer_painter(egui::LayerId::new(egui::Order::Tooltip, egui::Id::new("held stack")));
        paint_stack(&painter, Rect::from_center_size(pointer, Vec2::splat(SLOT)), content, stack);
    }
}
