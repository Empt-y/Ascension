//! Keyboard and mouse state, collected from winit events and read once per frame.

use ahash::AHashSet;
use glam::Vec2;
use winit::event::{ElementState, MouseButton, MouseScrollDelta};
use winit::keyboard::KeyCode;

#[derive(Default)]
pub struct Input {
    held: AHashSet<KeyCode>,
    pressed: AHashSet<KeyCode>,
    buttons_held: AHashSet<MouseButton>,
    buttons_pressed: AHashSet<MouseButton>,
    /// Raw mouse motion since the last frame.
    pub mouse_delta: Vec2,
    /// Wheel notches since the last frame (positive = away from the user).
    pub wheel: f32,
}

impl Input {
    pub fn key(&mut self, code: KeyCode, state: ElementState, repeat: bool) {
        match state {
            ElementState::Pressed => {
                if !repeat {
                    self.pressed.insert(code);
                }
                self.held.insert(code);
            }
            ElementState::Released => {
                self.held.remove(&code);
            }
        }
    }

    pub fn button(&mut self, button: MouseButton, state: ElementState) {
        match state {
            ElementState::Pressed => {
                self.buttons_pressed.insert(button);
                self.buttons_held.insert(button);
            }
            ElementState::Released => {
                self.buttons_held.remove(&button);
            }
        }
    }

    pub fn scroll(&mut self, delta: MouseScrollDelta) {
        self.wheel += match delta {
            MouseScrollDelta::LineDelta(_, y) => y,
            MouseScrollDelta::PixelDelta(p) => p.y as f32 / 40.0,
        };
    }

    pub fn held(&self, code: KeyCode) -> bool {
        self.held.contains(&code)
    }

    pub fn pressed(&self, code: KeyCode) -> bool {
        self.pressed.contains(&code)
    }

    pub fn button_held(&self, button: MouseButton) -> bool {
        self.buttons_held.contains(&button)
    }

    pub fn button_pressed(&self, button: MouseButton) -> bool {
        self.buttons_pressed.contains(&button)
    }

    /// Forget everything, e.g. when the window loses focus and releases would be missed.
    pub fn clear(&mut self) {
        *self = Self::default();
    }

    /// Call at the end of each frame.
    pub fn end_frame(&mut self) {
        self.pressed.clear();
        self.buttons_pressed.clear();
        self.mouse_delta = Vec2::ZERO;
        self.wheel = 0.0;
    }
}
