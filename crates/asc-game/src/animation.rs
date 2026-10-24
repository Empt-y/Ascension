//! Posing skinned models for drawing: a clip (or a crossfade from one clip to another) sampled on
//! an `Animated` model's skeleton, into the joint palette the renderer wants.

use std::sync::Arc;

use asc_render::skeleton::{Pose, blend};
use glam::Mat4;

use crate::assets::Animated;

/// One clip playing: which (of the model's), how far in (seconds), and whether it loops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Playing {
    pub clip: usize,
    pub time: f32,
    pub looped: bool,
}

/// Scratch space, reused from model to model.
#[derive(Default)]
pub struct Poser {
    a: Pose,
    b: Pose,
    mixed: Pose,
    palette: Vec<Mat4>,
}

impl Poser {
    /// `now`'s pose, blended in from `from` by `weight` (0 all `from`, 1 all `now`), if given.
    pub fn palette(&mut self, model: &Animated, now: Playing, from: Option<(Playing, f32)>) -> Arc<[Mat4]> {
        let s = &model.skeleton;
        let (clip, binding) = &model.clips[now.clip];
        s.sample(clip, binding, now.time, now.looped, &mut self.a);
        let pose = match from {
            Some((was, weight)) if weight < 1.0 => {
                let (clip, binding) = &model.clips[was.clip];
                s.sample(clip, binding, was.time, was.looped, &mut self.b);
                blend(&self.b, &self.a, weight.clamp(0.0, 1.0), &mut self.mixed);
                &self.mixed
            }
            _ => &self.a,
        };
        s.palette(pose, &mut self.palette);
        Arc::from(self.palette.as_slice())
    }

    /// The model at rest (no clips).
    pub fn rest(&mut self, model: &Animated) -> Arc<[Mat4]> {
        model.skeleton.palette(&model.skeleton.rest_pose(), &mut self.palette);
        Arc::from(self.palette.as_slice())
    }
}
