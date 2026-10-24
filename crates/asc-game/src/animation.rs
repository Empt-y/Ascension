//! Posing skinned models for drawing: a clip (or a crossfade from one clip to another) sampled on
//! an `Animated` model's skeleton, into the joint palette the renderer wants; optionally with a
//! second clip over part of the body (a tool held up while the legs walk on).

use std::sync::Arc;

use asc_render::skeleton::{Pose, blend, blend_masked};
use glam::Mat4;

use crate::assets::Animated;

/// One clip playing: which (of the model's), how far in (seconds), and whether it loops.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Playing {
    pub clip: usize,
    pub time: f32,
    pub looped: bool,
}

/// A clip over part of the body: what plays (blended with a second by a weight, as a crossfade or
/// an aim between up and down), how much of it shows (0..1), and on which joints.
pub struct Layer<'a> {
    pub now: Playing,
    pub with: Option<(Playing, f32)>,
    pub weight: f32,
    pub mask: &'a [bool],
}

/// Scratch space, reused from model to model.
#[derive(Default)]
pub struct Poser {
    a: Pose,
    b: Pose,
    mixed: Pose,
    layer: Pose,
    layer_b: Pose,
    layer_mixed: Pose,
    out: Pose,
    palette: Vec<Mat4>,
}

impl Poser {
    /// `now`'s pose, blended in from `from` by `weight` (0 all `from`, 1 all `now`), if given.
    pub fn palette(&mut self, model: &Animated, now: Playing, from: Option<(Playing, f32)>) -> Arc<[Mat4]> {
        self.layered(model, now, from, None)
    }

    /// As `palette`, with `layer` over its joints.
    pub fn layered(&mut self, model: &Animated, now: Playing, from: Option<(Playing, f32)>, layer: Option<Layer>) -> Arc<[Mat4]> {
        let s = &model.skeleton;
        let (clip, binding) = &model.clips[now.clip];
        s.sample(clip, binding, now.time, now.looped, &mut self.a);
        let base = match from {
            Some((was, weight)) if weight < 1.0 => {
                let (clip, binding) = &model.clips[was.clip];
                s.sample(clip, binding, was.time, was.looped, &mut self.b);
                blend(&self.b, &self.a, weight.clamp(0.0, 1.0), &mut self.mixed);
                &self.mixed
            }
            _ => &self.a,
        };
        let pose = match layer {
            Some(l) if l.weight > 0.0 => {
                let (clip, binding) = &model.clips[l.now.clip];
                s.sample(clip, binding, l.now.time, l.now.looped, &mut self.layer);
                let over = match l.with {
                    Some((other, w)) if w > 0.0 => {
                        let (clip, binding) = &model.clips[other.clip];
                        s.sample(clip, binding, other.time, other.looped, &mut self.layer_b);
                        blend(&self.layer, &self.layer_b, w.clamp(0.0, 1.0), &mut self.layer_mixed);
                        &self.layer_mixed
                    }
                    _ => &self.layer,
                };
                blend_masked(base, over, l.weight.clamp(0.0, 1.0), l.mask, &mut self.out);
                &self.out
            }
            _ => base,
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
