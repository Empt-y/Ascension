//! Skeletons and their animations, loaded from glTF: what a skinned model's joints are, clips of
//! keyframes for them, and posing a skeleton from a clip (or a blend of two) into the joint
//! matrices the skinned pipeline draws with.
//!
//! Clips are keyed by joint name, not index, so one file's animations drive any skeleton whose
//! joints share their names: the Universal Animation Library plays on every base character, and
//! a character's hair (skinned to the same skeleton) moves with its head.

use glam::{Mat4, Quat, Vec3};

/// A joint's local transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    pub scale: Vec3,
}

impl Transform {
    pub const IDENTITY: Self = Self { translation: Vec3::ZERO, rotation: Quat::IDENTITY, scale: Vec3::ONE };

    pub fn matrix(&self) -> Mat4 {
        Mat4::from_scale_rotation_translation(self.scale, self.rotation, self.translation)
    }

    fn lerp(&self, other: &Self, t: f32) -> Self {
        Self {
            translation: self.translation.lerp(other.translation, t),
            rotation: self.rotation.slerp(other.rotation, t),
            scale: self.scale.lerp(other.scale, t),
        }
    }
}

/// A skinned model's joints.
#[derive(Clone, Debug)]
pub struct Skeleton {
    pub names: Vec<String>,
    /// Each joint's parent joint, or `None` for a root.
    pub parents: Vec<Option<usize>>,
    /// For a root joint, the world matrix of the (non-joint) nodes above it, such as an armature's
    /// own scale; identity for the rest.
    pub above: Vec<Mat4>,
    pub inverse_bind: Vec<Mat4>,
    /// The joints' local transforms at rest (the pose with no clip playing).
    pub rest: Vec<Transform>,
    /// Joint indices with every parent before its children (a skin needn't list them so).
    pub order: Vec<usize>,
}

/// One joint's keyframes for one property.
#[derive(Clone, Debug)]
pub enum Track {
    Translation(Vec<Vec3>),
    Rotation(Vec<Quat>),
    Scale(Vec<Vec3>),
}

#[derive(Clone, Debug)]
pub struct Channel {
    pub joint: String,
    pub times: Vec<f32>,
    pub track: Track,
    /// Hold each key until the next (glTF STEP) rather than interpolate.
    pub step: bool,
}

/// A named animation.
#[derive(Clone, Debug)]
pub struct Clip {
    pub name: String,
    pub duration: f32,
    pub channels: Vec<Channel>,
}

/// A clip's channels resolved against one skeleton: the joint each drives (`None`: no such joint).
#[derive(Clone, Debug)]
pub struct Binding(Vec<Option<usize>>);

pub type Pose = Vec<Transform>;

impl Skeleton {
    pub fn len(&self) -> usize {
        self.names.len()
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    pub fn index(&self, name: &str) -> Option<usize> {
        self.names.iter().position(|n| n == name)
    }

    /// Which of this skeleton's joints each of `clip`'s channels drives.
    pub fn bind(&self, clip: &Clip) -> Binding {
        Binding(clip.channels.iter().map(|c| self.index(&c.joint)).collect())
    }

    pub fn rest_pose(&self) -> Pose {
        self.rest.clone()
    }

    /// `clip` at `t` seconds (wrapped round if `looped`, held at the end if not), over the rest
    /// pose for any joint it doesn't move.
    pub fn sample(&self, clip: &Clip, binding: &Binding, t: f32, looped: bool, pose: &mut Pose) {
        pose.clone_from(&self.rest);
        let t = if looped && clip.duration > 0.0 { t.rem_euclid(clip.duration) } else { t.clamp(0.0, clip.duration) };
        for (channel, joint) in clip.channels.iter().zip(&binding.0) {
            let Some(j) = *joint else { continue };
            let (i, f) = key_at(&channel.times, t, channel.step);
            let p = &mut pose[j];
            match &channel.track {
                Track::Translation(v) => p.translation = v[i].lerp(v[(i + 1).min(v.len() - 1)], f),
                Track::Rotation(v) => p.rotation = v[i].slerp(v[(i + 1).min(v.len() - 1)], f).normalize(),
                Track::Scale(v) => p.scale = v[i].lerp(v[(i + 1).min(v.len() - 1)], f),
            }
        }
    }

    /// Each joint's model-space transform in `pose`.
    pub fn globals(&self, pose: &Pose, out: &mut Vec<Mat4>) {
        out.clear();
        out.resize(self.len(), Mat4::IDENTITY);
        for &j in &self.order {
            let local = pose[j].matrix();
            out[j] = match self.parents[j] {
                Some(p) => out[p] * local,
                None => self.above[j] * local,
            };
        }
    }

    /// The matrices skinning needs: each joint's model-space transform times its inverse bind
    /// matrix.
    pub fn palette(&self, pose: &Pose, out: &mut Vec<Mat4>) {
        let mut global = Vec::new();
        self.globals(pose, &mut global);
        out.clear();
        out.extend(global.iter().zip(&self.inverse_bind).map(|(g, ib)| *g * *ib));
    }

    /// Which joints are `joint` or below it (a mask for blending part of a body: the arms and
    /// chest from `spine_02` up, say).
    pub fn below(&self, joint: usize) -> Vec<bool> {
        let mut mask = vec![false; self.len()];
        mask[joint] = true;
        for &j in &self.order {
            if let Some(p) = self.parents[j]
                && mask[p]
            {
                mask[j] = true;
            }
        }
        mask
    }
}

/// The key at or before `t` and how far towards the next one `t` is.
fn key_at(times: &[f32], t: f32, step: bool) -> (usize, f32) {
    if times.len() < 2 || t <= times[0] {
        return (0, 0.0);
    }
    let i = times.partition_point(|&k| k <= t).saturating_sub(1).min(times.len() - 2);
    if t >= times[times.len() - 1] {
        return (times.len() - 1, 0.0);
    }
    let span = times[i + 1] - times[i];
    let f = if step || span <= 0.0 { 0.0 } else { (t - times[i]) / span };
    (i, f.clamp(0.0, 1.0))
}

/// `a` blended towards `b` by `w` (0 all `a`, 1 all `b`), into `out`.
pub fn blend(a: &Pose, b: &Pose, w: f32, out: &mut Pose) {
    out.clear();
    out.extend(a.iter().zip(b).map(|(x, y)| x.lerp(y, w)));
}

/// `a`, with the joints in `mask` blended towards `b` by `w` (0 all `a`, 1 all `b`).
pub fn blend_masked(a: &Pose, b: &Pose, w: f32, mask: &[bool], out: &mut Pose) {
    out.clear();
    out.extend(a.iter().zip(b).zip(mask).map(|((x, y), m)| if *m { x.lerp(y, w) } else { *x }));
}

/// Every node's world matrix, by node index.
fn world_matrices(doc: &gltf::Document) -> Vec<Mat4> {
    let mut world = vec![Mat4::IDENTITY; doc.nodes().len()];
    let scene = doc.default_scene().or_else(|| doc.scenes().next());
    let mut stack: Vec<(gltf::Node, Mat4)> = scene.map(|s| s.nodes().map(|n| (n, Mat4::IDENTITY)).collect()).unwrap_or_default();
    while let Some((node, parent)) = stack.pop() {
        let m = parent * Mat4::from_cols_array_2d(&node.transform().matrix());
        world[node.index()] = m;
        stack.extend(node.children().map(|c| (c, m)));
    }
    world
}

/// The skeleton of a glTF skin: its joints in the skin's order (which is what vertices' joint
/// indices refer to), each joint's parent among them, and the transforms above the roots.
pub fn skeleton_of(doc: &gltf::Document, skin: &gltf::Skin, buffers: &[gltf::buffer::Data]) -> Skeleton {
    let world = world_matrices(doc);
    let joints: Vec<gltf::Node> = skin.joints().collect();
    let index_of = |n: usize| joints.iter().position(|j| j.index() == n);
    // Each node's parent (glTF only stores children).
    let mut parent_of = vec![None; doc.nodes().len()];
    for n in doc.nodes() {
        for c in n.children() {
            parent_of[c.index()] = Some(n.index());
        }
    }
    let reader = skin.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
    let inverse_bind: Vec<Mat4> = match reader.read_inverse_bind_matrices() {
        Some(m) => m.map(|c| Mat4::from_cols_array_2d(&c)).collect(),
        None => vec![Mat4::IDENTITY; joints.len()],
    };
    let mut names = Vec::new();
    let mut parents = Vec::new();
    let mut above = Vec::new();
    let mut rest = Vec::new();
    for j in &joints {
        names.push(j.name().unwrap_or("").to_owned());
        let parent = parent_of[j.index()];
        let joint_parent = parent.and_then(index_of);
        parents.push(joint_parent);
        above.push(match (joint_parent, parent) {
            (None, Some(p)) => world[p],
            _ => Mat4::IDENTITY,
        });
        let (t, r, s) = j.transform().decomposed();
        rest.push(Transform { translation: Vec3::from(t), rotation: Quat::from_array(r), scale: Vec3::from(s) });
    }
    let order = parents_first(&parents);
    Skeleton { names, parents, above, inverse_bind, rest, order }
}

/// Joint indices ordered so each comes after its parent.
fn parents_first(parents: &[Option<usize>]) -> Vec<usize> {
    let depth = |mut j: usize| {
        let mut d = 0;
        while let Some(p) = parents[j] {
            j = p;
            d += 1;
            if d > parents.len() {
                break;
            }
        }
        d
    };
    let mut order: Vec<usize> = (0..parents.len()).collect();
    order.sort_by_key(|&j| depth(j));
    order
}

/// Every animation in a glTF file, keyed by the names of the nodes it moves. A clip named
/// `Armature|Walk` is called `Walk`; where two clips end up with one name, the first is kept.
pub fn clips_of(doc: &gltf::Document, buffers: &[gltf::buffer::Data]) -> Vec<Clip> {
    let mut clips: Vec<Clip> = Vec::new();
    for anim in doc.animations() {
        let name = anim.name().unwrap_or("").rsplit('|').next().unwrap_or("").to_owned();
        if clips.iter().any(|c| c.name == name) {
            continue;
        }
        let mut channels = Vec::new();
        let mut duration = 0.0f32;
        for ch in anim.channels() {
            let reader = ch.reader(|b| buffers.get(b.index()).map(|d| &d.0[..]));
            let Some(times) = reader.read_inputs().map(|i| i.collect::<Vec<f32>>()) else { continue };
            let Some(outputs) = reader.read_outputs() else { continue };
            let cubic = ch.sampler().interpolation() == gltf::animation::Interpolation::CubicSpline;
            // Cubic splines store (in-tangent, value, out-tangent) per key: keep the values.
            let values = |n: usize| -> Vec<usize> { if cubic { (0..n).map(|i| i * 3 + 1).collect() } else { (0..n).collect() } };
            use gltf::animation::util::ReadOutputs;
            let track = match outputs {
                ReadOutputs::Translations(v) => {
                    let v: Vec<Vec3> = v.map(Vec3::from).collect();
                    Track::Translation(values(times.len()).into_iter().map(|i| v[i]).collect())
                }
                ReadOutputs::Rotations(v) => {
                    let v: Vec<Quat> = v.into_f32().map(Quat::from_array).collect();
                    Track::Rotation(values(times.len()).into_iter().map(|i| v[i]).collect())
                }
                ReadOutputs::Scales(v) => {
                    let v: Vec<Vec3> = v.map(Vec3::from).collect();
                    Track::Scale(values(times.len()).into_iter().map(|i| v[i]).collect())
                }
                ReadOutputs::MorphTargetWeights(_) => continue,
            };
            duration = duration.max(times.last().copied().unwrap_or(0.0));
            channels.push(Channel {
                joint: ch.target().node().name().unwrap_or("").to_owned(),
                times,
                track,
                step: ch.sampler().interpolation() == gltf::animation::Interpolation::Step,
            });
        }
        clips.push(Clip { name, duration, channels });
    }
    clips
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_joints() -> Skeleton {
        let rest = vec![
            Transform::IDENTITY,
            Transform { translation: Vec3::new(0.0, 1.0, 0.0), ..Transform::IDENTITY },
        ];
        // Bind pose = rest pose: the inverse binds undo the rest globals.
        let inverse_bind = vec![Mat4::IDENTITY, Mat4::from_translation(Vec3::new(0.0, -1.0, 0.0))];
        Skeleton {
            names: vec!["root".into(), "child".into()],
            parents: vec![None, Some(0)],
            above: vec![Mat4::IDENTITY; 2],
            inverse_bind,
            rest,
            order: vec![0, 1],
        }
    }

    #[test]
    fn the_rest_pose_leaves_the_mesh_as_it_was_bound() {
        let s = two_joints();
        let mut palette = Vec::new();
        s.palette(&s.rest_pose(), &mut palette);
        for m in palette {
            assert!(m.abs_diff_eq(Mat4::IDENTITY, 1e-6), "{m}");
        }
    }

    #[test]
    fn a_clip_hits_its_keys_and_blends_between_them() {
        let s = two_joints();
        let clip = Clip {
            name: "turn".into(),
            duration: 1.0,
            channels: vec![Channel {
                joint: "child".into(),
                times: vec![0.0, 1.0],
                track: Track::Rotation(vec![Quat::IDENTITY, Quat::from_rotation_z(std::f32::consts::FRAC_PI_2)]),
                step: false,
            }],
        };
        let binding = s.bind(&clip);
        let mut pose = s.rest_pose();
        s.sample(&clip, &binding, 1.0, false, &mut pose);
        assert!(pose[1].rotation.abs_diff_eq(Quat::from_rotation_z(std::f32::consts::FRAC_PI_2), 1e-5));
        s.sample(&clip, &binding, 0.5, false, &mut pose);
        assert!(pose[1].rotation.abs_diff_eq(Quat::from_rotation_z(std::f32::consts::FRAC_PI_4), 1e-5));
        // Unmoved joints keep their rest transform; channels for missing joints are ignored.
        assert_eq!(pose[0], Transform::IDENTITY);
        let other = Clip { channels: vec![Channel { joint: "tail".into(), ..clip.channels[0].clone() }], ..clip };
        assert!(s.bind(&other).0[0].is_none());
    }

    #[test]
    fn looping_wraps_and_one_shots_hold_their_end() {
        assert_eq!(key_at(&[0.0, 1.0, 2.0], 1.5, false), (1, 0.5));
        assert_eq!(key_at(&[0.0, 1.0, 2.0], 1.5, true), (1, 0.0));
        assert_eq!(key_at(&[0.0, 1.0, 2.0], 5.0, false), (2, 0.0));
    }
}

#[cfg(test)]
mod layer_tests {
    use super::*;

    #[test]
    fn a_masked_blend_moves_only_the_masked_joints() {
        // A chain: 0 (hips) -> 1 (chest) -> 2 (arm), and 3 (leg) off the hips.
        let s = Skeleton {
            names: ["hips", "chest", "arm", "leg"].map(String::from).to_vec(),
            parents: vec![None, Some(0), Some(1), Some(0)],
            above: vec![Mat4::IDENTITY; 4],
            inverse_bind: vec![Mat4::IDENTITY; 4],
            rest: vec![Transform::IDENTITY; 4],
            order: vec![0, 1, 2, 3],
        };
        let mask = s.below(1);
        assert_eq!(mask, vec![false, true, true, false]);
        let a = s.rest_pose();
        let b: Pose = a.iter().map(|t| Transform { translation: Vec3::Y, ..*t }).collect();
        let mut out = Vec::new();
        blend_masked(&a, &b, 1.0, &mask, &mut out);
        assert_eq!(out[1].translation, Vec3::Y);
        assert_eq!(out[2].translation, Vec3::Y);
        assert_eq!(out[3].translation, Vec3::ZERO, "the leg stays");
        assert_eq!(out[0].translation, Vec3::ZERO);
    }
}
