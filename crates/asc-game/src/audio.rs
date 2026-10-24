//! Sound: one-shot effects where things happen (footsteps, digging, bombs, blasts) and looping
//! ambience that follows the world round the listener (wind, rain, water, lava, fire, a volcano's
//! rumble and roar). The sounds themselves are made by tools/sounds.py into assets/sounds; free
//! library recordings to compare them with go in assets/sounds_library (tools/library_sounds.py).
//!
//! Everything in the world plays through one mixer track whose low-pass filter closes when the
//! listener is underwater. Sounds from far off arrive late, at the speed of sound (a blast's boom
//! lands as its shockwave does). Without an audio device the game runs silent.

use std::collections::HashMap;
use std::path::Path;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use glam::{Mat3, Quat, Vec2, Vec3};
use kira::effect::filter::{FilterBuilder, FilterHandle};
use kira::listener::ListenerHandle;
use kira::sound::static_sound::{StaticSoundData, StaticSoundHandle};
use kira::track::{SpatialTrackBuilder, SpatialTrackHandle, TrackBuilder, TrackHandle};
use kira::{AudioManager, AudioManagerSettings, Decibels, DefaultBackend, Tween};

pub const SPEED_OF_SOUND: f32 = 340.0;
/// How often the ambience is looked at again (it changes slowly; looking costs worldgen samples).
const AMBIENCE_EVERY: Duration = Duration::from_millis(250);
/// The low-pass cutoff in air and underwater (Hz).
const OPEN: f64 = 20_000.0;
const MUFFLED: f64 = 700.0;

/// Where a one-shot plays: its range (full volume within the first, silent beyond the second).
#[derive(Clone, Copy)]
pub struct Range(pub f32, pub f32);

/// The looping ambience, each its own sound and (for those with a place) track.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Ambience {
    Wind,
    Rain,
    River,
    Lake,
    Lava,
    Fire,
    Rumble,
    Roar,
    Underwater,
}

impl Ambience {
    const ALL: [Self; 9] =
        [Self::Wind, Self::Rain, Self::River, Self::Lake, Self::Lava, Self::Fire, Self::Rumble, Self::Roar, Self::Underwater];

    fn file(self) -> &'static str {
        match self {
            Self::Wind => "wind",
            Self::Rain => "rain",
            Self::River => "river",
            Self::Lake => "lake",
            Self::Lava => "lava",
            Self::Fire => "fire",
            Self::Rumble => "rumble",
            Self::Roar => "roar",
            Self::Underwater => "underwater",
        }
    }

    /// Where it's heard from: `None` all round the listener (wind, rain, being underwater), else
    /// its range.
    fn range(self) -> Option<Range> {
        match self {
            Self::Wind | Self::Rain | Self::Underwater => None,
            Self::River => Some(Range(4.0, 60.0)),
            Self::Lake => Some(Range(3.0, 40.0)),
            Self::Lava => Some(Range(4.0, 90.0)),
            Self::Fire => Some(Range(4.0, 120.0)),
            Self::Rumble => Some(Range(300.0, 9000.0)),
            Self::Roar => Some(Range(200.0, 6000.0)),
        }
    }
}

/// What the ambience should be doing now: for each, where (if it has a place) and how loud (0..1).
#[derive(Default)]
pub struct Scene {
    pub levels: HashMap<Ambience, (Option<Vec3>, f32)>,
    pub underwater: bool,
}

struct Loop {
    track: Option<SpatialTrackHandle>,
    sound: Option<StaticSoundHandle>,
    level: f32,
    /// What brings the playing set's recording to the generated one's loudness.
    gain: f32,
}

/// One set of sounds: each name's variations, and for each the gain that brings it to the
/// generated set's loudness (so a comparison is of the sound, not its level).
#[derive(Default)]
struct Bank {
    sounds: HashMap<String, Vec<StaticSoundData>>,
    gain: HashMap<String, f32>,
}

impl Bank {
    /// Every sound in `dir`: `<name>_<n>.<ext>` variations and `<name>.<ext>` loops (WAV, Ogg,
    /// MP3 or FLAC).
    fn load(dir: &Path) -> Result<Self> {
        let mut bank = Self::default();
        let mut files: Vec<_> = std::fs::read_dir(dir)
            .with_context(|| format!("reading {}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| ["wav", "ogg", "mp3", "flac"].contains(&&*x.to_string_lossy())))
            .collect();
        files.sort();
        for path in &files {
            let stem = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
            // `dig_soft_2` is the third variation of `dig_soft`; a loop has no number.
            let name = match stem.rsplit_once('_') {
                Some((base, n)) if n.parse::<u32>().is_ok() => base.to_owned(),
                _ => stem.clone(),
            };
            let data = StaticSoundData::from_file(path).with_context(|| format!("loading {}", path.display()))?;
            bank.sounds.entry(name).or_default().push(data);
        }
        Ok(bank)
    }

    /// How loud a sound is: a loop by its average (RMS), a one-shot by its peaks.
    fn loudness(&self, name: &str) -> Option<f32> {
        let variations = self.sounds.get(name)?;
        let looped = Ambience::ALL.iter().any(|a| a.file() == name);
        let each = variations.iter().map(|d| {
            let mono = d.frames.iter().map(|f| 0.5 * (f.left + f.right));
            if looped {
                (mono.map(|x| x * x).sum::<f32>() / d.frames.len().max(1) as f32).sqrt()
            } else {
                mono.fold(0.0f32, |m, x| m.max(x.abs()))
            }
        });
        Some(each.sum::<f32>() / variations.len().max(1) as f32)
    }
}

pub struct Audio {
    manager: AudioManager<DefaultBackend>,
    listener: ListenerHandle,
    /// The world's sounds, through the underwater filter.
    world: TrackHandle,
    muffle: FilterHandle,
    underwater: bool,
    /// The generated sounds, and recordings from free libraries to compare (F8 switches; a sound
    /// the libraries don't have stays generated).
    generated: Bank,
    library: Bank,
    use_library: bool,
    loops: HashMap<Ambience, Loop>,
    /// Sounds still travelling: when they arrive, which, where, its range and volume.
    travelling: Vec<(Instant, String, Vec3, Range, f32)>,
    at: Vec3,
    last_ambience: Option<Instant>,
    rng: u64,
}

fn db(level: f32) -> Decibels {
    if level <= 0.001 { Decibels::SILENCE } else { Decibels((20.0 * level.log10()).max(-60.0)) }
}

fn tween(seconds: f32) -> Tween {
    Tween { duration: Duration::from_secs_f32(seconds), ..Default::default() }
}

impl Audio {
    /// Open the audio device and load the generated sounds in `dir`, and the library ones in
    /// `library` if it's there.
    pub fn new(dir: &Path, library: &Path) -> Result<Self> {
        let mut manager = AudioManager::<DefaultBackend>::new(AudioManagerSettings {
            capacities: kira::Capacities { sub_track_capacity: 256, ..Default::default() },
            ..Default::default()
        })
        .context("opening the audio device")?;
        let listener = manager.add_listener(Vec3::ZERO, Quat::IDENTITY)?;
        let mut builder = TrackBuilder::new();
        let muffle = builder.add_effect(FilterBuilder::new().cutoff(OPEN));
        let world = manager.add_sub_track(builder)?;
        let generated = Bank::load(dir)?;
        let mut library = if library.is_dir() { Bank::load(library)? } else { Bank::default() };
        let names: Vec<String> = library.sounds.keys().cloned().collect();
        for name in names {
            let gain = match (generated.loudness(&name), library.loudness(&name)) {
                (Some(g), Some(l)) if l > 1e-5 => (g / l).clamp(0.05, 20.0),
                _ => 1.0,
            };
            library.gain.insert(name, gain);
        }
        log::info!("audio: {} generated sounds, {} from libraries", generated.sounds.len(), library.sounds.len());
        let mut audio = Self {
            manager,
            listener,
            world,
            muffle,
            underwater: false,
            generated,
            library,
            use_library: false,
            loops: HashMap::new(),
            travelling: Vec::new(),
            at: Vec3::ZERO,
            last_ambience: None,
            rng: 0x5eed_50f7,
        };
        audio.start_loops()?;
        Ok(audio)
    }

    /// Switch between the generated sounds and the library recordings (where there are any).
    pub fn toggle_library(&mut self) -> bool {
        self.use_library = !self.use_library && !self.library.sounds.is_empty();
        if let Err(err) = self.start_loops() {
            log::warn!("restarting the ambience: {err}");
        }
        log::info!("sounds: {}", if self.use_library { "library recordings" } else { "generated" });
        self.use_library
    }

    /// `name`'s variations from the set in use, and the gain to play them at.
    fn variations(&self, name: &str) -> Option<(&[StaticSoundData], f32)> {
        if self.use_library
            && let Some(v) = self.library.sounds.get(name)
        {
            return Some((v, self.library.gain.get(name).copied().unwrap_or(1.0)));
        }
        self.generated.sounds.get(name).map(|v| (v.as_slice(), 1.0))
    }

    /// (Re)start every loop from the set in use, at its current level.
    fn start_loops(&mut self) -> Result<()> {
        for a in Ambience::ALL {
            let Some((data, gain)) = self.variations(a.file()).and_then(|(v, g)| v.first().cloned().map(|d| (d, g))) else {
                log::warn!("no {} sound: that ambience is silent", a.file());
                continue;
            };
            let old = self.loops.remove(&a);
            let level = old.as_ref().map_or(0.0, |l| l.level);
            let mut track = match old {
                Some(Loop { track, sound, .. }) => {
                    if let Some(mut sound) = sound {
                        sound.stop(tween(0.3));
                    }
                    track
                }
                None => match a.range() {
                    Some(Range(near, far)) => Some(self.world.add_spatial_sub_track(
                        &self.listener,
                        Vec3::ZERO,
                        SpatialTrackBuilder::new().distances((near, far)),
                    )?),
                    None => None,
                },
            };
            let data = data.loop_region(..).volume(db(level * gain));
            let sound = match &mut track {
                Some(track) => track.play(data)?,
                None => self.world.play(data)?,
            };
            self.loops.insert(a, Loop { track, sound: Some(sound), level, gain });
        }
        Ok(())
    }

    fn random(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(0x1405_7B7E_F767_814F);
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Where the listener is and which way it faces (game yaw: forward is (cos, 0, sin)).
    pub fn listen(&mut self, at: Vec3, yaw: f32) {
        self.at = at;
        let right = Vec3::new(-yaw.sin(), 0.0, yaw.cos());
        let back = -Vec3::new(yaw.cos(), 0.0, yaw.sin());
        let orientation = Quat::from_mat3(&Mat3::from_cols(right, Vec3::Y, back));
        self.listener.set_position(at, Tween::default());
        self.listener.set_orientation(orientation, Tween::default());
        let now = Instant::now();
        let mut arrived = Vec::new();
        self.travelling.retain(|s| {
            let due = s.0 <= now;
            if due {
                arrived.push(s.clone());
            }
            !due
        });
        for (_, name, at, range, volume) in arrived {
            self.play_now(&name, at, range, volume);
        }
    }

    /// Play a variation of `name` at `at` (heard within `range`), `volume` 0..1, arriving after the
    /// time sound takes to get here.
    pub fn play_at(&mut self, name: &str, at: Vec3, range: Range, volume: f32) {
        let d = at.distance(self.at);
        if d > range.1 {
            return;
        }
        let delay = d / SPEED_OF_SOUND;
        if delay < 0.03 {
            self.play_now(name, at, range, volume);
        } else {
            let when = Instant::now() + Duration::from_secs_f32(delay);
            self.travelling.push((when, name.to_owned(), at, range, volume));
        }
    }

    /// Play a variation of `name` with no place (the player's own body: hurt, a blast's boom).
    pub fn play(&mut self, name: &str, volume: f32) {
        let Some(data) = self.pick(name, volume) else { return };
        if let Err(err) = self.world.play(data) {
            log::debug!("couldn't play {name}: {err}");
        }
    }

    /// A variation of `name` at `volume` (0..1), from the set in use.
    fn pick(&mut self, name: &str, volume: f32) -> Option<StaticSoundData> {
        let n = self.variations(name).map_or(0, |(v, _)| v.len());
        if n == 0 {
            log::warn!("no sound {name}");
            return None;
        }
        let i = ((self.random() * n as f32) as usize).min(n - 1);
        // A little higher or lower each time, so repeats don't sound like a machine gun.
        let rate = 1.0 + (self.random() as f64 - 0.5) * 0.12;
        let (v, gain) = self.variations(name)?;
        Some(v[i].playback_rate(rate).volume(db(volume * gain)))
    }

    fn play_now(&mut self, name: &str, at: Vec3, range: Range, volume: f32) {
        let Some(data) = self.pick(name, volume) else { return };
        let track = self.world.add_spatial_sub_track(
            &self.listener,
            at,
            SpatialTrackBuilder::new().distances((range.0, range.1)).persist_until_sounds_finish(true),
        );
        match track {
            Ok(mut track) => {
                if let Err(err) = track.play(data) {
                    log::debug!("couldn't play {name}: {err}");
                }
            }
            Err(err) => log::debug!("no track free for {name}: {err}"),
        }
    }

    /// Whether the ambience is due another look (`set_scene`).
    pub fn ambience_due(&mut self) -> bool {
        let now = Instant::now();
        if self.last_ambience.is_some_and(|t| now - t < AMBIENCE_EVERY) {
            return false;
        }
        self.last_ambience = Some(now);
        true
    }

    /// Move each loop to where it's heard from now and fade it to its level.
    pub fn set_scene(&mut self, scene: &Scene) {
        for (a, l) in &mut self.loops {
            let (at, level) = scene.levels.get(a).copied().unwrap_or((None, 0.0));
            if let (Some(track), Some(at)) = (&mut l.track, at) {
                track.set_position(at, tween(0.25));
            }
            if (level - l.level).abs() > 0.01 {
                if let Some(sound) = &mut l.sound {
                    sound.set_volume(db(level * l.gain), tween(0.5));
                }
                l.level = level;
            }
        }
        if scene.underwater != self.underwater {
            self.underwater = scene.underwater;
            self.muffle.set_cutoff(if scene.underwater { MUFFLED } else { OPEN }, tween(0.15));
        }
    }

    /// Everything at `volume` 0..1 (the main mix).
    pub fn set_volume(&mut self, volume: f32) {
        self.manager.main_track().set_volume(db(volume), tween(0.2));
    }
}

/// The sound of walking on (or digging) a material: soft ground, sand, snow, or hard stone.
pub fn surface(material: asc_world::Material) -> &'static str {
    use asc_world::Material::*;
    match material {
        Grass | Soil | DryGrass | Moss | Mud | Air => "grass",
        Sand | Ash | Pumice => "sand",
        Snow => "snow",
        Rock | Ore | Ice | Sandstone | Basalt | Obsidian | Sulfur => "hard",
    }
}

/// Bombs that were in the air last report and aren't now have landed: where (a bomb flies at most
/// `step` metres between reports, so one with none of the new ones near it is gone).
pub fn landed(before: &[(Vec3, u8)], now: &[(Vec3, u8)], step: f32) -> Vec<(Vec3, u8)> {
    before
        .iter()
        .filter(|(b, _)| !now.iter().any(|(n, _)| n.distance_squared(*b) < step * step))
        .copied()
        .collect()
}

/// Where something is heard from and how strongly (0..1).
pub type Spot = Option<(Vec3, f32)>;

/// The nearest natural water round `at` (looked at on a grid out to `reach`): flowing (a river)
/// and still (a lake or the sea), each with where and how strongly.
pub fn water_near(worldgen: &asc_world::WorldGen, at: Vec3, reach: f32) -> (Spot, Spot) {
    let (mut river, mut lake): (Spot, Spot) = (None, None);
    let step = reach / 4.0;
    for j in -4..=4 {
        for i in -4..=4 {
            let p = Vec2::new(at.x, at.z) + Vec2::new(i as f32, j as f32) * step;
            let Some(level) = asc_world::fluid::water_level(worldgen, p.x, p.y) else { continue };
            if worldgen.height(p.x, p.y) >= level {
                continue;
            }
            let here = Vec3::new(p.x, level, p.y);
            let d = here.distance(at);
            let flow = worldgen.water_flow(p.x, p.y).length();
            let (slot, strength) = if flow > 0.3 { (&mut river, (flow / 2.0).min(1.0)) } else { (&mut lake, 1.0) };
            if slot.is_none_or(|(s, _)| s.distance(at) > d) {
                *slot = Some((here, strength));
            }
        }
    }
    (river, lake)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bomb_gone_from_the_report_has_landed() {
        let before = [(Vec3::new(0.0, 10.0, 0.0), 1), (Vec3::new(50.0, 5.0, 0.0), 2)];
        // The first flew on 2 m; the second is gone.
        let now = [(Vec3::new(1.5, 9.0, 0.5), 1)];
        assert_eq!(landed(&before, &now, 4.0), vec![(Vec3::new(50.0, 5.0, 0.0), 2)]);
    }

    #[test]
    fn every_material_has_a_surface_sound() {
        for m in asc_world::Material::ALL {
            assert!(["grass", "sand", "snow", "hard"].contains(&surface(m)));
        }
    }
}
