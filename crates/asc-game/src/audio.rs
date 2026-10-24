//! Sound: one-shot effects where things happen (footsteps, digging, bombs, blasts, steam) and
//! looping ambience that follows the world round the listener (wind, rain, storms, water, lava,
//! fire, a volcano's rumble and roar, and what a place sounds like by biome and time of day). The
//! generated sounds are made by tools/sounds.py into assets/sounds; free library recordings, used
//! where there are any, go in assets/sounds_library (tools/library_sounds.py).
//!
//! Everything in the world plays through one mixer track whose low-pass filter closes when the
//! listener is underwater, and which sends to a reverb heard in caves. Sounds from far off arrive
//! late, at the speed of sound (a blast's boom lands as its shockwave does). Loops stream from
//! their files (the ambience recordings run a minute or more each) and pause while silent.
//! Without an audio device the game runs silent.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use glam::{Mat3, Quat, Vec2, Vec3};
use kira::effect::eq_filter::{EqFilterBuilder, EqFilterKind};
use kira::effect::filter::{FilterBuilder, FilterHandle};
use kira::effect::reverb::ReverbBuilder;
use kira::listener::ListenerHandle;
use kira::sound::FromFileError;
use kira::sound::static_sound::StaticSoundData;
use kira::sound::streaming::{StreamingSoundData, StreamingSoundHandle};
use kira::track::{SendTrackBuilder, SendTrackHandle, SpatialTrackBuilder, SpatialTrackHandle, TrackBuilder, TrackHandle};
use kira::{AudioManager, AudioManagerSettings, Decibels, DefaultBackend, Mix, Tween};

pub const SPEED_OF_SOUND: f32 = 340.0;
/// How often the ambience is looked at again (it changes slowly; looking costs worldgen samples).
const AMBIENCE_EVERY: Duration = Duration::from_millis(250);
/// The low-pass cutoff in air and underwater (Hz).
const OPEN: f64 = 20_000.0;
const MUFFLED: f64 = 700.0;
/// The footsteps' bass boost: a low shelf at this frequency (Hz), this many dB.
const BODY_SHELF: (f64, f32) = (200.0, 3.0);
/// How loud the cave reverb gets, fully enclosed.
const CAVE_REVERB: f32 = 0.7;

/// Where a one-shot plays: its range (full volume within the first, silent beyond the second).
#[derive(Clone, Copy)]
pub struct Range(pub f32, pub f32);

/// The looping ambience, each its own sound and (for those with a place) track.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Ambience {
    Wind,
    Rain,
    Storm,
    River,
    Lake,
    Ocean,
    Waterfall,
    Lava,
    LavaFlow,
    Fire,
    Campfire,
    Rumble,
    Roar,
    Crater,
    Underwater,
    Cave,
    ForestDay,
    ForestNight,
    Pine,
    Rainforest,
    Clearing,
    Evening,
    Swamp,
    DesertWind,
    WinterWind,
}

impl Ambience {
    pub const ALL: [Self; 25] = [
        Self::Wind,
        Self::Rain,
        Self::Storm,
        Self::River,
        Self::Lake,
        Self::Ocean,
        Self::Waterfall,
        Self::Lava,
        Self::LavaFlow,
        Self::Fire,
        Self::Campfire,
        Self::Rumble,
        Self::Roar,
        Self::Crater,
        Self::Underwater,
        Self::Cave,
        Self::ForestDay,
        Self::ForestNight,
        Self::Pine,
        Self::Rainforest,
        Self::Clearing,
        Self::Evening,
        Self::Swamp,
        Self::DesertWind,
        Self::WinterWind,
    ];

    fn file(self) -> &'static str {
        match self {
            Self::Wind => "wind",
            Self::Rain => "rain",
            Self::Storm => "storm",
            Self::River => "river",
            Self::Lake => "lake",
            Self::Ocean => "ocean",
            Self::Waterfall => "waterfall",
            Self::Lava => "lava",
            Self::LavaFlow => "lava_flow",
            Self::Fire => "fire",
            Self::Campfire => "campfire",
            Self::Rumble => "rumble",
            Self::Roar => "roar",
            Self::Crater => "crater",
            Self::Underwater => "underwater",
            Self::Cave => "cave",
            Self::ForestDay => "forest_day",
            Self::ForestNight => "forest_night",
            Self::Pine => "pine",
            Self::Rainforest => "rainforest",
            Self::Clearing => "clearing",
            Self::Evening => "evening",
            Self::Swamp => "swamp",
            Self::DesertWind => "desert_wind",
            Self::WinterWind => "winter_wind",
        }
    }

    /// Where it's heard from: `None` all round the listener (weather, being underwater or in a
    /// cave, what a place sounds like), else its range.
    fn range(self) -> Option<Range> {
        match self {
            Self::River => Some(Range(4.0, 60.0)),
            Self::Lake => Some(Range(3.0, 40.0)),
            Self::Ocean => Some(Range(6.0, 160.0)),
            Self::Waterfall => Some(Range(6.0, 160.0)),
            Self::Lava => Some(Range(4.0, 90.0)),
            Self::LavaFlow => Some(Range(4.0, 90.0)),
            Self::Fire => Some(Range(4.0, 120.0)),
            Self::Campfire => Some(Range(3.0, 50.0)),
            Self::Rumble => Some(Range(300.0, 9000.0)),
            Self::Roar => Some(Range(200.0, 6000.0)),
            _ => None,
        }
    }
}

/// What the ambience should be doing now: for each, where (if it has a place) and how loud (0..1);
/// whether the listener is underwater, and how enclosed (0 open air .. 1 deep in a cave).
#[derive(Default)]
pub struct Scene {
    pub levels: HashMap<Ambience, (Option<Vec3>, f32)>,
    pub underwater: bool,
    pub enclosed: f32,
}

struct Loop {
    track: Option<SpatialTrackHandle>,
    sound: Option<StreamingSoundHandle<FromFileError>>,
    level: f32,
    /// Paused while silent (a playing loop keeps decoding).
    paused: bool,
    /// What brings the playing set's recording to the generated one's loudness.
    gain: f32,
}

/// One set of sounds: each one-shot's variations (in memory), each loop's file (streamed), and
/// the gain that brings a sound to the generated set's loudness (so the two compare as sounds,
/// not levels).
#[derive(Default)]
struct Bank {
    sounds: HashMap<String, Vec<StaticSoundData>>,
    loops: HashMap<String, PathBuf>,
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
        for path in files {
            let stem = path.file_stem().unwrap_or_default().to_string_lossy().into_owned();
            if Ambience::ALL.iter().any(|a| a.file() == stem) {
                bank.loops.insert(stem, path);
                continue;
            }
            // `dig_soft_2` is the third variation of `dig_soft`.
            let name = match stem.rsplit_once('_') {
                Some((base, n)) if n.parse::<u32>().is_ok() => base.to_owned(),
                _ => stem.clone(),
            };
            let data = StaticSoundData::from_file(&path).with_context(|| format!("loading {}", path.display()))?;
            bank.sounds.entry(name).or_default().push(data);
        }
        Ok(bank)
    }

    /// How loud a sound is: a one-shot by its peaks, a loop by its average (RMS: decoded once
    /// to measure it).
    fn loudness(&self, name: &str) -> Option<f32> {
        if let Some(path) = self.loops.get(name) {
            let d = StaticSoundData::from_file(path).ok()?;
            let sum: f32 = d.frames.iter().map(|f| (0.5 * (f.left + f.right)).powi(2)).sum();
            return Some((sum / d.frames.len().max(1) as f32).sqrt());
        }
        let variations = self.sounds.get(name)?;
        let peaks = variations.iter().map(|d| d.frames.iter().fold(0.0f32, |m, f| m.max((0.5 * (f.left + f.right)).abs())));
        Some(peaks.sum::<f32>() / variations.len().max(1) as f32)
    }
}

pub struct Audio {
    manager: AudioManager<DefaultBackend>,
    listener: ListenerHandle,
    /// The world's sounds, through the underwater filter; the player's footsteps and landings
    /// through a further bass boost (they sounded thin; a soft low thump is layered under them too).
    world: TrackHandle,
    body: TrackHandle,
    muffle: FilterHandle,
    underwater: bool,
    /// The reverb everything in the world sends to, turned up in caves.
    reverb: SendTrackHandle,
    enclosed: f32,
    /// The generated sounds, and recordings from free libraries (used where there are any; F8
    /// switches to the generated set).
    generated: Bank,
    library: Bank,
    use_library: bool,
    loops: HashMap<Ambience, Loop>,
    /// Sounds still travelling: when they arrive, which, where, its range and volume.
    travelling: Vec<(Instant, String, Vec3, Range, f32)>,
    at: Vec3,
    last_ambience: Option<Instant>,
    rng: u64,
    /// Sounds asked for that neither set has (warned about once).
    missing: std::collections::HashSet<String>,
}

fn db(level: f32) -> Decibels {
    if level <= 0.001 { Decibels::SILENCE } else { Decibels((20.0 * level.log10()).max(-60.0)) }
}

fn tween(seconds: f32) -> Tween {
    Tween { duration: Duration::from_secs_f32(seconds), ..Default::default() }
}

impl Audio {
    /// Open the audio device and load the generated sounds in `dir`, and the library ones in
    /// `library_dir` if it's there.
    pub fn new(dir: &Path, library_dir: &Path) -> Result<Self> {
        let mut manager = AudioManager::<DefaultBackend>::new(AudioManagerSettings {
            capacities: kira::Capacities { sub_track_capacity: 256, ..Default::default() },
            ..Default::default()
        })
        .context("opening the audio device")?;
        let listener = manager.add_listener(Vec3::ZERO, Quat::IDENTITY)?;
        let reverb = manager.add_send_track(
            SendTrackBuilder::new()
                .volume(Decibels::SILENCE)
                .with_effect(ReverbBuilder::new().feedback(0.85).damping(0.4).mix(Mix::WET)),
        )?;
        let mut builder = TrackBuilder::new().with_send(&reverb, Decibels::IDENTITY);
        let muffle = builder.add_effect(FilterBuilder::new().cutoff(OPEN));
        let mut world = manager.add_sub_track(builder)?;
        let body = world.add_sub_track(TrackBuilder::new().with_effect(EqFilterBuilder::new(
            EqFilterKind::LowShelf,
            BODY_SHELF.0,
            Decibels(BODY_SHELF.1),
            0.7,
        )))?;
        let generated = Bank::load(dir)?;
        let mut library = if library_dir.is_dir() { Bank::load(library_dir)? } else { Bank::default() };
        let names: Vec<String> = library.sounds.keys().chain(library.loops.keys()).cloned().collect();
        for name in names {
            // (Only measured where there's a generated sound to match: measuring a loop decodes it.)
            let gain = match generated.loudness(&name).and_then(|g| Some((g, library.loudness(&name)?))) {
                Some((g, l)) if l > 1e-5 => (g / l).clamp(0.05, 20.0),
                _ => 1.0,
            };
            library.gain.insert(name, gain);
        }
        log::info!(
            "audio: {} generated sounds and {} loops, {} and {} from libraries",
            generated.sounds.len(),
            generated.loops.len(),
            library.sounds.len(),
            library.loops.len()
        );
        // The library recordings where there are any (preferred over the generated ones).
        let use_library = !library.sounds.is_empty();
        let mut audio = Self {
            manager,
            listener,
            world,
            body,
            muffle,
            underwater: false,
            reverb,
            enclosed: 0.0,
            generated,
            library,
            use_library,
            loops: HashMap::new(),
            travelling: Vec::new(),
            at: Vec3::ZERO,
            last_ambience: None,
            rng: 0x5eed_50f7,
            missing: Default::default(),
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

    /// The file of loop `name` from the set in use, and its gain.
    fn loop_file(&self, name: &str) -> Option<(PathBuf, f32)> {
        if self.use_library
            && let Some(p) = self.library.loops.get(name)
        {
            return Some((p.clone(), self.library.gain.get(name).copied().unwrap_or(1.0)));
        }
        self.generated.loops.get(name).map(|p| (p.clone(), 1.0))
    }

    /// (Re)start every loop from the set in use, at its current level (paused if silent).
    fn start_loops(&mut self) -> Result<()> {
        for a in Ambience::ALL {
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
            let Some((path, gain)) = self.loop_file(a.file()) else {
                log::debug!("no {} sound: that ambience is silent", a.file());
                self.loops.insert(a, Loop { track, sound: None, level, paused: true, gain: 1.0 });
                continue;
            };
            let data = StreamingSoundData::from_file(&path)
                .with_context(|| format!("opening {}", path.display()))?
                .loop_region(..)
                .volume(db(level * gain));
            let mut sound = match &mut track {
                Some(track) => track.play(data)?,
                None => self.world.play(data)?,
            };
            let paused = level <= 0.001;
            if paused {
                sound.pause(Tween::default());
            }
            self.loops.insert(a, Loop { track, sound: Some(sound), level, paused, gain });
        }
        Ok(())
    }

    fn random(&mut self) -> f32 {
        self.rng = self.rng.wrapping_mul(0x5851_F42D_4C95_7F2D).wrapping_add(0x1405_7B7E_F767_814F);
        (self.rng >> 40) as f32 / (1u64 << 24) as f32
    }

    /// True with probability `p` (for sounds that come now and then).
    pub fn chance(&mut self, p: f32) -> bool {
        self.random() < p
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
            if self.missing.insert(name.to_owned()) {
                log::warn!("no sound {name}");
            }
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
        let parent = if name.starts_with("step_") || name == "land" || name == "thump" { &mut self.body } else { &mut self.world };
        let track = parent.add_spatial_sub_track(
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

    /// Move each loop to where it's heard from now and fade it to its level (pausing it once
    /// silent, resuming it when it's wanted again).
    pub fn set_scene(&mut self, scene: &Scene) {
        for (a, l) in &mut self.loops {
            let (at, level) = scene.levels.get(a).copied().unwrap_or((None, 0.0));
            if let (Some(track), Some(at)) = (&mut l.track, at) {
                track.set_position(at, tween(0.25));
            }
            let Some(sound) = &mut l.sound else { continue };
            if level > 0.001 && l.paused {
                sound.set_volume(db(level * l.gain), Tween::default());
                sound.resume(tween(0.5));
                l.paused = false;
            } else if level <= 0.001 && !l.paused {
                sound.pause(tween(0.5));
                l.paused = true;
            } else if !l.paused && (level - l.level).abs() > 0.01 {
                sound.set_volume(db(level * l.gain), tween(0.5));
            }
            l.level = level;
        }
        if scene.underwater != self.underwater {
            self.underwater = scene.underwater;
            self.muffle.set_cutoff(if scene.underwater { MUFFLED } else { OPEN }, tween(0.15));
        }
        if (scene.enclosed - self.enclosed).abs() > 0.02 {
            self.enclosed = scene.enclosed;
            self.reverb.set_volume(db(scene.enclosed * CAVE_REVERB), tween(1.0));
        }
    }

    /// Everything at `volume` 0..1 (the main mix).
    pub fn set_volume(&mut self, volume: f32) {
        self.manager.main_track().set_volume(db(volume), tween(0.2));
    }
}

/// The sound of walking on (or digging) a material: soft ground, sand, loose volcanic gravel, snow,
/// or hard stone.
pub fn surface(material: asc_world::Material) -> &'static str {
    use asc_world::Material::*;
    match material {
        Grass | Soil | DryGrass | Moss | Mud | Air => "grass",
        Sand => "sand",
        Ash | Pumice => "gravel",
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

/// The natural water round `at`, looked at on a grid out to `reach`: the nearest flowing water (a
/// river), still water (a lake), the sea, and a waterfall (where a river's surface steps down
/// between neighbouring samples), each with where and how strongly.
#[derive(Default)]
pub struct Waters {
    pub river: Spot,
    pub lake: Spot,
    pub sea: Spot,
    pub waterfall: Spot,
}

pub fn water_near(worldgen: &asc_world::WorldGen, at: Vec3, reach: f32) -> Waters {
    const N: i32 = 4;
    let step = reach / N as f32;
    let side = (2 * N + 1) as usize;
    let mut levels = vec![None; side * side];
    let mut out = Waters::default();
    let nearer = |slot: &mut Spot, here: Vec3, strength: f32| {
        if slot.is_none_or(|(s, _)| s.distance(at) > here.distance(at)) {
            *slot = Some((here, strength));
        }
    };
    for j in -N..=N {
        for i in -N..=N {
            let p = Vec2::new(at.x, at.z) + Vec2::new(i as f32, j as f32) * step;
            let Some(level) = asc_world::fluid::water_level(worldgen, p.x, p.y) else { continue };
            if worldgen.height(p.x, p.y) >= level {
                continue;
            }
            levels[(i + N) as usize + side * (j + N) as usize] = Some(level);
            let here = Vec3::new(p.x, level, p.y);
            let flow = worldgen.water_flow(p.x, p.y).length();
            if flow > 0.3 {
                nearer(&mut out.river, here, (flow / 2.0).min(1.0));
            } else if (level - asc_world::SEA_LEVEL).abs() < 0.5 {
                nearer(&mut out.sea, here, 1.0);
            } else {
                nearer(&mut out.lake, here, 1.0);
            }
        }
    }
    // A drop of more than a couple of metres between neighbouring water samples is a waterfall.
    for j in 0..side {
        for i in 0..side {
            let Some(a) = levels[i + side * j] else { continue };
            for (di, dj) in [(1, 0), (0, 1)] {
                let (ni, nj) = (i + di, j + dj);
                if ni >= side || nj >= side {
                    continue;
                }
                let Some(b) = levels[ni + side * nj] else { continue };
                let drop = (a - b).abs();
                if drop > 2.0 {
                    let mid = Vec2::new(at.x, at.z)
                        + Vec2::new((i + ni) as f32 * 0.5 - N as f32, (j + nj) as f32 * 0.5 - N as f32) * step;
                    nearer(&mut out.waterfall, Vec3::new(mid.x, a.min(b), mid.y), (drop / 10.0).clamp(0.3, 1.0));
                }
            }
        }
    }
    out
}

/// What a place sounds like at (x, z), by its biome and the time of day (`daylight` 0 night ..
/// 1 day), and how much (its share of the ground sampled round the listener).
pub fn place_sounds(worldgen: &asc_world::WorldGen, at: Vec3, daylight: f32) -> Vec<(Ambience, f32)> {
    use asc_world::Biome;
    let mut shares: HashMap<Ambience, f32> = HashMap::new();
    let samples = [(0.0, 0.0), (40.0, 0.0), (-40.0, 0.0), (0.0, 40.0), (0.0, -40.0)];
    let each = 1.0 / samples.len() as f32;
    for (dx, dz) in samples {
        let g = worldgen.ground(at.x + dx, at.z + dz);
        if g.underwater {
            continue;
        }
        let (day, night) = match g.biome {
            Biome::Forest => (Ambience::ForestDay, Ambience::ForestNight),
            Biome::Taiga => (Ambience::Pine, Ambience::ForestNight),
            Biome::Jungle => (Ambience::Rainforest, Ambience::Rainforest),
            Biome::Swamp => (Ambience::Swamp, Ambience::Swamp),
            Biome::Meadow | Biome::Savanna | Biome::Shore => (Ambience::Clearing, Ambience::Evening),
            Biome::Desert => (Ambience::DesertWind, Ambience::DesertWind),
            Biome::Tundra | Biome::Arctic => (Ambience::WinterWind, Ambience::WinterWind),
        };
        // Bare ground (fresh rock, ash, ice) is quiet of life.
        let life = 1.0 - g.barren;
        *shares.entry(day).or_default() += each * daylight * life;
        *shares.entry(night).or_default() += each * (1.0 - daylight) * life;
    }
    shares.into_iter().filter(|(_, s)| *s > 0.01).collect()
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
            assert!(["grass", "sand", "gravel", "snow", "hard"].contains(&surface(m)));
        }
    }

    #[test]
    fn every_ambience_has_its_own_file() {
        let mut files: Vec<_> = Ambience::ALL.iter().map(|a| a.file()).collect();
        files.sort();
        files.dedup();
        assert_eq!(files.len(), Ambience::ALL.len());
    }

    #[test]
    fn a_place_sounds_like_its_biome_by_day_and_night() {
        let wg = asc_world::WorldGen::new(1337);
        let at = Vec3::new(300.0, wg.height(300.0, 200.0), 200.0);
        let day = place_sounds(&wg, at, 1.0);
        let night = place_sounds(&wg, at, 0.0);
        let total = |v: &[(Ambience, f32)]| v.iter().map(|(_, s)| s).sum::<f32>();
        assert!(total(&day) <= 1.0 + 1e-4 && total(&night) <= 1.0 + 1e-4);
        assert!(!day.iter().any(|(a, _)| *a == Ambience::ForestNight), "night sounds by day: {day:?}");
    }
}
