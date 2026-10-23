//! Saving and loading worlds.
//!
//! A world is a directory: `world.toml` (seed, time), `players/<name>.toml` (each player's position
//! and inventory; `player.toml` before format 4), `edits.bin` (every terrain edit, postcard-encoded), `structures.bin` (building pieces) and
//! `nodes.bin` (the ore nodes mined away). Terrain itself is never saved: it is rebuilt
//! from the seed plus the edits. Each file is written atomically (to `.tmp`, the old copy kept as
//! `.bak`, then renamed into place) and loading falls back to `.bak` if a file is damaged.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use asc_world::EditOp;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::Mode;

/// Bumped when the save layout changes. 2 added `structures.bin`, 3 `nodes.bin` (older saves load
/// without them); 4 moved `player.toml` to `players/<name>.toml` (older saves load theirs as the
/// local player); 5 added players' vitals and `stashes.bin` (older saves: full health, none).
pub const FORMAT_VERSION: u32 = 5;

/// The single-player player's name.
pub const LOCAL_PLAYER: &str = "local";

/// A player name as a file name: letters, digits, `-` and `_` only.
fn player_file(name: &str) -> String {
    let clean: String =
        name.chars().map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' }).collect();
    format!("{clean}.toml")
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WorldMeta {
    pub format_version: u32,
    /// Terrain from the same seed only matches across builds with the same worldgen version.
    pub worldgen_version: u32,
    pub seed: i32,
    pub ticks: u64,
    pub time_of_day: f32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SlotSave {
    pub index: usize,
    pub item: String,
    pub count: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PlayerSave {
    pub mode: Mode,
    pub feet: [f32; 3],
    pub yaw: f32,
    pub pitch: f32,
    pub selected: usize,
    #[serde(default, rename = "slot")]
    pub slots: Vec<SlotSave>,
    #[serde(default)]
    pub remainders: std::collections::BTreeMap<String, f32>,
    /// Missing in saves before format 5: full health.
    #[serde(default)]
    pub vitals: Option<crate::vitals::Vitals>,
}

/// What a player left where they died, to come back for.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StashSave {
    pub id: u32,
    pub at: [f32; 3],
    pub slots: Vec<SlotSave>,
    pub remainders: std::collections::BTreeMap<String, f32>,
}

#[derive(Serialize, Deserialize)]
struct StashesFile {
    format_version: u32,
    stashes: Vec<StashSave>,
}

#[derive(Serialize, Deserialize)]
struct EditsFile {
    format_version: u32,
    ops: Vec<EditOp>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SaveGame {
    pub meta: WorldMeta,
    /// By player name.
    pub players: std::collections::BTreeMap<String, PlayerSave>,
    pub ops: Vec<EditOp>,
    pub structures: Vec<asc_build::Piece>,
    /// Ore nodes mined away, by `scatter::node_id`.
    pub nodes: Vec<[i32; 3]>,
    pub stashes: Vec<StashSave>,
    /// Materialised fluid chunks: coordinate, kinds and amounts (`fluid::CELLS` each).
    pub fluids: Vec<(glam::IVec3, Vec<u8>, Vec<u16>)>,
    /// Burnt ground (fire cells; any still burning is saved as burnt).
    pub burnt: Vec<[i32; 2]>,
    /// Where pyroclastic flows and blasts have passed.
    pub scars: Vec<crate::volcanism::Scar>,
}

#[derive(Serialize, Deserialize)]
struct FireFile {
    format_version: u32,
    burnt: Vec<[i32; 2]>,
    #[serde(default)]
    scars: Vec<crate::volcanism::Scar>,
}

/// One fluid chunk as runs of (count, kind, amount).
type FluidRuns = Vec<(u16, u8, u16)>;

#[derive(Serialize, Deserialize)]
struct FluidsFile {
    format_version: u32,
    /// Run-length packed: (run, kind, amount) per chunk.
    chunks: Vec<([i32; 3], FluidRuns)>,
}

fn pack_fluid(kind: &[u8], amount: &[u16]) -> Vec<(u16, u8, u16)> {
    let mut runs: Vec<(u16, u8, u16)> = Vec::new();
    for (&k, &a) in kind.iter().zip(amount) {
        match runs.last_mut() {
            Some((n, rk, ra)) if *rk == k && *ra == a && *n < u16::MAX => *n += 1,
            _ => runs.push((1, k, a)),
        }
    }
    runs
}

fn unpack_fluid(runs: &[(u16, u8, u16)]) -> (Vec<u8>, Vec<u16>) {
    let (mut kind, mut amount) = (Vec::new(), Vec::new());
    for &(n, k, a) in runs {
        for _ in 0..n {
            kind.push(k);
            amount.push(a);
        }
    }
    (kind, amount)
}

#[derive(Serialize, Deserialize)]
struct NodesFile {
    format_version: u32,
    mined: Vec<[i32; 3]>,
}

#[derive(Serialize, Deserialize)]
struct StructuresFile {
    format_version: u32,
    pieces: Vec<asc_build::Piece>,
}

/// `%APPDATA%\Ascension\saves` on Windows (the platform data directory elsewhere).
pub fn default_saves_dir() -> PathBuf {
    dirs::data_dir().unwrap_or_else(|| PathBuf::from(".")).join("Ascension").join("saves")
}

fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    path.with_file_name(name)
}

/// Write `bytes` to `path` so that a crash at any point leaves either the old or the new file (or
/// the old one as `.bak`) intact.
fn write_atomic(path: &Path, bytes: &[u8]) -> Result<()> {
    let tmp = with_suffix(path, ".tmp");
    std::fs::write(&tmp, bytes).with_context(|| format!("writing {}", tmp.display()))?;
    if path.exists() {
        std::fs::rename(path, with_suffix(path, ".bak"))
            .with_context(|| format!("backing up {}", path.display()))?;
    }
    std::fs::rename(&tmp, path).with_context(|| format!("replacing {}", path.display()))?;
    Ok(())
}

/// Read and decode a file, falling back to its `.bak` if it is missing or damaged.
fn read_with_backup<T>(path: &Path, decode: impl Fn(&[u8]) -> Result<T>) -> Result<T> {
    let primary = std::fs::read(path).map_err(anyhow::Error::from).and_then(|b| decode(&b));
    match primary {
        Ok(v) => Ok(v),
        Err(err) => {
            let bak = with_suffix(path, ".bak");
            log::warn!("{} unreadable ({err:#}); trying {}", path.display(), bak.display());
            let bytes =
                std::fs::read(&bak).with_context(|| format!("no usable {} or backup", path.display()))?;
            decode(&bytes).with_context(|| format!("backup {} is damaged too", bak.display()))
        }
    }
}

fn decode_toml<T: DeserializeOwned>(bytes: &[u8]) -> Result<T> {
    Ok(toml::from_str(std::str::from_utf8(bytes)?)?)
}

pub fn save(dir: &Path, game: &SaveGame) -> Result<()> {
    std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
    write_atomic(&dir.join("world.toml"), toml::to_string(&game.meta)?.as_bytes())?;
    let players = dir.join("players");
    std::fs::create_dir_all(&players).with_context(|| format!("creating {}", players.display()))?;
    for (name, player) in &game.players {
        write_atomic(&players.join(player_file(name)), toml::to_string(player)?.as_bytes())?;
    }
    let edits = EditsFile { format_version: FORMAT_VERSION, ops: game.ops.clone() };
    write_atomic(&dir.join("edits.bin"), &postcard::to_stdvec(&edits)?)?;
    let structures = StructuresFile { format_version: FORMAT_VERSION, pieces: game.structures.clone() };
    write_atomic(&dir.join("structures.bin"), &postcard::to_stdvec(&structures)?)?;
    let nodes = NodesFile { format_version: FORMAT_VERSION, mined: game.nodes.clone() };
    write_atomic(&dir.join("nodes.bin"), &postcard::to_stdvec(&nodes)?)?;
    let stashes = StashesFile { format_version: FORMAT_VERSION, stashes: game.stashes.clone() };
    write_atomic(&dir.join("stashes.bin"), &postcard::to_stdvec(&stashes)?)?;
    let fluids = FluidsFile {
        format_version: FORMAT_VERSION,
        chunks: game.fluids.iter().map(|(c, k, a)| (c.to_array(), pack_fluid(k, a))).collect(),
    };
    write_atomic(&dir.join("fluids.bin"), &postcard::to_stdvec(&fluids)?)?;
    let fire = FireFile { format_version: FORMAT_VERSION, burnt: game.burnt.clone(), scars: game.scars.clone() };
    write_atomic(&dir.join("fire.bin"), &postcard::to_stdvec(&fire)?)?;
    Ok(())
}

/// Load a world, or `None` if the directory has no world in it.
pub fn load(dir: &Path) -> Result<Option<SaveGame>> {
    let world = dir.join("world.toml");
    if !world.exists() && !with_suffix(&world, ".bak").exists() {
        return Ok(None);
    }
    let meta: WorldMeta = read_with_backup(&world, decode_toml)?;
    if meta.format_version > FORMAT_VERSION {
        bail!(
            "{} was saved by a newer Ascension (format {} > {FORMAT_VERSION}); refusing to load it",
            dir.display(),
            meta.format_version
        );
    }
    if meta.worldgen_version != asc_world::WORLDGEN_VERSION {
        log::warn!(
            "world was generated by worldgen v{} but this build has v{}; unedited terrain may differ",
            meta.worldgen_version,
            asc_world::WORLDGEN_VERSION
        );
    }
    let mut players = std::collections::BTreeMap::new();
    let players_dir = dir.join("players");
    if meta.format_version < 4 && !players_dir.is_dir() {
        let player: PlayerSave = read_with_backup(&dir.join("player.toml"), decode_toml)?;
        players.insert(LOCAL_PLAYER.to_owned(), player);
    } else {
        let mut names = std::collections::BTreeSet::new();
        for entry in std::fs::read_dir(&players_dir).with_context(|| format!("reading {}", players_dir.display()))? {
            let file = entry?.file_name().to_string_lossy().into_owned();
            // A player whose file was lost mid-save still has its backup.
            let name = file.strip_suffix(".toml.bak").or_else(|| file.strip_suffix(".toml"));
            if let Some(name) = name {
                names.insert(name.to_owned());
            }
        }
        for name in names {
            let player: PlayerSave = read_with_backup(&players_dir.join(player_file(&name)), decode_toml)?;
            players.insert(name, player);
        }
    }
    let edits: EditsFile = read_with_backup(&dir.join("edits.bin"), |b| Ok(postcard::from_bytes(b)?))?;
    let structures_path = dir.join("structures.bin");
    let structures = if meta.format_version < 2 && !structures_path.exists() {
        Vec::new()
    } else {
        let file: StructuresFile = read_with_backup(&structures_path, |b| Ok(postcard::from_bytes(b)?))?;
        file.pieces
    };
    let nodes_path = dir.join("nodes.bin");
    let nodes = if meta.format_version < 3 && !nodes_path.exists() {
        Vec::new()
    } else {
        let file: NodesFile = read_with_backup(&nodes_path, |b| Ok(postcard::from_bytes(b)?))?;
        file.mined
    };
    let stashes_path = dir.join("stashes.bin");
    let stashes = if meta.format_version < 5 && !stashes_path.exists() {
        Vec::new()
    } else {
        let file: StashesFile = read_with_backup(&stashes_path, |b| Ok(postcard::from_bytes(b)?))?;
        file.stashes
    };
    let fluids_path = dir.join("fluids.bin");
    let fluids = if !fluids_path.exists() && !with_suffix(&fluids_path, ".bak").exists() {
        Vec::new()
    } else {
        let file: FluidsFile = read_with_backup(&fluids_path, |b| Ok(postcard::from_bytes(b)?))?;
        file.chunks
            .into_iter()
            .map(|(c, runs)| {
                let (k, a) = unpack_fluid(&runs);
                (glam::IVec3::from_array(c), k, a)
            })
            .collect()
    };
    let fire_path = dir.join("fire.bin");
    let (burnt, scars) = if !fire_path.exists() && !with_suffix(&fire_path, ".bak").exists() {
        (Vec::new(), Vec::new())
    } else {
        let file: FireFile = read_with_backup(&fire_path, |b| Ok(postcard::from_bytes(b)?))?;
        (file.burnt, file.scars)
    };
    Ok(Some(SaveGame { meta, players, ops: edits.ops, structures, nodes, stashes, fluids, burnt, scars }))
}

#[cfg(test)]
mod tests {
    use super::*;
    use asc_world::{EditKind, Material, World};
    use glam::Vec3;

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("ascension-test-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        dir
    }

    fn sample_game() -> SaveGame {
        SaveGame {
            meta: WorldMeta {
                format_version: FORMAT_VERSION,
                worldgen_version: asc_world::WORLDGEN_VERSION,
                seed: 1337,
                ticks: 4242,
                time_of_day: 0.61,
            },
            players: [(LOCAL_PLAYER.to_owned(), PlayerSave {
                mode: Mode::Walk,
                feet: [1.5, 20.25, -3.0],
                yaw: 0.7,
                pitch: -0.2,
                selected: 2,
                slots: vec![
                    SlotSave { index: 0, item: "rock".into(), count: 37 },
                    SlotSave { index: 12, item: "soil".into(), count: 5 },
                ],
                remainders: [("rock".to_owned(), 0.375)].into(),
                vitals: Some(crate::vitals::Vitals { health: 640, ..Default::default() }),
            })]
            .into(),
            ops: (0..50)
                .map(|i| EditOp {
                    center: Vec3::new(i as f32 * 0.7, 9.0 - (i % 4) as f32, 2.0),
                    radius: 0.9,
                    kind: if i % 5 == 0 { EditKind::Fill } else { EditKind::Dig },
                    material: if i % 5 == 0 { Material::Rock } else { Material::Air },
                })
                .collect(),
            structures: vec![
                asc_build::Piece {
                    key: asc_build::PieceKey::foundation(glam::IVec2::new(3, -2), 4),
                    tier: 1,
                    grounded: true,
                },
                asc_build::Piece {
                    key: asc_build::PieceKey::wall(glam::IVec2::new(3, -2), 6, asc_build::Edge::East),
                    tier: 3,
                    grounded: false,
                },
            ],
            nodes: vec![[12, -40, 7], [-3, 5, 900]],
            stashes: vec![StashSave {
                id: 3,
                at: [1.0, 2.0, 3.0],
                slots: vec![SlotSave { index: 0, item: "soil".into(), count: 9 }],
                remainders: Default::default(),
            }],
            fluids: vec![(glam::IVec3::new(1, -1, 2), vec![1; 32768], vec![700; 32768])],
            burnt: vec![[4, -7], [5, -7]],
            scars: vec![crate::volcanism::Scar::Flow {
                origin: glam::Vec2::new(10.0, 20.0),
                dir: glam::Vec2::X,
                length: 3000.0,
                width: 250.0,
            }],
        }
    }

    #[test]
    fn version_1_saves_load_without_structures() {
        let dir = temp_dir("v1");
        let mut game = sample_game();
        game.meta.format_version = 1;
        save(&dir, &game).unwrap();
        std::fs::remove_file(dir.join("structures.bin")).unwrap();
        std::fs::remove_file(dir.join("nodes.bin")).unwrap();
        std::fs::remove_file(dir.join("stashes.bin")).unwrap();
        let loaded = load(&dir).unwrap().unwrap();
        assert!(loaded.structures.is_empty());
        assert!(loaded.nodes.is_empty());
        assert_eq!(loaded.ops, game.ops);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn version_3_saves_load_their_player_as_the_local_one() {
        let dir = temp_dir("v3");
        let mut game = sample_game();
        game.meta.format_version = 3;
        save(&dir, &game).unwrap();
        let player = game.players[LOCAL_PLAYER].clone();
        std::fs::remove_dir_all(dir.join("players")).unwrap();
        std::fs::write(dir.join("player.toml"), toml::to_string(&player).unwrap()).unwrap();
        let loaded = load(&dir).unwrap().unwrap();
        assert_eq!(loaded.players[LOCAL_PLAYER], player);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn several_players_each_get_a_file() {
        let dir = temp_dir("players");
        let mut game = sample_game();
        let mut other = game.players[LOCAL_PLAYER].clone();
        other.feet = [9.0, 9.0, 9.0];
        game.players.insert("Ash/../x".to_owned(), other);
        save(&dir, &game).unwrap();
        assert!(dir.join("players").join("Ash____x.toml").is_file(), "names are made safe for files");
        let loaded = load(&dir).unwrap().unwrap();
        assert_eq!(loaded.players.len(), 2);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn round_trip() {
        let dir = temp_dir("roundtrip");
        assert_eq!(load(&dir).unwrap(), None);
        let game = sample_game();
        save(&dir, &game).unwrap();
        assert_eq!(load(&dir).unwrap(), Some(game));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn terrain_rebuilt_from_a_save_matches() {
        let dir = temp_dir("terrain");
        let game = sample_game();
        let mut before = World::new(game.meta.seed);
        for op in &game.ops {
            before.edits.push(*op);
        }
        save(&dir, &game).unwrap();
        let loaded = load(&dir).unwrap().unwrap();
        let mut after = World::new(loaded.meta.seed);
        after.edits = asc_world::EditLog::from_ops(loaded.ops);
        let coord = asc_world::chunk_of(Vec3::new(10.0, 8.0, 2.0));
        let (a, b) = (before.build_chunk(coord), after.build_chunk(coord));
        assert_eq!(a.density, b.density);
        assert_eq!(a.material, b.material);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn damaged_file_falls_back_to_backup() {
        let dir = temp_dir("backup");
        let first = sample_game();
        save(&dir, &first).unwrap();
        let mut second = first.clone();
        second.meta.ticks += 100;
        save(&dir, &second).unwrap();
        // Truncate the current edits file, as a crash mid-write of a non-atomic save would.
        let edits = dir.join("edits.bin");
        let bytes = std::fs::read(&edits).unwrap();
        std::fs::write(&edits, &bytes[..bytes.len() / 3]).unwrap();
        let loaded = load(&dir).unwrap().unwrap();
        assert_eq!(loaded.ops, first.ops, "edits come from the backup");
        assert_eq!(loaded.meta.ticks, second.meta.ticks, "undamaged files still load the latest");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn old_edit_kinds_keep_their_encoding() {
        // Saves store enum variant indices, so Dig and Fill must stay the first two kinds; new
        // kinds (the flatten boxes) are appended after them.
        assert_eq!(postcard::to_stdvec(&EditKind::Dig).unwrap(), [0]);
        assert_eq!(postcard::to_stdvec(&EditKind::Fill).unwrap(), [1]);
    }

    #[test]
    fn newer_format_is_refused() {
        let dir = temp_dir("newer");
        let mut game = sample_game();
        game.meta.format_version = FORMAT_VERSION + 1;
        save(&dir, &game).unwrap();
        let err = load(&dir).unwrap_err();
        assert!(format!("{err:#}").contains("newer"), "{err:#}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
