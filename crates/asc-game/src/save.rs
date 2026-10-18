//! Saving and loading worlds.
//!
//! A world is a directory: `world.toml` (seed, time), `player.toml` (position, inventory) and
//! `edits.bin` (every terrain edit, postcard-encoded). Terrain itself is never saved: it is rebuilt
//! from the seed plus the edits. Each file is written atomically (to `.tmp`, the old copy kept as
//! `.bak`, then renamed into place) and loading falls back to `.bak` if a file is damaged.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use asc_world::EditOp;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::player::Mode;

/// Bumped when the save layout changes. 2 added `structures.bin`; version 1 saves load with no
/// structures.
pub const FORMAT_VERSION: u32 = 2;

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
}

#[derive(Serialize, Deserialize)]
struct EditsFile {
    format_version: u32,
    ops: Vec<EditOp>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SaveGame {
    pub meta: WorldMeta,
    pub player: PlayerSave,
    pub ops: Vec<EditOp>,
    pub structures: Vec<asc_build::Piece>,
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
    write_atomic(&dir.join("player.toml"), toml::to_string(&game.player)?.as_bytes())?;
    let edits = EditsFile { format_version: FORMAT_VERSION, ops: game.ops.clone() };
    write_atomic(&dir.join("edits.bin"), &postcard::to_stdvec(&edits)?)?;
    let structures = StructuresFile { format_version: FORMAT_VERSION, pieces: game.structures.clone() };
    write_atomic(&dir.join("structures.bin"), &postcard::to_stdvec(&structures)?)?;
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
    let player: PlayerSave = read_with_backup(&dir.join("player.toml"), decode_toml)?;
    let edits: EditsFile = read_with_backup(&dir.join("edits.bin"), |b| Ok(postcard::from_bytes(b)?))?;
    let structures_path = dir.join("structures.bin");
    let structures = if meta.format_version < 2 && !structures_path.exists() {
        Vec::new()
    } else {
        let file: StructuresFile = read_with_backup(&structures_path, |b| Ok(postcard::from_bytes(b)?))?;
        file.pieces
    };
    Ok(Some(SaveGame { meta, player, ops: edits.ops, structures }))
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
            player: PlayerSave {
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
            },
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
        }
    }

    #[test]
    fn version_1_saves_load_without_structures() {
        let dir = temp_dir("v1");
        let mut game = sample_game();
        game.meta.format_version = 1;
        save(&dir, &game).unwrap();
        std::fs::remove_file(dir.join("structures.bin")).unwrap();
        let loaded = load(&dir).unwrap().unwrap();
        assert!(loaded.structures.is_empty());
        assert_eq!(loaded.ops, game.ops);
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
