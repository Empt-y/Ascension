//! Game content loaded from `assets/data`: items and per-material mining properties.
//!
//! Debug builds read the files from disk (edit and restart, no recompile); release builds embed them.

use std::collections::BTreeMap;
use std::path::PathBuf;

use ahash::AHashMap;
use anyhow::{Context, Result, bail};
use asc_world::{Material, ore_index};
use serde::Deserialize;

const EMBEDDED_ITEMS: &str = include_str!("../../../assets/data/items.toml");
const EMBEDDED_MATERIALS: &str = include_str!("../../../assets/data/materials.toml");

/// Index into [`Content::items`]. Saves store the string id instead, so this may change between runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize)]
pub struct ItemId(pub u16);

#[derive(Clone, Debug)]
pub struct ItemDef {
    pub id: String,
    pub name: String,
    pub color: [f32; 3],
    pub stack: u32,
    pub places: Option<Material>,
    /// A tool, held to use.
    pub tool: Option<crate::tools::ToolKind>,
}

#[derive(Clone, Copy, Debug, Default)]
pub struct MaterialDef {
    /// Seconds of mining per bite.
    pub hardness: f32,
    pub drop: Option<ItemId>,
}

pub struct Content {
    pub items: Vec<ItemDef>,
    by_id: AHashMap<String, ItemId>,
    pub materials: [MaterialDef; Material::COUNT],
    /// Per ore (index into `asc_world::ores::builtin()`): its item and mining time per bite.
    pub ores: Vec<MaterialDef>,
}

#[derive(Deserialize)]
struct ItemsFile {
    item: Vec<RawItem>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawItem {
    id: String,
    name: String,
    color: [f32; 3],
    #[serde(default = "default_stack")]
    stack: u32,
    places: Option<String>,
    tool: Option<crate::tools::ToolKind>,
}

fn default_stack() -> u32 {
    64
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawMaterial {
    hardness: f32,
    drop: Option<String>,
    /// Ores only: extra hardness per tool tier.
    #[serde(default)]
    tier_hardness: f32,
}

impl Content {
    pub fn load() -> Result<Self> {
        let dir = cfg!(debug_assertions)
            .then(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../assets/data"))
            .filter(|d| d.is_dir());
        let read = |name: &str, embedded: &str| -> Result<String> {
            match &dir {
                Some(d) => {
                    let path = d.join(name);
                    std::fs::read_to_string(&path).with_context(|| format!("reading {}", path.display()))
                }
                None => Ok(embedded.to_owned()),
            }
        };
        Self::parse(&read("items.toml", EMBEDDED_ITEMS)?, &read("materials.toml", EMBEDDED_MATERIALS)?)
    }

    pub fn parse(items_toml: &str, materials_toml: &str) -> Result<Self> {
        let items_file: ItemsFile = toml::from_str(items_toml).context("parsing items.toml")?;
        let mut items = Vec::new();
        let mut by_id = AHashMap::new();
        for raw in items_file.item {
            let places = match &raw.places {
                Some(name) => Some(
                    Material::from_name(name)
                        .with_context(|| format!("item {}: unknown material {name:?} in `places`", raw.id))?,
                ),
                None => None,
            };
            if places == Some(Material::Air) {
                bail!("item {}: cannot place air", raw.id);
            }
            if raw.stack == 0 {
                bail!("item {}: stack must be at least 1", raw.id);
            }
            let id = ItemId(items.len() as u16);
            if by_id.insert(raw.id.clone(), id).is_some() {
                bail!("item {} is defined twice", raw.id);
            }
            items.push(ItemDef { id: raw.id, name: raw.name, color: raw.color, stack: raw.stack, places, tool: raw.tool });
        }
        // An item per ore.
        let table = asc_world::ores::builtin();
        let mut ore_items = Vec::new();
        for ore in &table.ores {
            let id = format!("{}_ore", ore.id);
            let item = ItemId(items.len() as u16);
            if by_id.insert(id.clone(), item).is_some() {
                bail!("ore item {id} clashes with an item in items.toml");
            }
            items.push(ItemDef {
                id,
                name: format!("{} Ore", ore.name),
                color: ore.color,
                stack: 64,
                places: None,
                tool: None,
            });
            ore_items.push(item);
        }

        let raw_materials: BTreeMap<String, RawMaterial> =
            toml::from_str(materials_toml).context("parsing materials.toml")?;
        let mut materials = [MaterialDef::default(); Material::COUNT];
        let mut seen = [false; Material::COUNT];
        for (name, raw) in &raw_materials {
            let m = Material::from_name(name)
                .with_context(|| format!("materials.toml: unknown material {name:?}"))?;
            if m == Material::Air {
                bail!("materials.toml: air has no properties");
            }
            if raw.hardness <= 0.0 {
                bail!("material {name}: hardness must be positive");
            }
            if (m == Material::Ore) != raw.drop.is_none() {
                bail!("material {name}: every material but ore needs a drop, and ore must not have one");
            }
            let drop = match &raw.drop {
                Some(item) => Some(
                    *by_id
                        .get(item)
                        .with_context(|| format!("material {name}: drop {item:?} is not an item"))?,
                ),
                None => None,
            };
            materials[m as usize] = MaterialDef { hardness: raw.hardness, drop };
            seen[m as usize] = true;
        }
        for m in Material::ALL.into_iter().filter(|m| *m != Material::Air) {
            if !seen[m as usize] {
                bail!("materials.toml: no entry for {}", m.name());
            }
        }
        let tier_hardness = raw_materials.get("ore").map_or(0.0, |r| r.tier_hardness);
        let base = materials[Material::Ore as usize].hardness;
        let ores = table
            .ores
            .iter()
            .zip(ore_items)
            .map(|(o, item)| MaterialDef { hardness: base + tier_hardness * o.tier as f32, drop: Some(item) })
            .collect();
        Ok(Self { items, by_id, materials, ores })
    }

    pub fn item(&self, id: ItemId) -> &ItemDef {
        &self.items[id.0 as usize]
    }

    /// The item that is the tool `kind`.
    pub fn tool_item(&self, kind: crate::tools::ToolKind) -> Option<ItemId> {
        self.items.iter().position(|i| i.tool == Some(kind)).map(|i| ItemId(i as u16))
    }

    pub fn item_id(&self, name: &str) -> Option<ItemId> {
        self.by_id.get(name).copied()
    }

    pub fn material(&self, m: Material) -> &MaterialDef {
        &self.materials[m as usize]
    }

    /// Mining properties of a sample byte: its ore's own if it's an ore.
    pub fn sample(&self, byte: u8) -> &MaterialDef {
        match ore_index(byte) {
            Some(i) => &self.ores[i],
            None => self.material(Material::from_u8(byte)),
        }
    }

    /// What a sample byte is called, for the HUD.
    pub fn sample_name(byte: u8) -> &'static str {
        match ore_index(byte) {
            Some(i) => asc_world::ores::builtin().ores[i].name.as_str(),
            None => Material::from_u8(byte).name(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_content_is_valid() {
        let c = Content::parse(EMBEDDED_ITEMS, EMBEDDED_MATERIALS).expect("content should load");
        for m in Material::ALL.into_iter().filter(|m| *m != Material::Air && *m != Material::Ore) {
            let def = c.material(m);
            assert!(def.hardness > 0.0, "{} has no hardness", m.name());
            assert!(def.drop.is_some(), "{} drops nothing", m.name());
        }
        let table = asc_world::ores::builtin();
        assert_eq!(c.ores.len(), table.ores.len());
        let coal = table.ore_index("coal").unwrap();
        let def = c.sample(asc_world::ore_byte(coal));
        assert_eq!(c.item(def.drop.unwrap()).id, "coal_ore");
        let hardest = c.ores.iter().map(|d| d.hardness).fold(0.0, f32::max);
        assert!(hardest > def.hardness, "higher tiers take longer");
        assert_eq!(c.item(c.material(Material::Grass).drop.unwrap()).id, "soil");
        assert_eq!(c.item(c.item_id("rock").unwrap()).places, Some(Material::Rock));
    }

    #[test]
    fn bad_references_are_rejected() {
        let items = "[[item]]\nid = \"a\"\nname = \"A\"\ncolor = [1, 1, 1]\nplaces = \"lava\"\n";
        assert!(Content::parse(items, EMBEDDED_MATERIALS).is_err());
        let materials = EMBEDDED_MATERIALS.replace("drop = \"rock\"", "drop = \"nope\"");
        assert!(Content::parse(EMBEDDED_ITEMS, &materials).is_err());
        let missing = EMBEDDED_MATERIALS.replace("[snow]", "[snow_typo]");
        assert!(Content::parse(EMBEDDED_ITEMS, &missing).is_err());
    }
}
