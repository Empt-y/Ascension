/// Sample bytes from this value up are ores: byte `ORE_BASE + i` is ore `i` of the ore table
/// ([`crate::ores::builtin`]). They all read as [`Material::Ore`]; [`ore_index`] gives which one.
pub const ORE_BASE: u8 = 64;

/// The sample byte for ore `index`.
pub fn ore_byte(index: usize) -> u8 {
    debug_assert!(index < (256 - ORE_BASE as usize));
    ORE_BASE + index as u8
}

/// Which ore a sample byte holds, if it's one.
pub fn ore_index(byte: u8) -> Option<usize> {
    (byte >= ORE_BASE).then(|| (byte - ORE_BASE) as usize)
}

/// What a solid sample is made of. Stored as one byte per sample (ores as `ORE_BASE + index`).
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Material {
    Air = 0,
    Grass = 1,
    Soil = 2,
    Rock = 3,
    Sand = 4,
    Snow = 5,
    /// Any ore; the sample byte says which (see [`ORE_BASE`]).
    Ore = 6,
    /// Sun-bleached savanna and steppe grass.
    DryGrass = 7,
    /// Tundra and taiga ground cover.
    Moss = 8,
    /// Swamp ground.
    Mud = 9,
    /// Arctic ice sheets.
    Ice = 10,
    /// Desert bedrock and cliffs.
    Sandstone = 11,
    /// Lava that cooled as it flowed; the rock of volcanoes.
    Basalt = 12,
    /// Lava quenched still (by water): black glass.
    Obsidian = 13,
    /// Frothy volcanic rock, light enough to float.
    Pumice = 14,
    /// Volcanic ash, settled.
    Ash = 15,
    /// Sulfur crust round vents and fumaroles.
    Sulfur = 16,
}

impl Material {
    pub const COUNT: usize = 17;

    pub const ALL: [Self; Self::COUNT] = [
        Self::Air,
        Self::Grass,
        Self::Soil,
        Self::Rock,
        Self::Sand,
        Self::Snow,
        Self::Ore,
        Self::DryGrass,
        Self::Moss,
        Self::Mud,
        Self::Ice,
        Self::Sandstone,
        Self::Basalt,
        Self::Obsidian,
        Self::Pumice,
        Self::Ash,
        Self::Sulfur,
    ];

    /// Stable lowercase name, as used in content files and saves.
    pub fn name(self) -> &'static str {
        match self {
            Self::Air => "air",
            Self::Grass => "grass",
            Self::Soil => "soil",
            Self::Rock => "rock",
            Self::Sand => "sand",
            Self::Snow => "snow",
            Self::Ore => "ore",
            Self::DryGrass => "dry_grass",
            Self::Moss => "moss",
            Self::Mud => "mud",
            Self::Ice => "ice",
            Self::Sandstone => "sandstone",
            Self::Basalt => "basalt",
            Self::Obsidian => "obsidian",
            Self::Pumice => "pumice",
            Self::Ash => "ash",
            Self::Sulfur => "sulfur",
        }
    }

    pub fn from_name(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|m| m.name() == name)
    }

    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Grass,
            2 => Self::Soil,
            3 => Self::Rock,
            4 => Self::Sand,
            5 => Self::Snow,
            6 => Self::Ore,
            7 => Self::DryGrass,
            8 => Self::Moss,
            9 => Self::Mud,
            10 => Self::Ice,
            11 => Self::Sandstone,
            12 => Self::Basalt,
            13 => Self::Obsidian,
            14 => Self::Pumice,
            15 => Self::Ash,
            16 => Self::Sulfur,
            v if v >= ORE_BASE => Self::Ore,
            _ => Self::Air,
        }
    }
}
