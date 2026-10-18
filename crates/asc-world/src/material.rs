/// What a solid sample is made of. Stored as one byte per sample.
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
    /// Placeholder ore so the material channel is exercised end to end.
    CopperOre = 6,
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
}

impl Material {
    pub const COUNT: usize = 12;

    pub const ALL: [Self; Self::COUNT] = [
        Self::Air,
        Self::Grass,
        Self::Soil,
        Self::Rock,
        Self::Sand,
        Self::Snow,
        Self::CopperOre,
        Self::DryGrass,
        Self::Moss,
        Self::Mud,
        Self::Ice,
        Self::Sandstone,
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
            Self::CopperOre => "copper_ore",
            Self::DryGrass => "dry_grass",
            Self::Moss => "moss",
            Self::Mud => "mud",
            Self::Ice => "ice",
            Self::Sandstone => "sandstone",
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
            6 => Self::CopperOre,
            7 => Self::DryGrass,
            8 => Self::Moss,
            9 => Self::Mud,
            10 => Self::Ice,
            11 => Self::Sandstone,
            _ => Self::Air,
        }
    }
}
