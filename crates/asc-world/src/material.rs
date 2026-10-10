/// What a solid sample is made of. Stored as one byte per sample.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Material {
    Air = 0,
    Grass = 1,
    Soil = 2,
    Rock = 3,
    Sand = 4,
    Snow = 5,
    /// Placeholder ore so the material channel is exercised end to end.
    CopperOre = 6,
}

impl Material {
    pub const COUNT: usize = 7;

    pub fn from_u8(v: u8) -> Self {
        match v {
            1 => Self::Grass,
            2 => Self::Soil,
            3 => Self::Rock,
            4 => Self::Sand,
            5 => Self::Snow,
            6 => Self::CopperOre,
            _ => Self::Air,
        }
    }
}
