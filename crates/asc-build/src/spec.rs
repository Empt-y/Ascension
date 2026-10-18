//! `assets/data/building.toml`: grid size, part sizes and tiers.

use std::collections::BTreeMap;

use anyhow::{Context, Result, bail};
use glam::Vec3;
use serde::Deserialize;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PartKind {
    Foundation,
    Wall,
    Ceiling,
}

impl PartKind {
    pub const ALL: [Self; 3] = [Self::Foundation, Self::Wall, Self::Ceiling];

    pub fn id(self) -> &'static str {
        match self {
            Self::Foundation => "foundation",
            Self::Wall => "wall",
            Self::Ceiling => "ceiling",
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct PartSpec {
    pub name: String,
    /// Width (x), height (y), depth (z) in metres.
    pub size: [f32; 3],
    /// "bottom" or "top": which face the model's origin sits on.
    pub origin: String,
}

impl PartSpec {
    pub fn size(&self) -> Vec3 {
        Vec3::from(self.size)
    }

    /// Model-space bounds (before rotation).
    pub fn bounds(&self) -> (Vec3, Vec3) {
        let s = self.size();
        let (y0, y1) = if self.origin == "top" { (-s.y, 0.0) } else { (0.0, s.y) };
        (Vec3::new(-s.x / 2.0, y0, -s.z / 2.0), Vec3::new(s.x / 2.0, y1, s.z / 2.0))
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct TierSpec {
    pub id: String,
    pub name: String,
}

#[derive(Clone, Debug, Deserialize)]
pub struct BuildingSpec {
    pub cell: f32,
    pub height_unit: f32,
    parts: BTreeMap<String, PartSpec>,
    #[serde(rename = "tier")]
    pub tiers: Vec<TierSpec>,
}

impl BuildingSpec {
    pub fn parse(text: &str) -> Result<Self> {
        let spec: Self = toml::from_str(text).context("parsing building.toml")?;
        for kind in PartKind::ALL {
            let part = spec
                .parts
                .get(kind.id())
                .with_context(|| format!("building.toml: no [parts.{}]", kind.id()))?;
            if part.size.iter().any(|v| *v <= 0.0) || !matches!(part.origin.as_str(), "top" | "bottom") {
                bail!("building.toml: bad size or origin for {}", kind.id());
            }
        }
        if spec.tiers.is_empty() || spec.cell <= 0.0 || spec.height_unit <= 0.0 {
            bail!("building.toml: needs a cell size, height unit and at least one tier");
        }
        Ok(spec)
    }

    pub fn part(&self, kind: PartKind) -> &PartSpec {
        &self.parts[kind.id()]
    }

    /// File name of a part's model, e.g. `stone_wall.glb`.
    pub fn model_file(&self, tier: usize, kind: PartKind) -> String {
        format!("{}_{}.glb", self.tiers[tier].id, kind.id())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHIPPED: &str = include_str!("../../../assets/data/building.toml");

    #[test]
    fn shipped_spec_parses() {
        let spec = BuildingSpec::parse(SHIPPED).unwrap();
        assert_eq!(spec.cell, 2.0);
        assert_eq!(spec.tiers.len(), 4);
        let (lo, hi) = spec.part(PartKind::Ceiling).bounds();
        assert_eq!((lo.y, hi.y), (-0.2, 0.0));
        assert_eq!(spec.model_file(1, PartKind::Wall), "stone_wall.glb");
    }
}
