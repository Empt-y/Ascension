//! Ascension game binary.

mod app;
// The game state and rules live in asc-sim; these keep `crate::content` and friends working.
use asc_sim::{content, inventory, ore_nodes, save, sim};
mod assets;
mod building;
mod far;
mod fluids;
mod hud;
mod input;
mod inventory_ui;
mod mining;
mod player;
mod scatter;
mod streaming;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=error"),
    )
    .init();
    app::run(app::Options::from_args()?)
}
