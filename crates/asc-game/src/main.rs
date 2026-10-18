//! Ascension game binary.

mod app;
mod content;
mod hud;
mod input;
mod inventory;
mod inventory_ui;
mod mining;
mod player;
mod save;
mod sim;
mod streaming;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=error"),
    )
    .init();
    app::run(app::Options::from_args()?)
}
