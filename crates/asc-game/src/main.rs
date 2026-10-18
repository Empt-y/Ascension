//! Ascension game binary.

mod app;
#[allow(dead_code)] // wired up by mining in the next step
mod content;
mod hud;
mod input;
mod player;
mod sim;
mod streaming;

fn main() -> anyhow::Result<()> {
    env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info,wgpu_core=warn,wgpu_hal=error"),
    )
    .init();
    app::run(app::Options::from_args()?)
}
