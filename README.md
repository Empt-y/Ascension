# Ascension

A GTNH-style progression and factory game in a custom Rust engine. It plays like Minecraft (mining,
crafting, a deep tech tree) but the world isn't made of blocks: it's a smooth, diggable density field,
closer to No Man's Sky. Single-player.

**Status: Milestone 2.** You walk an infinite procedural world. You mine it at a speed set by each
material's hardness, collect what you dig into an inventory, place it back, and your world saves.
It renders with sky, sun shadows, fog, bloom and day/night.

## Running

```
cargo run --release -p asc-game
```

The debug build also works and hot-reloads shaders with F5, but use release to judge performance.

| Input | Action |
| --- | --- |
| Click | Capture the mouse |
| Esc | Close the inventory / release the mouse; press again to quit |
| W A S D | Walk |
| Space | Jump |
| Shift | Sprint |
| Hold left mouse | Mine (time depends on the material) |
| Right mouse | Place the selected item |
| 1–9 / wheel | Select a hotbar slot |
| E | Inventory: click to pick up or put down, right-click to split, Shift+click to move |
| F | Toggle fly mode (debug). Flying: Space/Ctrl up/down, mouse buttons dig/fill a brush, wheel sets brush size, Alt+wheel sets speed |
| F9 | Save now |
| T / Shift+T | Move time of day forward / back |
| `[` / `]` | View radius |
| F3 | Debug HUD |
| F5 | Reload shaders (debug builds) |
| F6 | Toggle vsync |
| F11 | Fullscreen |

Command-line options:

- `--world <name>`: world to load or create (default `world1`)
- `--new-world`: start that world over (the old one is kept, renamed)
- `--seed <n>`: seed for a new world
- `--saves-dir <dir>`: where worlds live (default `%APPDATA%\Ascension\saves`)
- `--autopilot`: fly a fixed route, dig once, and log performance every 2 s
- `--walk-test`: sprint and jump across the terrain, logging collision clearance
- `--mine-test`: mine and then place unattended, logging the inventory
- `--show-inventory`: start with the inventory open
- `--exit-after <s>`: quit after that many seconds
- `--time <0..1>`: start time of day (0 midnight, 0.5 noon)
- `--camera x,y,z,yaw,pitch`: starting pose
- `--view-radius <chunks>`: view radius in chunks
- `--vsync`: start with vsync on
- `--hide-hud`: start with the HUD hidden

`ASC_WORKERS=<n>` overrides the number of chunk worker threads.

## Layout

```
crates/asc-world   CPU only, no GPU: density chunks, worldgen, surface nets, edit log, raycast,
                   collision, volume census
crates/asc-render  wgpu renderer: terrain, sky, fog, shadows, bloom, tonemap, egui
crates/asc-game    the `ascension` binary: window, input, streaming, sim tick, player, mining,
                   inventory, saves, HUD
assets/shaders     WGSL; common.wgsl is prepended to every shader
assets/data        items.toml and materials.toml (hardness, drops); read from disk in debug builds
tools/             check.sh (fmt + clippy + tests), fg_check.py, screenshot.py, lowprio.py
```

## How the world works

- **Density field.** Each sample stores a signed distance in metres (negative = solid) plus a
  material byte. Chunks are 32³ cells at 1 m and store a padded 35³ grid. The padding lets every
  chunk mesh on its own while its borders still match its neighbours exactly (a test checks this).
- **Worldgen** is a pure function of seed and position:
  - a continent/hill/ridged-mountain height field
  - spaghetti tunnels and big deep caverns underground; tunnels pinch shut near the surface except
    at rare hillside entrances (about 0.1% of the surface is open, about 4% of the underground is
    cave; `cargo run --release -p asc-world --example terrainstats` measures both)
  - materials by depth and slope, plus a placeholder copper ore
- **Edits** are an ordered log of sphere CSG operations, indexed by the chunks they touch. A chunk
  is rebuilt from worldgen plus its ops whenever it's needed, so only meshes stay resident and
  edits survive unloading. Densities are clamped to ±4 m, which keeps each op's effect strictly
  local; that bound is what the index relies on.
- **Meshing** is naive surface nets. Each vertex blends up to 8 materials, packs a normal and a
  sky-occlusion term (darker under the original surface, e.g. in caves) into 24 bytes, and each
  quad is split along its shorter diagonal.
- **Streaming:**
  - A quarter of the cores, at low priority, build chunks by distance and frustum priority.
  - Each column gets a vertical band estimate, and the surface is followed up and down where it
    leaves a chunk.
  - Edits rebuild with top priority, with versioned results.
  - Using more workers starved the render thread and the GPU driver, which showed up as stutter.
- **GPU memory.** All chunk meshes share one vertex buffer and one index buffer, each
  sub-allocated and growable.
- **Simulation.** A fixed 20 Hz tick, separate from rendering. It only advances time of day for
  now; machines and networks will run on it.

## Playing

- **The player** is a capsule (three spheres) stepping at a fixed 60 Hz, and the camera is
  interpolated between steps.
  - Collision uses a true distance estimate (density divided by its gradient), because worldgen
    density is up to 2× steeper than a real distance near caves.
  - Walkable ground (normal y > 0.7) lifts you straight up, so you don't creep down hills.
  - A gap too low to fit stops you like a wall, and being wedged in a narrow ravine counts as
    standing.
  - If you end up buried (for example after placing on yourself), you rise out.
- **Mining:** hold the button and progress builds against the material's hardness. Each bite digs
  a 0.9 m sphere and gives the volume it removed as items, 1 per m³. Fractions are kept per item,
  so nothing rounds away. Placing pays for the air it fills.
- **Saves:** a world is a folder holding `world.toml` (seed, time), `player.toml` (position,
  inventory) and `edits.bin` (every edit). The terrain itself is rebuilt from the seed plus the
  edits.
  - Writes are atomic: the old copy is kept as `.bak`, and a damaged file falls back to it.
  - It saves on exit, every 2 minutes, and with F9.

## Rendering

- Everything is drawn camera-relative with reversed-Z depth.
- **Scene pass (HDR):** triplanar procedural terrain with GGX specular, a hemisphere ambient term
  and moonlight.
- **Shadows:** a 2048² texel-snapped sun shadow map, filtered with 3×3 PCF.
- **Sky:** a stylized analytic sky with sun, moon and stars.
- **Fog:** analytic height fog that fades fully to sky at the view distance, so chunks never
  visibly pop in.
- **Post:** a bloom chain, then an ACES tonemap and a saturation boost.

## Performance

Measured on an RTX 3070 Laptop at 1920×991, release build, view radius 10 chunks (320 m):

- **Frame rate:** about 350–420 fps flying at 40 m/s while streaming, and about 300–440 fps
  walking or mining.
- **Worst frame:** about 5–11 ms per 2 s window, once the first load is done.
- **Chunk build:** about 4–6 ms each on a worker, plus about 0.5 ms to mesh. A chunk with 1000
  mining bites rebuilds in about 8 ms.
- **Physics:** 0.05–0.25 ms per 60 Hz step. `density_at` costs about 1.75 µs per call.
- **Edits:** a bite is visible 7–15 ms after it completes. A save takes 2–5 ms.

## Next

- **M2.5, level of detail:** clipmap or octree LOD with seam stitching, so mountains are visible
  beyond 320 m.
- **M3, building:** machines placed on grid-snapped foundations, and connection detection.
- **Later:** per-material textures, grass and rock scatter, and tool tiers that change mining speed.
