# Ascension

A GTNH-style progression and factory game in a custom Rust engine. It plays like Minecraft (mining,
crafting, a deep tech tree) but the world isn't made of blocks: it's a smooth, diggable density field,
closer to No Man's Sky. Single-player.

**Status: Milestone 3.** You walk an infinite procedural world that you can see kilometres into,
with lakes and seas in the lowlands. You mine it at a speed set by each material's hardness,
collect what you dig into an inventory, place it back, and your world saves. You can level the
ground and build with foundations, walls and ceilings in four tiers (wood, stone, concrete, steel)
that snap to a 2 m grid. It renders with sky, sun shadows, haze, bloom and day/night.

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
| B | Build menu: pick a piece (or the flatten tool) to enter build mode |
| Build mode: left mouse | Place the ghost piece (blue: allowed, red: the status line says why not) |
| Build mode: wheel | Change tier |
| Build mode: X | Dismantle the piece under the crosshair |
| Build mode: right mouse / Esc | Leave build mode |
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
- `--far on|off`: far terrain and water (default on)
- `--fog <density>`: haze density (default 0.00045)
- `--msaa 1|4`: scene anti-aliasing (default 4)
- `--exit-after <s>`: quit after that many seconds
- `--time <0..1>`: start time of day (0 midnight, 0.5 noon)
- `--camera x,y,z,yaw,pitch`: starting pose
- `--view-radius <chunks>`: view radius in chunks
- `--vsync`: start with vsync on
- `--hide-hud`: start with the HUD hidden
- `--showcase`: lay out every building part near spawn
- `--build-test`: flatten a site, build a small house on it, walk in through the door, and log
- `--build-stress`: place 1600 pieces (a 10×10, three-storey block) for performance testing

`ASC_WORKERS=<n>` overrides the number of chunk worker threads.

## Layout

```
crates/asc-world   CPU only, no GPU: density chunks, worldgen, surface nets, edit log, raycast,
                   collision, volume census
crates/asc-render  wgpu renderer: terrain, building parts, sky, fog, shadows, bloom, tonemap, egui
crates/asc-build   CPU only: the building lattice, snapping, support rules, structure SDF
crates/asc-game    the `ascension` binary: window, input, streaming, sim tick, player, mining,
                   building, inventory, saves, HUD
assets/shaders     WGSL; common.wgsl is prepended to every shader
assets/data        items.toml, materials.toml (hardness, drops) and building.toml (piece sizes,
                   tiers); read from disk in debug builds
assets/models      the building parts as GLB, plus preview renders
tools/             check.sh (fmt + clippy + tests), fg_check.py, screenshot.py, lowprio.py,
                   build_parts.py and blender/parts.py (the part generator)
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
- **Edits** are an ordered log of sphere and box CSG operations, indexed by the chunks they touch. A chunk
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
  inventory), `edits.bin` (every edit) and `structures.bin` (every placed piece). The terrain
  itself is rebuilt from the seed plus the edits. Format version 2 added structures; version 1
  saves still load, with none.
  - Writes are atomic: the old copy is kept as `.bak`, and a damaged file falls back to it.
  - It saves on exit, every 2 minutes, and with F9.

## Building

- **The grid:** cells are 2 m across and heights go in 0.5 m units. A piece is identified by an
  integer key (kind, cell, height, and which edge for a wall), so placement is exact and saves
  cleanly.
  - Foundations are 2×2×1 m, walls 2×2 m and 0.2 m thick on a cell edge, ceilings 2×2×0.2 m. Two
    stacked walls make a 4 m room.
- **Snapping** comes from what the crosshair is on: a foundation on terrain levels itself above
  the highest ground under its footprint, and a foundation next to another joins it at the same
  height. Walls go on the nearest edge of a foundation, floor or wall top. Ceilings go on wall tops
  or next to another ceiling.
- **Rules:** pieces can't overlap. Each piece needs support (ground or a neighbouring foundation;
  something under a wall; a wall or ceiling beside a ceiling). A foundation is refused if there's a
  gap of more than 0.5 m under any part of it. You can't dismantle a piece that others rest on.
  Placing is free for now: costs wait until crafting is designed.
- **Flatten ground** (in the build menu) levels the cell you aim at to the height of the point you aim at, rounded to a height unit,
  cutting above and filling below with soil. It's meant to become an unlockable tool.
- **Collision:** pieces are boxes in a signed distance field that's combined with the terrain's,
  so walking, standing and raycasts work on both with the same code. You step up ledges of up to
  0.55 m, such as a foundation's rim.
- **Models** are made by a parametric Blender script run headless (`python tools/build_parts.py`),
  using sizes from `building.toml`. It checks every part's bounds against the spec to 1 mm and
  renders previews for the build menu. Parts are drawn instanced, cast and receive shadows, and use
  the terrain's lighting and procedural detail.

## Rendering

- Everything is drawn camera-relative with reversed-Z depth.
- **Far terrain** is a geometry clipmap of the height field: 7 nested rings with spacing 2 m to
  128 m, 253×253 vertices each, so the world reaches about 12 km before it fades into the sky.
  - The shared grid's quads are sorted by ring, so each level skips the middle that a finer level
    or the near chunks already cover. Drawing that middle and discarding it cost about 3 ms in
    elevated views.
  - Heights live in a toroidal texture array that one low-priority worker refreshes as you move.
    Coarse levels go first, a level switches window only once its whole update is ready, and a
    hysteresis rule stops re-centring when you stand on a snap boundary.
  - The outer band of each ring morphs into the next coarser one, so edges meet without cracks.
  - Inside the 320 m chunk radius only the full-detail chunks are drawn. In a 24 m overlap band
    the far surface sits 0.4 m lower so chunks always win.
  - It uses the same material rules and lighting (`shade_terrain` in `common.wgsl`), so near and
    far match.
  - It ignores edits and cave mouths. Beyond 320 m they're too small to matter.
- **Surface detail** is procedural: five noise octaves (0.45 to 36 per metre) give colour variation
  and, through derivative-based bump mapping, lit relief.
  - Each octave fades as its wavelength nears the pixel's footprint on the ground, the procedural
    equivalent of mipmapping.
  - Colour uses the geometric mean of the footprint's two axes, a cousin of anisotropic
    filtering, so ground seen at a grazing angle stays crisp instead of smearing.
- **Anti-aliasing:** 4× MSAA on the scene pass (`--msaa 1` turns it off).
- **Water** is one flat surface at sea level (2 m) with ripples, sky reflection with Fresnel and a
  sun glint. Its depth comes from the far height textures, which gives shore fades and deep-water
  colour, and dry land (including pits dug below sea level) gets no water. It's visual only for
  now: there's no swimming yet.
- **Scene pass (HDR):** triplanar procedural terrain with GGX specular, a hemisphere ambient term
  and moonlight.
- **Shadows:** a 2048² texel-snapped sun shadow map, filtered with 3×3 PCF.
- **Sky:** a stylized analytic sky with sun, moon and stars.
- **Fog:** analytic height fog that fades fully to sky at the view distance, so chunks never
  visibly pop in.
- **Post:** a bloom chain, then an ACES tonemap and a saturation boost.

## Performance

Measured on an RTX 3070 Laptop at 1920×991, release build, view radius 10 chunks (320 m):

- **Frame rate:** about 200–230 fps at ground level with 4× MSAA and full surface detail (about
  5 ms per frame), and about 300 fps in an elevated view over hills. The detail shader costs about
  1 ms and MSAA about 0.4 ms.
- **Building:** 1600 pieces in view (`--build-stress`) cost about 1.1 ms (297 → 223 fps in the
  same elevated view).
- **Worst frame:** about 5–7 ms in most 2 s windows, with spikes up to about 24 ms where chunk
  streaming is heaviest; `--far off` shows similar spikes in the same spots.
- **Far terrain:** a full refresh is about 455k height samples (0.15 s at about 3 M samples/s on
  one worker). Moving costs a few thousand samples per second.
- **Chunk build:** about 4–6 ms each on a worker, plus about 0.5 ms to mesh. A chunk with 1000
  mining bites rebuilds in about 8 ms.
- **Physics:** 0.05–0.25 ms per 60 Hz step.
- **Edits:** a bite is visible 7–15 ms after it completes. A save takes 2–5 ms.

## Next

- **Crafting:** the costs of pieces and a recipe system, the core of the game's difficulty.
- **Machines** on the building grid, and connection detection.
- **Water gameplay:** swimming, and an underwater view.
- **Later:** per-material textures, grass and rock scatter, tool tiers that change mining speed,
  and showing edits in far terrain.
