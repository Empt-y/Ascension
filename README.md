# Ascension

A GTNH-style progression and factory game in a custom Rust engine. It plays like Minecraft (mining,
crafting, a deep tech tree) but the world isn't made of blocks: it's a smooth, diggable density field,
closer to No Man's Sky. Single-player.

**Status: Milestone 1.** An infinite procedural world streams in around a fly camera. You can dig
and fill it in real time. It renders with sky, sun shadows, fog, bloom and day/night.

## Running

```
cargo run --release -p asc-game
```

The debug build also works and hot-reloads shaders with F5, but use release to judge performance.

| Input | Action |
| --- | --- |
| Click | Capture the mouse |
| Esc | Release the mouse; press again to quit |
| W A S D, Space, Ctrl | Fly |
| Shift | Fly faster |
| Alt + wheel | Change fly speed |
| Left / right mouse | Dig / fill |
| Wheel | Brush size (1–8 m) |
| T / Shift+T | Move time of day forward / back |
| `[` / `]` | View radius |
| F3 | Debug HUD |
| F5 | Reload shaders (debug builds) |
| F6 | Toggle vsync |
| F11 | Fullscreen |

Command-line options:

- `--autopilot`: fly a fixed route, dig once, and log performance every 2 s
- `--exit-after <s>`: quit after that many seconds
- `--time <0..1>`: start time of day (0 midnight, 0.5 noon)
- `--camera x,y,z,yaw,pitch`: starting pose
- `--view-radius <chunks>`: view radius in chunks
- `--vsync`: start with vsync on
- `--hide-hud`: start with the HUD hidden

`ASC_WORKERS=<n>` overrides the number of chunk worker threads.

## Layout

```
crates/asc-world   CPU only, no GPU: density chunks, worldgen, surface nets, edit log, raycast
crates/asc-render  wgpu renderer: terrain, sky, fog, shadows, bloom, tonemap, egui
crates/asc-game    the `ascension` binary: window, input, streaming, sim tick, HUD
assets/shaders     WGSL; common.wgsl is prepended to every shader
tools/             check.sh (fmt + clippy + tests), fg_check.py, screenshot.py
```

## How the world works

- **Density field.** Each sample stores a signed distance in metres (negative = solid) plus a
  material byte. Chunks are 32³ cells at 1 m and store a padded 35³ grid. The padding lets every
  chunk mesh on its own while its borders still match its neighbours exactly (a test checks this).
- **Worldgen** is a pure function of seed and position:
  - a continent/hill/ridged-mountain height field
  - 3D overhang noise on mountains
  - spaghetti caves that widen near the surface into ravines
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

- **Frame rate:** about 350–420 fps (2.4–2.9 ms) while flying at 40 m/s and streaming.
- **Worst frame:** about 6–11 ms per 2 s window.
- **Chunk build:** about 4–6 ms each on a worker, plus about 0.5 ms to mesh.
- **Edits:** a dig is visible 10–15 ms after the click.

## Next (M2 candidates)

- Level of detail (clipmap or octree with seam stitching), so mountains are visible beyond 320 m
- Saving and loading the edit log and world
- A walking player with collision, replacing the fly camera
- Per-material textures, and grass/rock scatter
