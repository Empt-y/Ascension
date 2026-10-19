"""Regenerate the ore node models with Blender and check them.

    python tools/build_ores.py [--blender PATH] [--previews] [--only id1,id2]

Runs tools/blender/ores.py headless for assets/data/ores.toml, then checks each model's size, how
far it sinks below the ground and its triangle budget. With --previews it also renders a preview
per model and a contact sheet per style in assets/models/ores/previews (this uses the GPU for a
few seconds, so it's off by default).
"""

import argparse
import json
import os
import subprocess
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "assets", "data", "ores.toml")
OUT = os.path.join(ROOT, "assets", "models", "ores")
DEFAULT_BLENDER = r"C:\Users\theas\tools\blender-5.2.2\blender.exe"
BUDGET = (60, 15000)
# (lowest point, highest point range, widest half-extent) per variant.
LIMITS = {"surface": (-0.6, (0.1, 1.2), 1.2), "deep": (-0.05, (0.5, 2.4), 2.0)}
STYLES = ("metal", "earthy", "banded", "crystal", "cubic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER))
    ap.add_argument("--previews", action="store_true")
    ap.add_argument("--only", default="")
    args = ap.parse_args()

    script = os.path.join(ROOT, "tools", "blender", "ores.py")
    cmd = [args.blender, "--background", "--factory-startup", "--python", script, "--", SPEC, OUT,
           "previews" if args.previews else "", args.only]
    print("running Blender...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in result.stdout.splitlines():
        if line.startswith(("PARTS DONE", "Traceback")) or "Error" in line:
            print("  " + line)
    if result.returncode != 0 or "PARTS DONE" not in result.stdout:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        sys.exit("Blender failed")

    with open(SPEC, "rb") as f:
        spec = tomllib.load(f)
    with open(os.path.join(OUT, "report.json")) as f:
        report = json.load(f)
    problems = []
    for ore in spec["ore"]:
        if ore["style"] not in STYLES:
            problems.append(f"{ore['id']}: unknown style {ore['style']}")
        for variant, (sink, (h0, h1), width) in LIMITS.items():
            name = f"{ore['id']}_{variant}"
            got = report.get(name)
            if got is None:
                problems.append(f"{name}: missing")
                continue
            lo, hi = got["min"], got["max"]
            if lo[1] < sink:
                problems.append(f"{name}: sinks to {lo[1]:.2f}")
            if not h0 <= hi[1] <= h1:
                problems.append(f"{name}: height {hi[1]:.2f} outside {h0}..{h1}")
            if max(abs(v) for v in (lo[0], lo[2], hi[0], hi[2])) > width:
                problems.append(f"{name}: spreads to {lo}..{hi}")
            if not BUDGET[0] <= got["triangles"] <= BUDGET[1]:
                problems.append(f"{name}: {got['triangles']} triangles outside {BUDGET}")

    if args.previews:
        try:
            from PIL import Image, ImageDraw

            previews = os.path.join(OUT, "previews")
            for style in STYLES:
                ores = [o for o in spec["ore"] if o["style"] == style]
                if not ores:
                    continue
                tiles = [(f"{o['id']}_{v}", o) for o in ores for v in ("surface", "deep")]
                w, h = 320, 240
                cols = 6
                rows = (len(tiles) + cols - 1) // cols
                sheet = Image.new("RGB", (w * cols, h * rows), (40, 40, 40))
                draw = ImageDraw.Draw(sheet)
                for n, (name, o) in enumerate(tiles):
                    im = Image.open(os.path.join(previews, f"{name}.png")).convert("RGB").resize((w, h))
                    x, y = (n % cols) * w, (n // cols) * h
                    sheet.paste(im, (x, y))
                    draw.text((x + 6, y + 4), f"{o['name']} ({o['formula']})  {name.split('_')[-1]}", fill=(255, 255, 255))
                sheet.save(os.path.join(previews, f"sheet_{style}.png"))
        except ImportError:
            print("Pillow not installed; skipping contact sheets")

    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} problem(s)")
    print(f"ok: {2 * len(spec['ore'])} ore models within spec")


if __name__ == "__main__":
    main()
