"""Regenerate the natural scatter models with Blender and check them against the spec.

    python tools/build_nature.py [--blender PATH] [--only id1,id2]

Runs tools/blender/nature.py headless, then checks each model's height against
assets/data/nature.toml, that nothing sinks too far below the ground or spreads too wide, and its
triangle budget for its kind. Assembles labelled contact sheets per kind in
assets/models/nature/previews.
"""

import argparse
import json
import os
import subprocess
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "assets", "data", "nature.toml")
OUT = os.path.join(ROOT, "assets", "models", "nature")
DEFAULT_BLENDER = r"C:\Users\theas\tools\blender-5.2.2\blender.exe"
# Scatter models are drawn by the thousand, so small things get small budgets.
BUDGETS = {
    "tree": (200, 4500),
    "shrub": (100, 2500),
    "plant": (40, 2000),
    "grass": (20, 600),
    "rock": (40, 1500),
    "debris": (60, 1500),
}
MAX_SINK = 0.6
MAX_HALF_WIDTH = 6.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER))
    ap.add_argument("--only", default="", help="comma-separated model ids to rebuild")
    args = ap.parse_args()

    script = os.path.join(ROOT, "tools", "blender", "nature.py")
    cmd = [args.blender, "--background", "--factory-startup", "--python", script, "--", SPEC, OUT, args.only]
    print("running Blender...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    for line in result.stdout.splitlines():
        if line.startswith(("PART ", "PARTS DONE", "Traceback")) or "Error" in line:
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
    for model in spec["model"]:
        got = report.get(model["id"])
        if got is None:
            problems.append(f"{model['id']}: missing")
            continue
        lo, hi = got["min"], got["max"]
        h0, h1 = model["height"]
        if not h0 <= hi[1] <= h1:
            problems.append(f"{model['id']}: height {hi[1]:.2f} outside {h0}..{h1}")
        if lo[1] < -MAX_SINK:
            problems.append(f"{model['id']}: sinks to {lo[1]:.2f}")
        if max(abs(v) for v in (lo[0], lo[2], hi[0], hi[2])) > MAX_HALF_WIDTH:
            problems.append(f"{model['id']}: spreads to {lo}..{hi}")
        b0, b1 = BUDGETS[model["kind"]]
        if not b0 <= got["triangles"] <= b1:
            problems.append(f"{model['id']}: {got['triangles']} triangles outside {model['kind']} budget {b0}..{b1}")

    try:
        from PIL import Image, ImageDraw

        previews = os.path.join(OUT, "previews")
        kinds = {}
        for model in spec["model"]:
            kinds.setdefault(model["kind"], []).append(model["id"])
        for kind, ids in kinds.items():
            images = [Image.open(os.path.join(previews, f"{i}.png")).convert("RGB") for i in ids]
            w, h = images[0].size
            w, h = w // 2, h // 2
            cols = min(4, len(images))
            rows = (len(images) + cols - 1) // cols
            sheet = Image.new("RGB", (w * cols, h * rows), (40, 40, 40))
            draw = ImageDraw.Draw(sheet)
            for n, (im, model_id) in enumerate(zip(images, ids)):
                x, y = (n % cols) * w, (n // cols) * h
                sheet.paste(im.resize((w, h)), (x, y))
                tris = report.get(model_id, {}).get("triangles", "?")
                draw.text((x + 6, y + 4), f"{model_id}  {tris} tris", fill=(255, 255, 255))
            sheet.save(os.path.join(previews, f"sheet_{kind}.png"))
    except ImportError:
        print("Pillow not installed; skipping contact sheets")

    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} problem(s)")
    print(f"ok: {len(spec['model'])} models within spec")


if __name__ == "__main__":
    main()
