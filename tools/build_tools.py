"""Regenerate the handheld tool models with Blender and check them.

    python tools/build_tools.py [--blender PATH]

Runs tools/blender/tools.py headless into assets/models/tools, checks each model's size and
triangle budget, and puts the previews side by side in assets/models/previews/tools.png.
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "models", "tools")
DEFAULT_BLENDER = r"C:\Users\theas\tools\blender-5.2.2\blender.exe"
# Largest extent each may have (metres), and the triangle budget.
SIZE = {"beacon": 1.0, "multitool": 0.7}
MAX_SIZE = 0.45
BUDGET = 12000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER))
    args = ap.parse_args()
    script = os.path.join(ROOT, "tools", "blender", "tools.py")
    cmd = [args.blender, "--background", "--factory-startup", "--python", script, "--", OUT]
    print("running Blender...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0 or "PARTS DONE" not in result.stdout:
        print(result.stdout[-4000:])
        print(result.stderr[-4000:])
        sys.exit("Blender failed")
    with open(os.path.join(OUT, "report.json")) as f:
        report = json.load(f)
    problems = []
    for name, got in sorted(report.items()):
        extent = max(hi - lo for lo, hi in zip(got["min"], got["max"]))
        limit = SIZE.get(name, MAX_SIZE)
        print(f"{name:24} {extent:5.2f} m  {got['triangles']:6} triangles")
        if extent > limit:
            problems.append(f"{name}: {extent:.2f} m across (at most {limit})")
        if got["triangles"] > BUDGET:
            problems.append(f"{name}: {got['triangles']} triangles")
    try:
        from PIL import Image

        names = [n for n in report if not n.startswith("multitool_light")]
        images = [Image.open(os.path.join(OUT, "previews", f"{n}.png")).convert("RGB") for n in names]
        w, h = images[0].size
        cols = 3
        sheet = Image.new("RGB", (w * cols, h * ((len(images) + cols - 1) // cols)))
        for i, im in enumerate(images):
            sheet.paste(im, ((i % cols) * w, (i // cols) * h))
        sheet.save(os.path.join(ROOT, "assets", "models", "previews", "tools.png"))
    except ImportError:
        print("(no Pillow: no contact sheet)")
    if problems:
        sys.exit("\n".join(problems))
    print("ok")


if __name__ == "__main__":
    main()
