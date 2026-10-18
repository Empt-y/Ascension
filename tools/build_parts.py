"""Regenerate the building-part models with Blender and check them against the spec.

    python tools/build_parts.py [--blender PATH]

Runs tools/blender/parts.py headless, then checks every part's bounding box against
assets/data/building.toml (within 1 mm) and its triangle budget, and assembles a preview contact
sheet per tier in assets/models/previews.
"""

import argparse
import json
import os
import subprocess
import sys
import tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "assets", "data", "building.toml")
OUT = os.path.join(ROOT, "assets", "models")
DEFAULT_BLENDER = r"C:\Users\theas\tools\blender-5.2.2\blender.exe"
# A floor catches broken (near-empty) exports; the ceiling keeps parts cheap to instance.
TRIANGLE_BUDGET = (100, 6000)
TOLERANCE = 0.001


def expected_bounds(part):
    w, h, d = part["size"]
    if part["origin"] == "top":
        y0, y1 = -h, 0.0
    else:
        y0, y1 = 0.0, h
    return [-w / 2, y0, -d / 2], [w / 2, y1, d / 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER))
    args = ap.parse_args()

    script = os.path.join(ROOT, "tools", "blender", "parts.py")
    cmd = [args.blender, "--background", "--factory-startup", "--python", script, "--", SPEC, OUT]
    print("running Blender...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith(("PART ", "PARTS DONE", "Error", "Traceback")) or "Error" in line:
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
    for tier in spec["tier"]:
        for part_id, part in spec["parts"].items():
            name = f"{tier['id']}_{part_id}"
            got = report.get(name)
            if got is None:
                problems.append(f"{name}: missing")
                continue
            lo, hi = expected_bounds(part)
            for axis in range(3):
                if abs(got["min"][axis] - lo[axis]) > TOLERANCE or abs(got["max"][axis] - hi[axis]) > TOLERANCE:
                    problems.append(f"{name}: bounds {got['min']}..{got['max']} != {lo}..{hi}")
                    break
            tris = got["triangles"]
            if not TRIANGLE_BUDGET[0] <= tris <= TRIANGLE_BUDGET[1]:
                problems.append(f"{name}: {tris} triangles outside budget {TRIANGLE_BUDGET}")

    try:
        from PIL import Image

        previews = os.path.join(OUT, "previews")
        parts = list(spec["parts"])
        for tier in spec["tier"]:
            images = [Image.open(os.path.join(previews, f"{tier['id']}_{p}.png")).convert("RGB") for p in parts]
            w, h = images[0].size
            sheet = Image.new("RGB", (w * len(images), h))
            for i, im in enumerate(images):
                sheet.paste(im, (i * w, 0))
            sheet.save(os.path.join(previews, f"sheet_{tier['id']}.png"))
    except ImportError:
        print("Pillow not installed; skipping contact sheets")

    if problems:
        print("\n".join(problems))
        sys.exit(f"{len(problems)} problem(s)")
    print(f"ok: {len(report)} parts within spec")


if __name__ == "__main__":
    main()
