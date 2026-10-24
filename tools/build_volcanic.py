"""Regenerate the volcanic bomb models with Blender and check them.

    python tools/build_volcanic.py [--blender PATH]

Runs tools/blender/volcanic.py headless into assets/models/volcanic, then checks each bomb's size
against its intended radius and its triangle budget.
"""

import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "models", "volcanic")
DEFAULT_BLENDER = r"C:\Users\theas\tools\blender-5.2.2\blender.exe"
BUDGET = (200, 20000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER))
    args = ap.parse_args()
    script = os.path.join(ROOT, "tools", "blender", "volcanic.py")
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
        r = got["radius"]
        extent = max(max(abs(a), abs(b)) for a, b in zip(got["min"], got["max"]))
        if not 0.6 * r <= extent <= 1.8 * r:
            problems.append(f"{name}: reaches {extent:.2f} m for a {r} m bomb")
        if not BUDGET[0] <= got["triangles"] <= BUDGET[1]:
            problems.append(f"{name}: {got['triangles']} triangles outside {BUDGET}")
        print(f"{name}: radius {r} m, reaches {extent:.2f} m, {got['triangles']} triangles")
    if problems:
        sys.exit("\n".join(problems))
    print("ok")


if __name__ == "__main__":
    main()
