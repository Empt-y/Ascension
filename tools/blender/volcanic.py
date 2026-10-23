"""Generate Ascension's volcanic bombs in Blender: lava rocks of several sizes, thrown out of an
erupting crater.

Run headless (tools/build_volcanic.py does this for you):

    blender --background --factory-startup --python tools/blender/volcanic.py -- <out dir>

Each bomb is a lumpy, rugged lump of dark basalt (centred on the origin: it tumbles about its
middle) with meandering cracks of glowing lava sunk into it: an emissive material, so the cracks
light up and bloom. Writes `<out>/bomb_<n>.glb` from the smallest to the largest and
`<out>/report.json` (bounds and triangle counts).
"""

import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from bl_common import MATERIALS, export, material, measure  # noqa: E402
from nature import Mesh, blob  # noqa: E402

# Radii (metres) of the bombs, smallest first.
SIZES = [0.3, 0.5, 0.8, 1.2, 1.8]


def build(m, rng, radius):
    crust = material("basalt_crust", (0.035, 0.032, 0.03), 0.92)
    glow = material("lava_crack", (0.9, 0.28, 0.04), 0.6, emission=((1.0, 0.33, 0.05), 1.0))
    # Not quite round: bombs are stretched and twisted in flight.
    radii = (radius * rng.uniform(1.0, 1.3), radius * rng.uniform(0.75, 0.95), radius * rng.uniform(0.85, 1.1))
    # Cracks: meandering lines (frequency per metre, width, relief: sunk in by a fraction of the
    # radius), denser on small bombs so each shows a few.
    cracks = (2.4 / radius ** 0.5, 0.09, -0.06 * radius)
    blob(m, (0.0, 0.0, 0.0), radii, crust, rng, subdiv=4, amp=0.22, freq=1.5, mat2=glow, veins=cracks, detail=0.07)


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    out_dir = argv[0]
    os.makedirs(out_dir, exist_ok=True)
    report = {}
    for n, radius in enumerate(SIZES):
        name = f"bomb_{n}"
        bpy.ops.wm.read_factory_settings(use_empty=True)
        MATERIALS.clear()
        m = Mesh(name)
        build(m, random.Random(1000 + n), radius)
        obj = m.finish()
        report[name] = measure(obj) | {"radius": radius}
        export(obj, os.path.join(out_dir, f"{name}.glb"))
        print(f"PART {name}: {report[name]['triangles']} triangles", flush=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("PARTS DONE", flush=True)


if __name__ == "__main__":
    main()
