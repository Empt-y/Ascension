"""Generate Ascension's ore nodes in Blender: a surface outcrop and an underground node per ore.

Run headless (tools/build_ores.py does this for you):

    blender --background --factory-startup --python tools/blender/ores.py -- <ores.toml> <out dir> [previews] [ids]

For every `[[ore]]` in the spec it builds:
  <id>_surface.glb  a small, partly buried outcrop showing a little of the ore, like GregTech's
                    surface indicator rocks over a vein
  <id>_deep.glb     the vein's node underground: a larger host rock on a flat base (y = 0), so it
                    sits on a cave floor or, turned, against a wall, with much more ore showing

The ore's look follows its `style`: metal veins and nuggets, hexagonal crystals, cubic crystals,
earthy lumps or banded strata. Writes `<out>/report.json` (bounds and triangle counts); with
`previews` also renders `<out>/previews/<id>_<variant>.png`.
"""

import json
import os
import random
import sys
import tomllib
import zlib

from mathutils import Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bpy  # noqa: E402
from bl_common import MATERIALS, export, material, measure, render_preview  # noqa: E402
from nature import UP, Mesh, blob, random_unit, tube  # noqa: E402


def host_mats(deep):
    if deep:
        return material("deep_rock_a", (0.11, 0.105, 0.1), 0.85), material("deep_rock_b", (0.075, 0.07, 0.065), 0.9)
    return material("rock_a", (0.25, 0.24, 0.24), 0.85), material("rock_b", (0.17, 0.165, 0.16), 0.9)


class Host:
    """The rock an ore sits in: an ellipsoid we can find surface points on."""

    def __init__(self, m, rng, deep, ore_mat, share=0.0, bands=0.0, veins=None):
        a, b = host_mats(deep)
        if deep:
            rx = rng.uniform(0.8, 1.15)
            self.radii = (rx, rx * rng.uniform(0.6, 0.8), rx * rng.uniform(0.75, 1.0))
            # Flat base at y = 0 so the node stands on a cave floor (or, turned, on a wall).
            self.center = Vector((0.0, self.radii[1] * 0.55, 0.0))
            floor = 0.0
        else:
            rx = rng.uniform(0.45, 0.7)
            self.radii = (rx, rx * rng.uniform(0.55, 0.75), rx * rng.uniform(0.7, 1.0))
            # Mostly buried: just the top shows above the ground.
            self.center = Vector((0.0, self.radii[1] * 0.25, 0.0))
            floor = None
        mat2, mix = (ore_mat, share) if share > 0 or bands > 0 or veins else (b, 0.35)
        # Fine (5120 faces) so bands, veins and patches have smooth edges; rugged, with a second
        # layer of smaller lumps.
        blob(m, self.center, self.radii, a, rng, subdiv=5, amp=0.24, freq=1.3, floor=floor, mat2=mat2, mix=mix,
             bands=bands, veins=veins, detail=0.05)
        self.floor = 0.0 if deep else -0.05

    def surface(self, d, inset=0.0):
        """A point on the host's surface in direction `d` (pulled in by `inset` metres)."""
        d = d.normalized()
        p = self.center + Vector((d.x * self.radii[0], d.y * self.radii[1], d.z * self.radii[2])) * 0.95
        return p - d * inset

    def up_direction(self, rng, spread=1.0, min_y=0.15):
        """A random outward direction over the upper part of the host."""
        while True:
            d = random_unit(rng)
            d.y = abs(d.y) * spread + (1.0 - spread)
            d.normalize()
            if d.y >= min_y:
                return d


def crystal(m, base, axis, length, radius, mat):
    """A hexagonal prism with a pointed tip, starting a little inside the rock at `base`."""
    axis = axis.normalized()
    start = base - axis * radius
    pts = [start, start + axis * length * 0.72, start + axis * length]
    tube(m, pts, [radius, radius * 0.95, radius * 0.04], 6, mat, cap_end=True, smooth=False)


def cube(m, center, size, rng, mat):
    """A randomly turned cube."""
    x = random_unit(rng)
    y = x.cross(random_unit(rng)).normalized()
    z = x.cross(y)
    h = size / 2
    corners = [center + (x * sx + y * sy + z * sz) * h for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
    v = m.verts(corners)
    for quad in ((0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)):
        m.face([v[i] for i in quad], mat, smooth=False, away=center)


def nuggets(m, rng, host, mat, count, size):
    for _ in range(count):
        d = host.up_direction(rng, spread=0.7, min_y=-0.1)
        r = rng.uniform(*size)
        p = host.surface(d, inset=r * 0.15)
        blob(m, p, (r, r * rng.uniform(0.6, 0.9), r * rng.uniform(0.7, 1.0)), mat, rng, subdiv=2, amp=0.25, smooth=False)


def build(m, rng, ore, deep):
    mat = material(f"ore_{ore['id']}", tuple(ore["color"]), ore["roughness"], ore["metallic"])
    style = ore["style"]
    if style == "metal":
        # Meandering veins through the rock, standing proud of it, plus nuggets.
        host = Host(m, rng, deep, mat, veins=(1.6 if deep else 1.9, 0.3 if deep else 0.24, 0.05))
        nuggets(m, rng, host, mat, rng.randint(8, 12) if deep else rng.randint(3, 5), (0.07, 0.16 if deep else 0.11))
    elif style == "earthy":
        host = Host(m, rng, deep, mat, share=0.55 if deep else 0.4)
        nuggets(m, rng, host, mat, rng.randint(9, 13) if deep else rng.randint(4, 6), (0.1, 0.22 if deep else 0.15))
    elif style == "banded":
        host = Host(m, rng, deep, mat, bands=0.09 if deep else 0.07)
        nuggets(m, rng, host, mat, rng.randint(3, 5) if deep else 2, (0.06, 0.12))
    elif style == "crystal":
        host = Host(m, rng, deep, mat, share=0.12)
        clusters = rng.randint(2, 3) if deep else rng.randint(1, 2)
        for _ in range(clusters):
            center = host.up_direction(rng, spread=0.6, min_y=0.25)
            for _ in range(rng.randint(4, 7) if deep else rng.randint(3, 5)):
                d = (center + random_unit(rng) * 0.35).normalized()
                base = host.surface(d)
                axis = (d + UP * 0.4 + random_unit(rng) * 0.25).normalized()
                length = rng.uniform(0.35, 0.85) if deep else rng.uniform(0.18, 0.4)
                crystal(m, base, axis, length, length * rng.uniform(0.14, 0.2), mat)
    elif style == "cubic":
        host = Host(m, rng, deep, mat, share=0.15)
        for _ in range(rng.randint(10, 16) if deep else rng.randint(4, 7)):
            d = host.up_direction(rng, spread=0.7, min_y=0.0)
            size = rng.uniform(0.18, 0.42) if deep else rng.uniform(0.12, 0.26)
            cube(m, host.surface(d, inset=size * 0.15), size, rng, mat)
    else:
        raise ValueError(f"{ore['id']}: unknown style {style}")
    m.root_shade = (0.3 if deep else 0.2, 0.3)


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    spec_path, out_dir = argv[0], argv[1]
    previews = len(argv) > 2 and argv[2] == "previews"
    only = set(argv[3].split(",")) if len(argv) > 3 and argv[3] else None
    with open(spec_path, "rb") as f:
        spec = tomllib.load(f)
    os.makedirs(os.path.join(out_dir, "previews"), exist_ok=True)
    report_path = os.path.join(out_dir, "report.json")
    report = {}
    if only and os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
    for ore in spec["ore"]:
        if only and ore["id"] not in only:
            continue
        for variant, deep in (("surface", False), ("deep", True)):
            name = f"{ore['id']}_{variant}"
            bpy.ops.wm.read_factory_settings(use_empty=True)
            MATERIALS.clear()
            m = Mesh(name)
            # Seeded from the ore's id, so a model stays the same when other ores are added.
            build(m, random.Random(zlib.crc32(name.encode())), ore, deep)
            obj = m.finish()
            report[name] = measure(obj)
            export(obj, os.path.join(out_dir, f"{name}.glb"))
            if previews:
                render_preview(obj, os.path.join(out_dir, "previews", f"{name}.png"))
            print(f"PART {name}: {report[name]['triangles']} triangles", flush=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("PARTS DONE", flush=True)


if __name__ == "__main__":
    main()
