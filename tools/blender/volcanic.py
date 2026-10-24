"""Generate Ascension's volcanic bombs in Blender: lava rocks of several sizes, thrown out of an
erupting crater.

Run headless (tools/build_volcanic.py does this for you):

    blender --background --factory-startup --python tools/blender/volcanic.py -- <out dir>

Each bomb is a rounded lump of dark basalt crust (centred on the origin: it tumbles about its
middle), stretched and twisted a little as it flew, split by a network of cracks with molten lava
showing through: long winding cracks, with finer ones branching off them in places. The lava is
two emissive materials, a bright core down the middle of each crack and a dimmer red edge where it
meets the crust, so the cracks glow hottest deep inside. Writes `<out>/bomb_<n>.glb` from the
smallest to the largest and `<out>/report.json` (bounds and triangle counts).
"""

import json
import math
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bmesh  # noqa: E402
import bpy  # noqa: E402
from bl_common import MATERIALS, export, material, measure  # noqa: E402
from mathutils import Vector, noise  # noqa: E402
from nature import Mesh, g2b, smoothstep  # noqa: E402

# Radii (metres) of the bombs, smallest first.
SIZES = [0.3, 0.5, 0.8, 1.2, 1.8]
# Where the crack field changes material: crust below the first, the cracks' red edge between,
# their bright core above the second.
LEVELS = (0.5, 0.7)


def line_distance(fn, d):
    """Roughly how far (radians over the sphere) direction `d` is from the nearest zero line of
    the field `fn`: its value over the size of its gradient along the surface. Thresholding this,
    rather than the field itself, keeps a crack the same width all along (a raw field is wide and
    blobby where it is flat)."""
    e = 0.01
    value = fn(d)
    grad = Vector((
        fn(d + Vector((e, 0.0, 0.0))) - fn(d - Vector((e, 0.0, 0.0))),
        fn(d + Vector((0.0, e, 0.0))) - fn(d - Vector((0.0, e, 0.0))),
        fn(d + Vector((0.0, 0.0, e))) - fn(d - Vector((0.0, 0.0, e))),
    )) / (2.0 * e)
    along = grad - d * grad.dot(d)
    return abs(value) / max(along.length, 1e-3)


def crack_field(d, off, freq, width):
    """How deep in a crack direction `d` is (0 crust .. 1 the middle of a crack), for cracks
    `width` radians across their glowing edges. Cracks are the zero lines of a warped noise field
    (so they wind and join up rather than lie straight), and finer ones, along the zero lines of a
    second field, branch off where a third lets them."""
    warp = lambda q: q + 0.3 * noise.noise_vector(q * 1.7 + off * 0.5)
    main = line_distance(lambda q: noise.noise(warp(q) * freq + off * 0.3), d)
    s = 1.0 - smoothstep(0.0, width, main)
    where = smoothstep(-0.05, 0.35, noise.noise(d * 1.4 + off * 1.1))
    if where > 0.0:
        fine = line_distance(lambda q: noise.noise(warp(q) * freq * 2.3 + off * 0.9), d)
        s = max(s, 0.9 * where * (1.0 - smoothstep(0.0, width * 0.6, fine)))
    return s


def banded(m, tri, field, mats):
    """Emit triangle `tri` (bmesh verts, with their crack field values) as up to three faces, cut
    exactly along the field's `LEVELS` contours, each with its band's material. The field is linear
    over a triangle, so each band is the convex polygon of the perimeter points (corners, and the
    contour crossings on the edges) whose field lies within it. Crossings are shared by key, so
    neighbouring triangles meet without cracks."""
    points = []
    for i in range(3):
        a, b = tri[i], tri[(i + 1) % 3]
        fa, fb = field[a], field[b]
        points.append((a, fa))
        crossings = []
        for level in LEVELS:
            if (fa - level) * (fb - level) < 0.0:
                t = (level - fa) / (fb - fa)
                key = (min(a.index, b.index), max(a.index, b.index), level)
                if key not in m.crossings:
                    m.crossings[key] = m.bm.verts.new(a.co.lerp(b.co, t))
                crossings.append((t, m.crossings[key], level))
        points.extend((v, level) for _, v, level in sorted(crossings, key=lambda c: c[0]))
    bounds = (-math.inf, *LEVELS, math.inf)
    for band in range(3):
        lo, hi = bounds[band], bounds[band + 1]
        poly = []
        for v, f in points:
            if lo - 1e-6 <= f <= hi + 1e-6 and (not poly or poly[-1] is not v):
                poly.append(v)
        if len(poly) > 2 and poly[0] is poly[-1]:
            poly.pop()
        if len(poly) < 3:
            continue
        face = m.bm.faces.new(poly)
        face.smooth = True
        face.material_index = m.mat_index(mats[band])


def build(m, rng, radius, n):
    crust = material("basalt_crust", (0.035, 0.032, 0.03), 0.92)
    edge = material("lava_edge", (0.2, 0.04, 0.01), 0.75, emission=((0.6, 0.09, 0.015), 1.0))
    core = material("lava_crack", (0.9, 0.3, 0.05), 0.6, emission=((1.0, 0.42, 0.08), 1.0))
    off = Vector((rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(-100, 100)))
    # Stretched along x as it flew, a little flattened, and twisted about its length.
    radii = (radius * rng.uniform(1.15, 1.4), radius * rng.uniform(0.8, 0.92), radius * rng.uniform(0.88, 1.0))
    twist = rng.uniform(-0.5, 0.5)
    # A few long cracks on every bomb (bigger ones a few more, and finer for their size), on a
    # finer mesh for the bigger ones so the cracks keep smooth outlines.
    freq = 1.5 + 0.25 * n
    width = 0.16 - 0.016 * n
    subdivisions = 4 if n < 2 else 5
    # The cracks gape: their middle sits this far (of the radius) below the crust, whose plates
    # bulge a little between them.
    depth, swell = 0.045, 0.015
    ico = bmesh.new()
    bmesh.ops.create_icosphere(ico, subdivisions=subdivisions, radius=1.0)
    ico.verts.index_update()
    # The crack field, smoothed a little over neighbouring points: sampled point by point it
    # jitters, which gives the cut outlines a saw edge. (More than this thins the cracks to pieces.)
    crack = {v: crack_field(v.co.normalized(), off, freq, width) for v in ico.verts}
    crack = {
        v: 0.7 * crack[v] + 0.3 * sum(crack[e.other_vert(v)] for e in v.link_edges) / len(v.link_edges)
        for v in ico.verts
    }
    made, field = {}, {}
    for v in ico.verts:
        d = v.co.normalized()
        # Rounded lumps: smooth noise on a warped direction (no sharp creases), and fine grain.
        w = d + 0.35 * noise.noise_vector(d * 1.3 + off)
        r = 1.0 + 0.15 * noise.noise(w * 1.5 + off) + 0.05 * noise.noise(w * 3.6 + off * 1.3)
        r += 0.012 * noise.noise(d * 11.0 + off * 2.0)
        s = crack[v]
        r += swell * (1.0 - s) - depth * s
        # Its ends taper (a spindle), and it's twisted about its length.
        r *= 1.0 - 0.15 * d.x ** 4
        p = Vector((d.x * radii[0], d.y * radii[1], d.z * radii[2])) * r
        a = twist * d.x
        p = Vector((p.x, p.y * math.cos(a) - p.z * math.sin(a), p.y * math.sin(a) + p.z * math.cos(a)))
        made[v] = m.bm.verts.new(g2b(p))
        field[made[v]] = s
    m.bm.verts.index_update()
    m.crossings = {}
    for f in ico.faces:
        banded(m, [made[v] for v in f.verts], field, (crust, edge, core))
    ico.free()


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
        # It tumbles: no darkening towards a bottom it doesn't have.
        m.root_shade = (1.0, 0.0)
        build(m, random.Random(1000 + n), radius, n)
        obj = m.finish()
        report[name] = measure(obj) | {"radius": radius}
        export(obj, os.path.join(out_dir, f"{name}.glb"))
        print(f"PART {name}: {report[name]['triangles']} triangles", flush=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("PARTS DONE", flush=True)


if __name__ == "__main__":
    main()
