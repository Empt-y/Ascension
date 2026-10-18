"""Generate Ascension's natural scatter models (trees, shrubs, plants, grass, rocks, debris).

Run headless (tools/build_nature.py does this for you):

    blender --background --factory-startup --python tools/blender/nature.py -- <nature.toml> <out dir>

Each `[[model]]` in the spec names a builder below and a seed; the builder grows the model from a
few organic primitives (tapered tubes, noise-displaced blobs, jagged cone skirts and double-sided
ribbons for leaves and blades). Exports `<out>/<id>.glb`, renders `<out>/previews/<id>.png` and
writes `<out>/report.json`.

Geometry is authored in game coordinates (y up, metres) with the origin on the ground at the
model's base; trunks, stems and rocks sink a little below y = 0 so they sit into slopes. The
engine culls back faces, so thin things (leaves, petals, blades) get a reversed copy of each face.
"""

import json
import math
import os
import random
import sys
import tomllib

import bmesh
import bpy
from mathutils import Vector, noise

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bl_common import MATERIALS, export, game_to_blender, material, measure, render_preview  # noqa: E402

UP = Vector((0.0, 1.0, 0.0))
DOWN = Vector((0.0, -1.0, 0.0))
TAU = 2.0 * math.pi


def g2b(p):
    return game_to_blender(p[0], p[1], p[2])


def horizontal(angle):
    return Vector((math.cos(angle), 0.0, math.sin(angle)))


def lerp(a, b, t):
    return a + (b - a) * t


# ---------------------------------------------------------------------------------------------
# Mesh kit


class Mesh:
    """One model's geometry, built straight into a bmesh with per-face materials."""

    def __init__(self, name):
        self.name = name
        self.bm = bmesh.new()
        self.mats = []

    def mat_index(self, mat):
        for i, m in enumerate(self.mats):
            if m == mat:
                return i
        self.mats.append(mat)
        return len(self.mats) - 1

    def verts(self, points):
        return [self.bm.verts.new(g2b(p)) for p in points]

    def face(self, verts, mat, smooth=True, away=None, facing=None):
        """A face, flipped if needed so it points away from game point `away` or along game
        direction `facing`."""
        f = self.bm.faces.new(verts)
        f.material_index = self.mat_index(mat)
        f.smooth = smooth
        f.normal_update()
        if away is not None:
            if f.normal.dot(f.calc_center_median() - g2b(away)) < 0:
                f.normal_flip()
        elif facing is not None:
            if f.normal.dot(g2b(facing)) < 0:
                f.normal_flip()
        return f

    def finish(self):
        mesh = bpy.data.meshes.new(self.name)
        self.bm.normal_update()
        self.bm.to_mesh(mesh)
        self.bm.free()
        for m in self.mats:
            mesh.materials.append(m)
        obj = bpy.data.objects.new(self.name, mesh)
        bpy.context.collection.objects.link(obj)
        return obj


def perpendicular(t):
    a = Vector((1.0, 0.0, 0.0)) if abs(t.x) < 0.9 else Vector((0.0, 0.0, 1.0))
    return t.cross(a).normalized()


def tube(m, points, radii, sides, mat, cap_end=True, cap_start=False, cap_mat=None, smooth=True, mat_fn=None):
    """A tube through `points` with a radius per point, framed by parallel transport so it doesn't
    twist. `mat_fn(segment, side)` can vary the material per face."""
    pts = [Vector(p) for p in points]
    n = len(pts)
    tangents = [(pts[min(i + 1, n - 1)] - pts[max(i - 1, 0)]).normalized() for i in range(n)]
    normal = perpendicular(tangents[0])
    rings = []
    for i in range(n):
        t = tangents[i]
        normal = (normal - t * normal.dot(t)).normalized()
        binormal = t.cross(normal)
        ring = [
            pts[i] + (normal * math.cos(TAU * k / sides) + binormal * math.sin(TAU * k / sides)) * radii[i]
            for k in range(sides)
        ]
        rings.append(m.verts(ring))
    for i in range(n - 1):
        center = (pts[i] + pts[i + 1]) / 2
        for k in range(sides):
            k2 = (k + 1) % sides
            face_mat = mat_fn(i, k) if mat_fn else mat
            m.face([rings[i][k], rings[i][k2], rings[i + 1][k2], rings[i + 1][k]], face_mat, smooth, away=center)
    if cap_end:
        m.face(list(rings[-1]), cap_mat or mat, smooth=False, facing=tangents[-1])
    if cap_start:
        m.face(list(rings[0]), cap_mat or mat, smooth=False, facing=-tangents[0])


def blob(
    m,
    center,
    radii,
    mat,
    rng,
    subdiv=3,
    amp=0.15,
    freq=1.4,
    smooth=True,
    floor=None,
    top_mat=None,
    top=0.6,
    mat2=None,
    mix=0.0,
):
    """A noise-displaced ellipsoid. `floor` flattens it at a height; `top_mat` paints faces whose
    normal points up past `top` (snow, moss); `mat2` paints a noise-chosen share `mix` of faces."""
    tmp = bmesh.new()
    bmesh.ops.create_icosphere(tmp, subdivisions=subdiv, radius=1.0)
    off = Vector((rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(-100, 100)))
    c = Vector(center)
    made = {}
    for v in tmp.verts:
        d = v.co.normalized()
        r = 1.0 + amp * noise.noise(d * freq + off) + amp * 0.5 * noise.noise(d * freq * 2.3 + off * 1.7)
        p = Vector((d.x * radii[0], d.y * radii[1], d.z * radii[2])) * r + c
        if floor is not None:
            p.y = max(p.y, floor)
        made[v] = m.bm.verts.new(g2b(p))
    # The icosphere's coordinates are reused as game directions; game_to_blender is a proper
    # rotation, so its outward winding survives.
    for f in tmp.faces:
        face = m.bm.faces.new([made[v] for v in f.verts])
        face.smooth = smooth
        face.normal_update()
        chosen = mat
        if mat2 is not None and noise.noise(face.calc_center_median() * 1.7 + off) > 1.0 - 2.0 * mix:
            chosen = mat2
        if top_mat is not None and face.normal.z > top:
            chosen = top_mat
        face.material_index = m.mat_index(chosen)
    tmp.free()


def ribbon(m, points, widths, side, mat, smooth=True, double=True):
    """A strip through `points`, `widths[i]` across along `side` (a vector, or one per point). A
    zero width makes a point. Double-sided unless `double` is False."""
    pts = [Vector(p) for p in points]
    sides = side if isinstance(side, list) else [side] * len(pts)
    rows = []
    for p, w, s in zip(pts, widths, sides):
        rows.append([p] if w < 1e-5 else [p - s * (w / 2), p + s * (w / 2)])
    for back in ([False, True] if double else [False]):
        made = [m.verts(r) for r in rows]
        for i in range(len(rows) - 1):
            a, b = made[i], made[i + 1]
            tangent = pts[i + 1] - pts[i]
            front = tangent.cross(sides[i]).normalized()
            if back:
                front = -front
            if len(a) == 2 and len(b) == 2:
                vs = [a[0], a[1], b[1], b[0]]
            elif len(a) == 2:
                vs = [a[0], a[1], b[0]]
            elif len(b) == 2:
                vs = [a[0], b[1], b[0]]
            else:
                continue
            m.face(vs, mat, smooth, facing=front)


def branch(m, start, direction, length, radius, mat, rng, segments=3, sides=6, curl=0.25, tip=0.45):
    """A gently curving tapered limb. Returns its end point."""
    d = Vector(direction).normalized()
    s = Vector(start)
    pts, radii = [], []
    for i in range(segments + 1):
        t = i / segments
        wobble = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1))) * 0.06 * length * t
        pts.append(s + d * length * t + UP * curl * length * t * t + wobble)
        radii.append(radius * lerp(1.0, tip, t))
    tube(m, pts, radii, sides, mat, cap_end=True)
    return pts[-1]


def skirt(m, center, radius, height, mat, rng, points=9, droop=0.25, upper=None):
    """One tier of a conifer: a jagged cone with drooping tips and a shallow underside. `upper`
    paints its upper half (snow)."""
    c = Vector(center)
    rot = rng.uniform(0, TAU)
    apex = c + UP * height
    under = c + UP * height * 0.22
    rim, mid = [], []
    for k in range(points * 2):
        a = rot + math.pi * k / points
        tip = k % 2 == 0
        r = radius * (1.0 if tip else 0.62) * rng.uniform(0.9, 1.08)
        p = c + horizontal(a) * r + UP * (-droop * radius * 0.35 if tip else 0.0)
        rim.append(p)
        mid.append(p.lerp(apex, 0.5) + horizontal(a) * radius * 0.06)
    rv, mv = m.verts(rim), m.verts(mid)
    (av,) = m.verts([apex])
    (uv,) = m.verts([under])
    count = len(rim)
    for k in range(count):
        k2 = (k + 1) % count
        m.face([rv[k], rv[k2], mv[k2], mv[k]], mat, away=c + UP * (height * 0.1))
        m.face([mv[k], mv[k2], av], upper or mat, away=c + UP * (height * 0.4))
        m.face([rv[k2], rv[k], uv], mat, facing=DOWN)


def random_unit(rng):
    while True:
        v = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
        if 0.01 < v.length <= 1.0:
            return v.normalized()


# ---------------------------------------------------------------------------------------------
# Palettes (linear base colours)


def bark(name, color, roughness=0.92):
    return material(f"bark_{name}", color, roughness)


def leaf(name, color, roughness=0.7):
    return material(f"leaf_{name}", color, roughness)


def snow_mat():
    return material("snow", (0.82, 0.86, 0.92), 0.55)


def rock_mats():
    return material("rock_a", (0.25, 0.24, 0.24), 0.85), material("rock_b", (0.17, 0.165, 0.16), 0.9)


# ---------------------------------------------------------------------------------------------
# Trees


def broadleaf(m, rng, leaves, bark_mat, crown_scale=1.0):
    height = rng.uniform(6.8, 8.4)
    trunk_top = height * rng.uniform(0.4, 0.48)
    lean = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.15, 0.45)
    pts = [Vector((0, -0.4, 0)) + lean * (i / 5) ** 2 + UP * ((trunk_top + 0.4) * i / 5) for i in range(6)]
    tube(m, pts, [0.44, 0.33, 0.29, 0.26, 0.24, 0.22], 8, bark_mat, cap_end=False)
    top = pts[-1]
    ends = []
    count = rng.randint(3, 5)
    for b in range(count):
        a = TAU * b / count + rng.uniform(-0.4, 0.4)
        out = (horizontal(a) + UP * rng.uniform(0.6, 1.1)).normalized()
        start = top - UP * rng.uniform(0.0, 0.9)
        ends.append(branch(m, start, out, rng.uniform(1.6, 2.5), 0.15, bark_mat, rng))
    crown = top + UP * (height - trunk_top) * 0.42
    r = 2.1 * crown_scale
    blob(m, crown, (r, r * 0.8, r), leaves[0], rng, amp=0.16)
    for e in ends:
        rr = rng.uniform(1.25, 1.75) * crown_scale
        blob(m, e + UP * 0.25, (rr, rr * 0.82, rr), rng.choice(leaves), rng, amp=0.18)


def oak(m, rng):
    leaves = [leaf("oak_a", (0.08, 0.24, 0.035)), leaf("oak_b", (0.14, 0.31, 0.05))]
    broadleaf(m, rng, leaves, bark("oak", (0.12, 0.08, 0.05)))


def maple(m, rng):
    leaves = [
        leaf("maple_a", (0.62, 0.16, 0.03)),
        leaf("maple_b", (0.72, 0.32, 0.04)),
        leaf("maple_c", (0.48, 0.07, 0.03)),
    ]
    broadleaf(m, rng, leaves, bark("maple", (0.14, 0.10, 0.07)), crown_scale=1.08)


def birch(m, rng):
    white = bark("birch", (0.72, 0.70, 0.65), 0.75)
    mark = bark("birch_mark", (0.06, 0.055, 0.05))
    leaves = [leaf("birch_a", (0.20, 0.38, 0.06)), leaf("birch_b", (0.30, 0.44, 0.08))]
    height = rng.uniform(8.5, 10.0)
    sway = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.2, 0.5)
    n = 14
    pts = [Vector((0, -0.3, 0)) + sway * math.sin(math.pi * i / n) + UP * ((height * 0.88 + 0.3) * i / n) for i in range(n + 1)]
    radii = [lerp(0.2, 0.06, i / n) for i in range(n + 1)]
    marks = {(i, k) for i in range(n) for k in range(7) if rng.random() < 0.16}
    tube(m, pts, radii, 7, white, mat_fn=lambda i, k: mark if (i, k) in marks else white)
    for b in range(rng.randint(4, 6)):
        i = rng.randint(n // 2, n - 2)
        out = (horizontal(rng.uniform(0, TAU)) + UP * rng.uniform(0.9, 1.6)).normalized()
        branch(m, pts[i], out, rng.uniform(0.9, 1.5), 0.06, white, rng, sides=5)
    def on_trunk(t):
        x = t * n
        i = min(int(x), n - 1)
        return pts[i].lerp(pts[i + 1], x - i)

    for b in range(rng.randint(6, 8)):
        p = on_trunk(rng.uniform(0.5, 0.9)) + horizontal(rng.uniform(0, TAU)) * rng.uniform(0.25, 0.65)
        r = rng.uniform(0.85, 1.15)
        blob(m, p, (r, r * rng.uniform(1.15, 1.45), r), rng.choice(leaves), rng, amp=0.2)
    blob(m, pts[-1], (0.9, 1.2, 0.9), leaves[1], rng, amp=0.2)


def conifer(m, rng, needles, bark_mat, height, snow=None):
    tiers = rng.randint(6, 8)
    base = 1.4
    top = height - 1.0
    pts = [Vector((0, -0.3, 0)), Vector((0, base, 0)), Vector((0, top * 0.6, 0)), Vector((0, top, 0))]
    tube(m, pts, [0.3, 0.24, 0.14, 0.05], 7, bark_mat)
    for i in range(tiers):
        t = i / (tiers - 1)
        y = lerp(base, top - 0.6, t ** 0.9)
        radius = lerp(2.5, 0.65, t) * rng.uniform(0.9, 1.1)
        h = lerp(2.2, 1.5, t)
        skirt(m, (0, y, 0), radius, h, needles[i % len(needles)], rng, upper=snow)
    skirt(m, (0, top - 0.2, 0), 0.35, 1.3, needles[0], rng, points=6, droop=0.0, upper=snow)


def pine(m, rng):
    needles = [leaf("pine_a", (0.035, 0.12, 0.05)), leaf("pine_b", (0.05, 0.16, 0.06))]
    conifer(m, rng, needles, bark("pine", (0.13, 0.07, 0.045)), rng.uniform(10.0, 13.0))


def snow_pine(m, rng):
    needles = [leaf("snow_pine_a", (0.03, 0.09, 0.05)), leaf("snow_pine_b", (0.04, 0.11, 0.06))]
    conifer(m, rng, needles, bark("pine", (0.13, 0.07, 0.045)), rng.uniform(8.5, 11.5), snow=snow_mat())


def palm(m, rng):
    trunk_mat = bark("palm", (0.30, 0.22, 0.13), 0.85)
    fronds = [leaf("palm_a", (0.10, 0.30, 0.05)), leaf("palm_b", (0.16, 0.34, 0.07))]
    nut = material("coconut", (0.22, 0.13, 0.05), 0.8)
    height = rng.uniform(6.0, 7.5)
    lean = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.8, 1.8)
    n = 14
    pts = [Vector((0, -0.3, 0)) + lean * (i / n) ** 1.8 + UP * ((height + 0.3) * i / n) for i in range(n + 1)]
    radii = [lerp(0.27, 0.16, i / n) * (1.0 if i % 2 == 0 else 0.86) for i in range(n + 1)]
    tube(m, pts, radii, 7, trunk_mat)
    top = pts[-1]
    blob(m, top, (0.3, 0.25, 0.3), fronds[0], rng, subdiv=1, amp=0.1)
    count = rng.randint(8, 10)
    for f in range(count):
        a = TAU * f / count + rng.uniform(-0.25, 0.25)
        d = horizontal(a)
        side = UP.cross(d).normalized()
        length = rng.uniform(2.6, 3.4)
        rise = rng.uniform(0.35, 0.6)
        steps = 8
        spine = [top + d * length * (i / steps) + UP * length * (rise * i / steps - 0.85 * (i / steps) ** 2) for i in range(steps + 1)]
        mat = rng.choice(fronds)
        ribbon(m, spine, [0.07] * steps + [0.0], side, mat)
        for i in range(1, steps):
            t = i / steps
            p = spine[i]
            tangent = (spine[i + 1] - spine[i - 1]).normalized()
            span = 0.95 * math.sin(math.pi * t) + 0.15
            for s in (-1, 1):
                out = (side * s + DOWN * 0.55 + tangent * 0.45).normalized()
                ribbon(m, [p, p + out * span * 0.55, p + out * span * 1.1], [0.2, 0.17, 0.0], tangent, mat)
    for k in range(3):
        a = TAU * k / 3 + rng.uniform(0, 1)
        blob(m, top + horizontal(a) * 0.22 - UP * 0.25, (0.15, 0.17, 0.15), nut, rng, subdiv=1, amp=0.05)


def willow(m, rng):
    bark_mat = bark("willow", (0.17, 0.13, 0.09))
    leaves = [leaf("willow_a", (0.18, 0.32, 0.07)), leaf("willow_b", (0.25, 0.38, 0.09))]
    trunk = [Vector((0, -0.4, 0)), Vector((0.1, 1.0, 0)), Vector((-0.1, 2.2, 0.15)), Vector((0, 3.0, 0))]
    tube(m, trunk, [0.5, 0.38, 0.33, 0.3], 8, bark_mat, cap_end=False)
    for b in range(4):
        out = (horizontal(TAU * b / 4 + rng.uniform(-0.3, 0.3)) + UP * 0.8).normalized()
        branch(m, trunk[-1], out, rng.uniform(1.8, 2.4), 0.16, bark_mat, rng)
    crown = Vector((0, 5.3, 0))
    blob(m, crown, (2.9, 1.5, 2.9), leaves[0], rng, amp=0.2)
    for _ in range(3):
        p = crown + horizontal(rng.uniform(0, TAU)) * 1.4 + UP * rng.uniform(0.2, 0.8)
        blob(m, p, (1.4, 0.9, 1.4), leaves[1], rng, amp=0.2)
    for s in range(40):
        a = rng.uniform(0, TAU)
        r = rng.uniform(1.4, 2.9)
        start = crown + horizontal(a) * r - UP * 0.4 * (r / 2.9) ** 2
        length = rng.uniform(2.0, 3.6)
        sway = horizontal(a) * rng.uniform(0.1, 0.4)
        strand = [start + sway * (i / 4) ** 2 - UP * length * i / 4 for i in range(5)]
        if strand[-1].y < 0.8:
            continue
        ribbon(m, strand, [0.32, 0.3, 0.24, 0.14, 0.0], horizontal(a + math.pi / 2), rng.choice(leaves))


def dead_tree(m, rng):
    bark_mat = bark("dead", (0.21, 0.18, 0.15), 0.95)

    def limb(start, d, length, radius, depth):
        end = branch(m, start, d, length, radius, bark_mat, rng, sides=4 + depth, curl=0.1, tip=0.55)
        if depth == 0:
            return
        for _ in range(rng.randint(2, 3)):
            s = Vector(start).lerp(end, rng.uniform(0.55, 1.0))
            nd = (d + random_unit(rng) * 0.9 + UP * 0.25).normalized()
            limb(s, nd, length * rng.uniform(0.5, 0.7), radius * 0.55, depth - 1)

    trunk_h = rng.uniform(2.4, 3.4)
    lean = horizontal(rng.uniform(0, TAU)) * 0.3
    pts = [Vector((0, -0.4, 0)), Vector((0, 1.0, 0)) + lean * 0.3, Vector((0, trunk_h, 0)) + lean]
    tube(m, pts, [0.36, 0.27, 0.2], 7, bark_mat, cap_end=False)
    for _ in range(rng.randint(2, 3)):
        d = (horizontal(rng.uniform(0, TAU)) + UP * rng.uniform(0.8, 1.6)).normalized()
        limb(pts[-1], d, rng.uniform(1.6, 2.3), 0.16, 2)


# ---------------------------------------------------------------------------------------------
# Shrubs


def bush(m, rng):
    leaves = [leaf("bush_a", (0.07, 0.22, 0.04)), leaf("bush_b", (0.12, 0.28, 0.05))]
    blobs = []
    for _ in range(rng.randint(4, 6)):
        c = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.0, 0.5) + UP * rng.uniform(0.35, 0.7)
        r = rng.uniform(0.42, 0.68)
        blob(m, c, (r, r * 0.85, r), rng.choice(leaves), rng, amp=0.2, freq=2.0, floor=-0.05)
        blobs.append((c, r))
    return blobs


def berry_bush(m, rng):
    berry = material("berry", (0.55, 0.02, 0.04), 0.35)
    blobs = bush(m, rng)
    for _ in range(18):
        c, r = rng.choice(blobs)
        d = random_unit(rng)
        d.y = abs(d.y) * 0.8 + 0.2
        d.normalize()
        p = c + Vector((d.x * r, d.y * r * 0.85, d.z * r)) * 1.04
        blob(m, p, (0.045, 0.045, 0.045), berry, rng, subdiv=1, amp=0.0)


def juniper(m, rng):
    leaves = [leaf("juniper_a", (0.04, 0.13, 0.10)), leaf("juniper_b", (0.06, 0.16, 0.11))]
    for _ in range(rng.randint(5, 7)):
        c = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.0, 0.9) + UP * rng.uniform(0.2, 0.42)
        r = rng.uniform(0.55, 0.9)
        blob(m, c, (r, r * rng.uniform(0.45, 0.6), r), rng.choice(leaves), rng, amp=0.28, freq=2.6, smooth=False, floor=-0.05)


def dry_shrub(m, rng):
    twig = bark("dry", (0.30, 0.24, 0.16), 0.95)

    def limb(start, d, length, radius, depth):
        end = branch(m, start, d, length, radius, twig, rng, segments=2, sides=4, curl=0.15, tip=0.5)
        if depth == 0:
            return
        for _ in range(2):
            s = Vector(start).lerp(end, rng.uniform(0.5, 1.0))
            limb(s, (d + random_unit(rng) * 0.8 + UP * 0.3).normalized(), length * 0.65, radius * 0.6, depth - 1)

    for k in range(rng.randint(4, 6)):
        d = (horizontal(rng.uniform(0, TAU)) * rng.uniform(0.3, 0.8) + UP).normalized()
        limb(Vector((0, -0.05, 0)), d, rng.uniform(0.45, 0.65), 0.028, 2)


# ---------------------------------------------------------------------------------------------
# Plants


def fern(m, rng):
    greens = [leaf("fern_a", (0.07, 0.24, 0.04)), leaf("fern_b", (0.10, 0.30, 0.05))]
    for f in range(rng.randint(8, 11)):
        a = TAU * f / 10 + rng.uniform(-0.3, 0.3)
        d = horizontal(a)
        side = UP.cross(d).normalized()
        length = rng.uniform(0.7, 1.05)
        steps = 7
        spine = [d * length * 0.85 * (i / steps) + UP * length * (1.5 * i / steps - 0.95 * (i / steps) ** 2) for i in range(steps + 1)]
        mat = rng.choice(greens)
        ribbon(m, spine, [0.035] * steps + [0.0], side, mat)
        for i in range(1, steps):
            t = i / steps
            p = spine[i]
            tangent = (spine[i + 1] - spine[i - 1]).normalized()
            span = 0.24 * (1.0 - t) + 0.06
            for s in (-1, 1):
                out = (side * s + tangent * 0.45 + DOWN * 0.12).normalized()
                ribbon(m, [p, p + out * span * 0.5, p + out * span], [0.05, 0.07, 0.0], tangent, mat)


def stems_with_heads(m, rng, count, height, stem_mat, head, leaves=None, bend=0.08):
    for _ in range(count):
        base = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.0, 0.15)
        h = rng.uniform(*height)
        lean = horizontal(rng.uniform(0, TAU)) * bend
        pts = [base - UP * 0.03, base + UP * h * 0.5 + lean * 0.4, base + UP * h + lean]
        tube(m, pts, [0.011, 0.009, 0.007], 4, stem_mat, cap_end=False)
        head(pts[-1], (pts[-1] - pts[-2]).normalized())
    if leaves:
        base_leaves(m, rng, leaves)


def base_leaves(m, rng, mat):
    for _ in range(rng.randint(3, 5)):
        d = horizontal(rng.uniform(0, TAU))
        length = rng.uniform(0.12, 0.2)
        pts = [Vector((0, 0.0, 0)), d * length * 0.5 + UP * length * 0.35, d * length + UP * length * 0.3]
        ribbon(m, pts, [0.02, 0.045, 0.0], UP.cross(d).normalized(), mat)


def petal_ring(m, center, count, length, width, mat, rng, lift=0.15, cup=0.0):
    rot = rng.uniform(0, TAU)
    for k in range(count):
        d = horizontal(rot + TAU * k / count)
        side = UP.cross(d).normalized()
        p1 = center + d * length * 0.5 + UP * length * (lift + cup * 0.6)
        p2 = center + d * length + UP * length * (lift * 1.4 + cup)
        ribbon(m, [center, p1, p2], [width * 0.5, width, width * 0.4], side, mat)


def daisies(m, rng):
    stem = leaf("stem", (0.08, 0.22, 0.04))
    white = material("petal_white", (0.85, 0.85, 0.8), 0.6)
    yellow = material("petal_yellow", (0.75, 0.5, 0.02), 0.6)

    def head(p, d):
        petal_ring(m, p, 10, 0.06, 0.022, white, rng)
        blob(m, p + UP * 0.008, (0.022, 0.014, 0.022), yellow, rng, subdiv=1, amp=0.0)

    stems_with_heads(m, rng, rng.randint(6, 9), (0.25, 0.45), stem, head, leaves=stem)


def poppies(m, rng):
    stem = leaf("stem", (0.08, 0.22, 0.04))
    red = material("petal_red", (0.65, 0.03, 0.02), 0.55)
    dark = material("petal_dark", (0.03, 0.025, 0.02), 0.7)

    def head(p, d):
        petal_ring(m, p, 5, 0.07, 0.07, red, rng, lift=0.3, cup=0.5)
        blob(m, p + UP * 0.012, (0.016, 0.016, 0.016), dark, rng, subdiv=1, amp=0.0)

    stems_with_heads(m, rng, rng.randint(4, 6), (0.45, 0.65), stem, head, leaves=stem, bend=0.1)


def bluebells(m, rng):
    stem = leaf("stem", (0.08, 0.22, 0.04))
    blue = material("petal_blue", (0.14, 0.12, 0.6), 0.55)
    for _ in range(rng.randint(4, 6)):
        base = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.0, 0.12)
        d = horizontal(rng.uniform(0, TAU))
        h = rng.uniform(0.28, 0.42)
        arch = [base - UP * 0.03 + d * 0.0, base + UP * h * 0.6 + d * 0.02, base + UP * h + d * 0.08, base + UP * h * 0.92 + d * 0.16]
        tube(m, arch, [0.01, 0.008, 0.007, 0.006], 4, stem, cap_end=False)
        for t in (0.35, 0.6, 0.85, 1.0):
            i = min(int(t * 3), 2)
            p = arch[i].lerp(arch[i + 1], t * 3 - i)
            # A small open bell hanging from the stem, double-sided so its inside shows.
            axis = p - UP * 0.025
            ring = [p - UP * 0.04 + horizontal(TAU * k / 6) * 0.022 for k in range(6)]
            for k in range(6):
                tri = [ring[k], ring[(k + 1) % 6], p]
                m.face(m.verts(tri), blue, away=axis)
                m.face(m.verts(tri), blue, facing=axis - (tri[0] + tri[1]) / 2)
    base_leaves(m, rng, stem)


def alpine_flowers(m, rng):
    cushion = leaf("cushion", (0.08, 0.2, 0.06))
    pink = material("petal_pink", (0.55, 0.12, 0.45), 0.55)
    yellow = material("petal_yellow", (0.75, 0.5, 0.02), 0.6)
    stem = leaf("stem", (0.08, 0.22, 0.04))
    blob(m, (0, 0.0, 0), (0.13, 0.055, 0.13), cushion, rng, amp=0.25, freq=3.0, floor=-0.02)

    def head(p, d):
        petal_ring(m, p, 5, 0.03, 0.022, pink, rng, lift=0.1)
        blob(m, p + UP * 0.004, (0.008, 0.006, 0.008), yellow, rng, subdiv=1, amp=0.0)

    stems_with_heads(m, rng, rng.randint(8, 12), (0.08, 0.2), stem, head, bend=0.03)


def mushroom(m, rng, base, stem_h, cap_r, cap_mat, stem_mat, spots=None, subdiv=3):
    b = Vector(base)
    lean = horizontal(rng.uniform(0, TAU)) * stem_h * 0.12
    tube(m, [b - UP * 0.02, b + UP * stem_h * 0.5 + lean * 0.5, b + UP * stem_h + lean], [cap_r * 0.24, cap_r * 0.2, cap_r * 0.22], 6, stem_mat, cap_end=False)
    c = b + UP * stem_h + lean
    blob(m, c, (cap_r, cap_r * 0.62, cap_r), cap_mat, rng, subdiv=subdiv, amp=0.06, floor=c.y - cap_r * 0.08)
    if spots:
        for _ in range(7):
            d = random_unit(rng)
            d.y = abs(d.y) * 0.7 + 0.3
            d.normalize()
            p = c + Vector((d.x * cap_r, d.y * cap_r * 0.62, d.z * cap_r)) * 1.0
            blob(m, p, (cap_r * 0.16, cap_r * 0.07, cap_r * 0.16), spots, rng, subdiv=1, amp=0.0)


def red_mushroom(m, rng):
    red = material("cap_red", (0.6, 0.04, 0.02), 0.45)
    white = material("mushroom_white", (0.8, 0.78, 0.7), 0.7)
    mushroom(m, rng, (0, 0, 0), rng.uniform(0.2, 0.26), 0.13, red, white, spots=white)
    mushroom(m, rng, horizontal(rng.uniform(0, TAU)) * 0.16, 0.1, 0.07, red, white, spots=white)


def brown_mushrooms(m, rng):
    cap = material("cap_brown", (0.28, 0.16, 0.08), 0.6)
    stem = material("mushroom_stem", (0.6, 0.55, 0.45), 0.75)
    for _ in range(rng.randint(5, 7)):
        base = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.0, 0.14)
        mushroom(m, rng, base, rng.uniform(0.04, 0.14), rng.uniform(0.03, 0.06), cap, stem, subdiv=2)


def blades(m, rng, count, height, mats, spread=0.12, lean=(0.2, 0.5), width=0.035, segments=3):
    """A tuft of grass blades. Returns the tips."""
    tips = []
    for _ in range(count):
        base = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.0, spread)
        d = horizontal(rng.uniform(0, TAU))
        h = rng.uniform(*height)
        bend = rng.uniform(*lean)
        pts = [base - UP * 0.02 + d * bend * h * (i / segments) ** 2 + UP * h * (i / segments) * (1 - 0.25 * bend * (i / segments)) for i in range(segments + 1)]
        widths = [width * lerp(1.0, 0.45, i / segments) for i in range(segments)] + [0.0]
        side = (UP.cross(d) + d * rng.uniform(-0.3, 0.3)).normalized()
        ribbon(m, pts, widths, side, rng.choice(mats))
        tips.append(pts[-1])
    return tips


def cattails(m, rng):
    green = [leaf("reed_a", (0.12, 0.26, 0.06)), leaf("reed_b", (0.18, 0.3, 0.08))]
    brown = material("cattail", (0.2, 0.1, 0.04), 0.85)
    blades(m, rng, rng.randint(12, 16), (0.9, 1.5), green, spread=0.2, lean=(0.15, 0.4), width=0.05)
    for _ in range(rng.randint(3, 5)):
        base = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.0, 0.15)
        h = rng.uniform(1.2, 1.7)
        lean = horizontal(rng.uniform(0, TAU)) * 0.08
        tube(m, [base - UP * 0.03, base + UP * h * 0.6 + lean * 0.4, base + UP * h + lean], [0.012, 0.01, 0.006], 4, green[0])
        top = base + UP * (h - 0.12) + lean
        tube(m, [top - UP * 0.2, top - UP * 0.1, top], [0.03, 0.034, 0.03], 6, brown, cap_start=True)


# ---------------------------------------------------------------------------------------------
# Grass


def grass(m, rng):
    mats = [leaf("grass_a", (0.10, 0.32, 0.05)), leaf("grass_b", (0.17, 0.38, 0.06)), leaf("grass_c", (0.24, 0.40, 0.07))]
    blades(m, rng, rng.randint(10, 16), (0.22, 0.45), mats)


def tall_grass(m, rng):
    mats = [leaf("grass_a", (0.10, 0.32, 0.05)), leaf("grass_b", (0.17, 0.38, 0.06)), leaf("grass_dry", (0.36, 0.38, 0.12))]
    seed = material("seed_head", (0.42, 0.34, 0.16), 0.8)
    tips = blades(m, rng, rng.randint(12, 16), (0.65, 1.0), mats, spread=0.15, lean=(0.15, 0.4), width=0.03)
    for p in tips[:5]:
        blob(m, p - UP * 0.05, (0.018, 0.07, 0.018), seed, rng, subdiv=1, amp=0.0)


def beach_grass(m, rng):
    mats = [leaf("beach_grass_a", (0.32, 0.32, 0.13)), leaf("beach_grass_b", (0.24, 0.30, 0.10))]
    blades(m, rng, rng.randint(14, 18), (0.45, 0.85), mats, spread=0.18, lean=(0.4, 0.8), width=0.03)


def alpine_grass(m, rng):
    mats = [leaf("alpine_grass_a", (0.16, 0.22, 0.07)), leaf("alpine_grass_b", (0.26, 0.26, 0.10))]
    blades(m, rng, rng.randint(14, 20), (0.1, 0.26), mats, spread=0.14, lean=(0.3, 0.7), width=0.025)


# ---------------------------------------------------------------------------------------------
# Rocks and debris


def rock(m, rng, top_mat=None, top=0.5, size=(0.9, 1.3)):
    a, b = rock_mats()
    rx = rng.uniform(*size)
    ry = rx * rng.uniform(0.55, 0.8)
    rz = rx * rng.uniform(0.7, 1.0)
    blob(m, (0, ry * 0.5, 0), (rx, ry, rz), a, rng, amp=0.24, freq=1.1, smooth=False, top_mat=top_mat, top=top, mat2=b, mix=0.45)


def boulder(m, rng):
    rock(m, rng)


def mossy_rock(m, rng):
    rock(m, rng, top_mat=leaf("moss", (0.08, 0.22, 0.04), 0.9), top=0.55, size=(0.7, 1.1))


def snow_rock(m, rng):
    rock(m, rng, top_mat=snow_mat(), top=0.4, size=(0.7, 1.1))


def pebbles(m, rng):
    a, b = rock_mats()
    for _ in range(rng.randint(6, 9)):
        c = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.0, 0.45)
        r = rng.uniform(0.05, 0.14)
        blob(m, c + UP * r * 0.35, (r, r * 0.6, r * 0.85), rng.choice((a, b)), rng, subdiv=2, amp=0.15, smooth=False)


def lying_log(m, rng, length, radius, bark_mat, end_mat, stubs):
    d = horizontal(rng.uniform(0, TAU))
    side = UP.cross(d).normalized()
    bow = rng.uniform(-0.15, 0.15)
    pts = [d * length * (i / 4 - 0.5) + side * bow * math.sin(math.pi * i / 4) + UP * radius * 0.85 for i in range(5)]
    radii = [radius * lerp(1.0, 0.85, i / 4) for i in range(5)]
    tube(m, pts, radii, 9, bark_mat, cap_end=True, cap_start=True, cap_mat=end_mat)
    for _ in range(stubs):
        i = rng.randint(1, 3)
        out = (side * rng.choice((-1, 1)) + UP * rng.uniform(0.2, 0.9)).normalized()
        branch(m, pts[i], out, radius * rng.uniform(1.6, 2.6), radius * 0.35, bark_mat, rng, segments=2, sides=5)


def log(m, rng):
    lying_log(m, rng, rng.uniform(3.0, 3.8), 0.28, bark("oak", (0.12, 0.08, 0.05)), material("wood_end", (0.45, 0.32, 0.18), 0.8), 3)


def driftwood(m, rng):
    pale = bark("drift", (0.48, 0.45, 0.40), 0.9)
    lying_log(m, rng, rng.uniform(1.8, 2.4), 0.12, pale, pale, 2)


def stump(m, rng):
    bark_mat = bark("oak", (0.12, 0.08, 0.05))
    end = material("wood_end", (0.45, 0.32, 0.18), 0.8)
    h = rng.uniform(0.4, 0.6)
    tube(m, [Vector((0, -0.25, 0)), Vector((0, h * 0.5, 0)), Vector((0, h, 0))], [0.42, 0.36, 0.34], 9, bark_mat, cap_mat=end)
    for k in range(rng.randint(4, 5)):
        d = horizontal(TAU * k / 5 + rng.uniform(-0.3, 0.3))
        tube(m, [d * 0.2 + UP * 0.28, d * 0.45 + UP * 0.06, d * 0.72 - UP * 0.1, d * 0.85 - UP * 0.2], [0.17, 0.12, 0.07, 0.04], 6, bark_mat)


BUILDERS = {
    f.__name__: f
    for f in (
        oak, maple, birch, pine, snow_pine, palm, willow, dead_tree,
        bush, berry_bush, juniper, dry_shrub,
        fern, daisies, poppies, bluebells, alpine_flowers, red_mushroom, brown_mushrooms, cattails,
        grass, tall_grass, beach_grass, alpine_grass,
        boulder, mossy_rock, snow_rock, pebbles, log, stump, driftwood,
    )
}  # fmt: skip


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    spec_path, out_dir = argv[0], argv[1]
    only = set(argv[2].split(",")) if len(argv) > 2 and argv[2] else None
    with open(spec_path, "rb") as f:
        spec = tomllib.load(f)
    os.makedirs(os.path.join(out_dir, "previews"), exist_ok=True)
    report_path = os.path.join(out_dir, "report.json")
    report = {}
    if only and os.path.exists(report_path):
        with open(report_path) as f:
            report = json.load(f)
    for model in spec["model"]:
        if only and model["id"] not in only:
            continue
        bpy.ops.wm.read_factory_settings(use_empty=True)
        MATERIALS.clear()
        m = Mesh(model["id"])
        BUILDERS[model["builder"]](m, random.Random(model["seed"]))
        obj = m.finish()
        report[model["id"]] = measure(obj)
        export(obj, os.path.join(out_dir, f"{model['id']}.glb"))
        render_preview(obj, os.path.join(out_dir, "previews", f"{model['id']}.png"))
        print(f"PART {model['id']}: {report[model['id']]['triangles']} triangles", flush=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("PARTS DONE", flush=True)


main()
