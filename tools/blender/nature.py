"""Generate Ascension's natural scatter models (trees, shrubs, plants, grass, rocks, debris).

Run headless (tools/build_nature.py does this for you):

    blender --background --factory-startup --python tools/blender/nature.py -- <nature.toml> <out dir>

With `burnt` after the out dir it instead writes `<id>_burnt1..3.glb` for every tree, shrub, plant and
grass: the same plant (same builder and seed) scorched, charred and burnt out (see `BURN`).

Each `[[model]]` in the spec names a builder below and a seed; the builder grows the model from a
few organic primitives: tapered tubes for trunks and limbs, noise-displaced blobs for rocks, and
leaf cards (double-sided quads and curved strips carrying the cut-out textures drawn by
tools/nature_textures.py) for foliage. Cards get normals pointing out of their crown, so a canopy
lights as one volume, and every vertex gets baked ambient occlusion in its colour's alpha. Exports `<out>/<id>.glb`, renders `<out>/previews/<id>.png` and
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
from bl_common import MATERIALS, export, game_to_blender, measure, render_preview  # noqa: E402
from bl_common import material as plain_material  # noqa: E402

UP = Vector((0.0, 1.0, 0.0))
DOWN = Vector((0.0, -1.0, 0.0))
TAU = 2.0 * math.pi


def g2b(p):
    return game_to_blender(p[0], p[1], p[2])


# ---------------------------------------------------------------------------------------------
# Burning: 0 untouched, 1 scorched (leaves browned, some fallen; bark darkened), 2 charred (most
# leaves gone, the rest shrivelled black-brown; bark charred), 3 burnt out (no leaves, black).

BURN = 0
# Share of leaf cards left at each stage, and of grass blades (burnt grass stays as black stubble).
BURN_KEEP = (1.0, 0.8, 0.35, 0.0)
GRASS_KEEP = (1.0, 0.9, 0.7, 0.5)
# The rule for the model being built: trees and shrubs lose all their leaves; grass and small
# plants keep some as black stubble.
KEEP = BURN_KEEP
# Which leaf cards fall: its own sequence, so the plant itself grows the same.
BURN_RNG = random.Random(0)
WOOD = ("bark_", "wood_", "trunk", "branch")
UNBURNT = ("rock", "sandstone", "snow", "ice")


def burnt_color(name, color):
    """A material's colour at the current burn stage: wood chars, living tissue scorches brown
    then goes black; rock, snow and ice stay as they are."""
    if BURN == 0 or name.startswith(UNBURNT):
        return color
    if name.startswith(WOOD):
        targets = ((0.05, 0.035, 0.025, 0.5), (0.02, 0.017, 0.015, 0.85), (0.014, 0.012, 0.011, 1.0))
    else:
        targets = ((0.30, 0.16, 0.05, 0.7), (0.06, 0.04, 0.025, 0.85), (0.02, 0.018, 0.016, 1.0))
    r, g, b, t = targets[BURN - 1]
    return tuple(lerp(c, x, t) for c, x in zip(color, (r, g, b)))


def material(name, color, roughness, metallic=0.0):
    """`bl_common.material`, coloured for the current burn stage (charred surfaces are rough)."""
    if BURN == 0:
        return plain_material(name, color, roughness, metallic)
    rough = roughness if name.startswith(UNBURNT) else max(roughness, 0.9)
    return plain_material(f"{name}_burnt{BURN}", burnt_color(name, color), rough, metallic)


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
        # Per-vertex normals that override the geometric ones (leaf cards shade as one crown).
        self.custom = {}
        # The mid-distance version of a tree: built from the same random sequence (so it's the same
        # tree), but tubes thinner than MID_MIN_RADIUS aren't drawn and only every other leaf card
        # is, larger. Thin twigs and many small card edges are what shimmer at a distance.
        self.mid = False
        self.cards = 0
        # What the mid version drops: tubes thinner than this, and all but every `card_keep`th
        # card (the rest grow by `card_scale`). Builders whose look depends on them change these.
        self.min_radius = MID_MIN_RADIUS
        self.card_keep = 2
        self.card_scale = MID_CARD_SCALE
        self.uv = self.bm.loops.layers.uv.verify()
        # Baked ambient occlusion (vertex colour alpha): ellipsoids that darken what's deep inside
        # them (crowns), and how much the bottom of the model darkens towards its roots.
        self.occluders = []
        self.root_shade = (0.5, 0.3)

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

    def card(self, corners, mat, normals, uvs=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))):
        """A double-sided textured quad: `corners` in (0,0) (1,0) (1,1) (0,1) texture order (or the
        given `uvs`), with `normals` (one per corner, game space) used on both sides."""
        if BURN and BURN_RNG.random() >= KEEP[BURN]:
            # Burnt off.
            return
        if self.mid:
            self.cards += 1
            if self.cards % self.card_keep:
                return
            centre = sum(corners, Vector()) / 4
            corners = [centre + (c - centre) * self.card_scale for c in corners]
        a, b = corners[1] - corners[0], corners[3] - corners[0]
        front = a.cross(b)
        for facing in (front, -front):
            vs = self.verts(corners)
            f = self.face(vs, mat, smooth=True, facing=facing)
            for loop in f.loops:
                loop[self.uv].uv = uvs[vs.index(loop.vert)]
            for v, n in zip(vs, normals):
                self.custom[v] = g2b(n.normalized())

    def bake_occlusion(self, mesh):
        attr = mesh.color_attributes.new("Col", "FLOAT_COLOR", "POINT")
        height, strength = self.root_shade
        for i, v in enumerate(mesh.vertices):
            p = Vector((v.co.x, v.co.z, -v.co.y))
            ao = lerp(1.0 - strength, 1.0, smoothstep(0.0, height, p.y))
            for center, radii in self.occluders:
                d = Vector(((p.x - center.x) / radii[0], (p.y - center.y) / radii[1], (p.z - center.z) / radii[2])).length
                ao *= lerp(0.4, 1.0, smoothstep(0.15, 1.0, d))
            attr.data[i].color = (1.0, 1.0, 1.0, ao)
        mesh.color_attributes.active_color_name = "Col"

    def finish(self):
        mesh = bpy.data.meshes.new(self.name)
        self.bm.normal_update()
        self.bm.verts.index_update()
        custom = {v.index: n for v, n in self.custom.items()}
        self.bm.to_mesh(mesh)
        self.bm.free()
        for m in self.mats:
            mesh.materials.append(m)
        self.bake_occlusion(mesh)
        if custom:
            normals = []
            for loop in mesh.loops:
                n = custom.get(loop.vertex_index)
                normals.append(n if n is not None else mesh.corner_normals[loop.index].vector.copy())
            mesh.normals_split_custom_set(normals)
        obj = bpy.data.objects.new(self.name, mesh)
        bpy.context.collection.objects.link(obj)
        return obj


def smoothstep(a, b, x):
    t = max(0.0, min(1.0, (x - a) / (b - a)))
    return t * t * (3 - 2 * t)


def perpendicular(t):
    a = Vector((1.0, 0.0, 0.0)) if abs(t.x) < 0.9 else Vector((0.0, 0.0, 1.0))
    return t.cross(a).normalized()


MID_MIN_RADIUS = 0.07
# True while a mid model is built: card textures with a `_mid` version (fewer, wider grass blades)
# use it.
BUILDING_MID = False
MID_TEXTURES_USED = []
MID_CARD_SCALE = 1.4


def tube(m, points, radii, sides, mat, cap_end=True, cap_start=False, cap_mat=None, smooth=True, mat_fn=None, ribs=0.0):
    """A tube through `points` with a radius per point, framed by parallel transport so it doesn't
    twist. `mat_fn(segment, side)` can vary the material per face; `ribs` pulls every other side in
    by that fraction (cactus ribs)."""
    if m.mid and max(radii) < m.min_radius:
        return
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
            pts[i]
            + (normal * math.cos(TAU * k / sides) + binormal * math.sin(TAU * k / sides))
            * radii[i]
            * (1.0 - ribs * (k % 2))
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
    bands=0.0,
    basis=None,
    veins=None,
    detail=0.0,
):
    """A noise-displaced ellipsoid. `floor` flattens it at a height; `top_mat` paints faces whose
    normal points up past `top` (snow, moss); `mat2` paints a noise-chosen share `mix` of faces."""
    tmp = bmesh.new()
    bmesh.ops.create_icosphere(tmp, subdivisions=subdiv, radius=1.0)
    off = Vector((rng.uniform(-100, 100), rng.uniform(-100, 100), rng.uniform(-100, 100)))
    c = Vector(center)
    made = {}
    vein_at = {}
    for v in tmp.verts:
        d = v.co.normalized()
        r = 1.0 + amp * noise.noise(d * freq + off) + amp * 0.5 * noise.noise(d * freq * 2.3 + off * 1.7)
        if detail > 0.0:
            # Fine, sharp-edged roughness (cracks and knobs) on top of the broad shape.
            r += detail * (abs(noise.noise(d * freq * 7.0 + off * 2.3)) - 0.3)
            r += detail * 0.5 * noise.noise(d * freq * 15.0 + off * 3.1)
        if veins is not None:
            # Meandering veins along the contours of a noise field (they branch and wander like
            # real ones), standing a little proud of the rock: `veins` = (frequency, width, relief).
            vfreq, width, relief = veins
            q = Vector((d.x * radii[0], d.y * radii[1], d.z * radii[2]))
            t = abs(noise.noise(q * vfreq + off * 0.3)) + 0.35 * abs(noise.noise(q * vfreq * 2.1 + off * 0.7))
            strength = 1.0 - smoothstep(width * 0.55, width, t)
            vein_at[v] = strength
            r += relief * strength / max(radii)
        if basis is None:
            p = Vector((d.x * radii[0], d.y * radii[1], d.z * radii[2])) * r + c
        else:
            # Axes of a right-handed frame, so the winding survives.
            p = (basis[0] * d.x * radii[0] + basis[1] * d.y * radii[1] + basis[2] * d.z * radii[2]) * r + c
        if floor is not None:
            p.y = max(p.y, floor)
        made[v] = m.bm.verts.new(g2b(p))
    # The icosphere's coordinates are reused as game directions; game_to_blender is a proper
    # rotation, so its outward winding survives.
    #
    # With `mat2`, a scalar field over the vertices decides which material each point gets (above
    # 0.5 is mat2). Triangles that straddle the 0.5 contour are cut exactly along it, so veins,
    # strata and patches get smooth outlines instead of following the triangles.
    field = None
    if mat2 is not None:
        if veins is not None:
            field = {v: vein_at[v] for v in tmp.verts}
        elif bands > 0.0:
            def band(v):
                c = made[v].co
                # Strata `bands` metres thick (Blender z is game y), warped to undulate like real
                # sediment.
                h = c.z + 0.6 * bands * noise.noise(c * 2.2 + off) + 0.25 * (c.x * 0.3 - c.y * 0.2)
                return 0.5 + 0.5 * math.sin(math.pi * h / bands)
            field = {v: band(v) for v in tmp.verts}
        else:
            field = {v: noise.noise(made[v].co * 1.7 + off) - (1.0 - 2.0 * mix) + 0.5 for v in tmp.verts}
    crossings = {}

    def crossing(a, b):
        key = (a.index, b.index) if a.index < b.index else (b.index, a.index)
        if key not in crossings:
            fa, fb = field[a], field[b]
            t = (0.5 - fa) / (fb - fa)
            crossings[key] = m.bm.verts.new(made[a].co.lerp(made[b].co, t))
        return crossings[key]

    tmp.verts.index_update()

    def emit(verts, chosen):
        face = m.bm.faces.new(verts)
        face.smooth = smooth
        face.normal_update()
        if top_mat is not None and face.normal.z > top:
            chosen = top_mat
        face.material_index = m.mat_index(chosen)

    for f in tmp.faces:
        vs = list(f.verts)
        if field is None:
            emit([made[v] for v in vs], mat)
            continue
        inside = [field[v] > 0.5 for v in vs]
        if all(inside) or not any(inside):
            emit([made[v] for v in vs], mat2 if inside[0] else mat)
            continue
        polys = {True: [], False: []}
        for i, v in enumerate(vs):
            w = vs[(i + 1) % len(vs)]
            polys[inside[i]].append(made[v])
            if inside[i] != inside[(i + 1) % len(vs)]:
                c = crossing(v, w)
                polys[True].append(c)
                polys[False].append(c)
        emit(polys[True], mat2)
        emit(polys[False], mat)
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


def random_unit(rng):
    while True:
        v = Vector((rng.uniform(-1, 1), rng.uniform(-1, 1), rng.uniform(-1, 1)))
        if 0.01 < v.length <= 1.0:
            return v.normalized()


# ---------------------------------------------------------------------------------------------
# Textured foliage

TEXTURES = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "assets", "textures", "nature"
)


def card_material(name, texture, tint=(1.0, 1.0, 1.0), roughness=0.75):
    """A cutout material: texels with alpha below 0.5 are discarded (glTF alphaMode MASK). The
    texture is multiplied by `tint`."""
    if BUILDING_MID and os.path.isfile(os.path.join(TEXTURES, f"{texture}_mid.png")):
        texture, name = f"{texture}_mid", f"{name}_mid"
        MID_TEXTURES_USED.append(texture)
    if BURN:
        # Scorched brown, then shrivelled dark (burnt-out plants keep no cards).
        burn_tint = ((1.2, 0.55, 0.2), (0.35, 0.22, 0.12), (0.1, 0.08, 0.06))[BURN - 1]
        tint = tuple(t * b for t, b in zip(tint, burn_tint))
        name = f"{name}_burnt{BURN}"
    key = f"card_{name}"
    if key in MATERIALS:
        return MATERIALS[key]
    mat = bpy.data.materials.new(key)
    mat.use_nodes = True
    nodes, links = mat.node_tree.nodes, mat.node_tree.links
    bsdf = nodes.get("Principled BSDF")
    tex = nodes.new("ShaderNodeTexImage")
    tex.image = bpy.data.images.load(os.path.join(TEXTURES, f"{texture}.png"), check_existing=True)
    mix = nodes.new("ShaderNodeMix")
    mix.data_type = "RGBA"
    mix.blend_type = "MULTIPLY"
    mix.inputs["Factor"].default_value = 1.0
    mix.inputs[7].default_value = (*tint, 1.0)
    links.new(tex.outputs["Color"], mix.inputs[6])
    links.new(mix.outputs[2], bsdf.inputs["Base Color"])
    cutoff = nodes.new("ShaderNodeMath")
    cutoff.operation = "GREATER_THAN"
    cutoff.inputs[1].default_value = 0.5
    links.new(tex.outputs["Alpha"], cutoff.inputs[0])
    links.new(cutoff.outputs[0], bsdf.inputs["Alpha"])
    bsdf.inputs["Roughness"].default_value = roughness
    if hasattr(mat, "surface_render_method"):
        mat.surface_render_method = "DITHERED"
    mat.use_backface_culling = True
    MATERIALS[key] = mat
    return mat


def crown_normal(p, center, radii):
    """Outward from an ellipsoidal crown, so the canopy lights as one soft volume."""
    d = p - center
    return Vector((d.x / radii[0] ** 2, d.y / radii[1] ** 2, d.z / radii[2] ** 2)).normalized()


def quad(base, up, across, length, width):
    return [
        base - across * width / 2,
        base + across * width / 2,
        base + across * width / 2 + up * length,
        base - across * width / 2 + up * length,
    ]


def bounds_of(points, pad):
    lo = Vector((min(p.x for p in points), min(p.y for p in points), min(p.z for p in points)))
    hi = Vector((max(p.x for p in points), max(p.y for p in points), max(p.z for p in points)))
    return (lo + hi) / 2, tuple((hi - lo) / 2 + Vector((pad, pad, pad)))


def hang_cards(m, rng, twigs, mat, size, per_twig, center, radii, outward=0.6, jitter=0.45, up_bias=0.15, floor=None):
    """Leaf cards on each twig `(point, direction)`: each starts at the twig, points mostly along
    it and away from the crown centre, and gets crown normals."""
    for point, direction in twigs:
        for _ in range(per_twig):
            out = (point - center).normalized() if (point - center).length > 1e-3 else UP
            up = (direction * (1.0 - outward) + out * outward + random_unit(rng) * jitter).normalized()
            across = up.cross(random_unit(rng))
            if across.length < 1e-3:
                continue
            across.normalize()
            length = size * rng.uniform(0.8, 1.2)
            base = point + random_unit(rng) * size * 0.15 - up * length * 0.12
            corners = quad(base, up, across, length, length)
            if floor is not None and min(c.y for c in corners) < floor:
                continue
            m.card(corners, mat, [crown_normal(c, center, radii) * (1 - up_bias) + UP * up_bias for c in corners])


def strip_card(m, points, widths, sides, mat, normals):
    """A curved double-sided textured strip: the texture's v runs along `points` (0 at the first),
    u across `widths[i]` along `sides[i]`. `normals[i]` is used at both edges of row i."""
    if BURN and BURN_RNG.random() >= KEEP[BURN]:
        # Burnt off.
        return
    n = len(points)
    for facing_sign in (1.0, -1.0):
        rows = []
        for i in range(n):
            half = sides[i] * (widths[i] / 2)
            rows.append(m.verts([points[i] - half, points[i] + half]))
        for i in range(n - 1):
            a, b = rows[i], rows[i + 1]
            vs = [a[0], a[1], b[1], b[0]]
            front = (points[i + 1] - points[i]).cross(sides[i]) * facing_sign
            f = m.face(vs, mat, smooth=True, facing=front)
            uv_of = {
                a[0]: (0.0, i / (n - 1)),
                a[1]: (1.0, i / (n - 1)),
                b[1]: (1.0, (i + 1) / (n - 1)),
                b[0]: (0.0, (i + 1) / (n - 1)),
            }
            for loop in f.loops:
                loop[m.uv].uv = uv_of[loop.vert]
        for i, row in enumerate(rows):
            for v in row:
                m.custom[v] = g2b(normals[i].normalized())


def grow(m, rng, start, direction, length, radius, depth, bark_mat, twigs, spread=0.8, rise=0.35, children=(2, 3)):
    """A recursive limb; the ends of the finest limbs (and points along them) become twigs."""
    end = branch(m, start, direction, length, radius, bark_mat, rng, sides=max(4, 3 + depth), curl=0.12, tip=0.55)
    if depth == 0:
        twigs.append((end, direction))
        twigs.append((Vector(start).lerp(end, 0.55), direction))
        return
    for _ in range(rng.randint(*children)):
        s = Vector(start).lerp(end, rng.uniform(0.5, 1.0))
        nd = (direction + random_unit(rng) * spread + UP * rise).normalized()
        grow(m, rng, s, nd, length * rng.uniform(0.55, 0.72), radius * 0.6, depth - 1, bark_mat, twigs, spread, rise, children)


def tuft_cards(m, rng, mat, height, width, cards, lean=0.15):
    """A grass tuft: vertical cards crossed round the centre, leaning out a little. Normals point
    mostly up so the tuft lights like the ground it grows from; occlusion darkens the roots."""
    m.card_keep, m.card_scale = 1, 1.0
    rot = rng.uniform(0, TAU)
    for k in range(cards):
        a = rot + math.pi * k / cards + rng.uniform(-0.2, 0.2)
        across = horizontal(a)
        out = UP.cross(across).normalized() * rng.choice((-1, 1))
        h = height * rng.uniform(0.85, 1.15)
        w = width * rng.uniform(0.85, 1.15)
        # Sunk well in, so the card's straight bottom edge stays underground on slopes.
        base = out * rng.uniform(0.0, 0.05) - UP * 0.12
        top = base + UP * (h + 0.09) + out * h * lean * rng.uniform(0.3, 1.0)
        points = [base, base.lerp(top, 0.5) - out * h * lean * 0.15, top]
        normals = [(UP * 0.8 + out * 0.2), (UP * 0.75 + out * 0.25), (UP * 0.7 + out * 0.3)]
        strip_card(m, points, [w, w, w], [across] * 3, mat, normals)
    m.root_shade = (height, 0.45)


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


def broadleaf(m, rng, leaves, bark_mat, height=(7.0, 8.5), trunk=0.38, limbs=(4, 5), card=1.7, per_twig=5, rise=0.35):
    """A branching broadleaf: a trunk, recursive limbs, and leaf cards on every twig."""
    h = rng.uniform(*height)
    trunk_top = h * trunk
    lean = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.15, 0.4)
    pts = [Vector((0, -0.4, 0)) + lean * (i / 5) ** 2 + UP * ((trunk_top + 0.4) * i / 5) for i in range(6)]
    tube(m, pts, [0.44, 0.34, 0.3, 0.27, 0.25, 0.23], 9, bark_mat, cap_end=False)
    twigs = []
    count = rng.randint(*limbs)
    for b in range(count):
        d = (horizontal(TAU * b / count + rng.uniform(-0.4, 0.4)) + UP * rng.uniform(0.7, 1.2)).normalized()
        grow(m, rng, pts[-1] - UP * rng.uniform(0, 0.6), d, rng.uniform(2.2, 2.9) * h / 7.7, 0.17, 2, bark_mat, twigs, rise=rise)
    grow(m, rng, pts[-1], UP, 2.4 * h / 7.7, 0.16, 2, bark_mat, twigs, rise=rise)
    center, radii = bounds_of([p for p, _ in twigs], 0.8)
    hang_cards(m, rng, twigs, leaves, card, per_twig, center, radii)
    m.occluders.append((center, radii))


def oak(m, rng):
    tint = rng.choice([(0.28, 0.5, 0.16), (0.24, 0.46, 0.13), (0.31, 0.52, 0.17)])
    broadleaf(m, rng, card_material(f"oak_{tint[0]}", "leaves_oak", tint), bark("oak", (0.12, 0.08, 0.05)))


def maple(m, rng):
    leaves = card_material("maple_autumn", "leaves_maple", (0.95, 0.42, 0.1))
    broadleaf(m, rng, leaves, bark("maple", (0.14, 0.10, 0.07)), card=1.6, per_twig=6)


def birch(m, rng):
    white = bark("birch", (0.6, 0.58, 0.54), 0.75)
    dark = bark("birch_mark", (0.12, 0.11, 0.10))
    grey = bark("birch_grey", (0.42, 0.40, 0.37), 0.85)
    # Only the trunk is white: the branches and the thin top are dark grey-brown, as on real
    # birches. (Thin white lines against dark foliage flickered at a distance.)
    twig_bark = bark("birch_twig", (0.16, 0.12, 0.1), 0.9)
    leaves = card_material("birch", "leaves_round", (0.36, 0.56, 0.15))
    height = rng.uniform(8.5, 10.0)
    sway = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.2, 0.5)
    # A fine trunk (faces ~10-18 cm) so the bark's marks are small dashes, not panels.
    n, sides = 48, 12
    pts = [Vector((0, -0.3, 0)) + sway * math.sin(math.pi * i / n) + UP * ((height * 0.85 + 0.3) * i / n) for i in range(n + 1)]
    radii = [lerp(0.2, 0.05, i / n) for i in range(n + 1)]
    # Birch marks: short horizontal dashes, one or two faces wide, darker low down.
    marks = {}
    for i in range(n):
        for k in range(sides):
            r = rng.random()
            if r < 0.05 + 0.05 * (1 - i / n):
                tone = dark if rng.random() < 0.7 else grey
                marks[(i, k)] = tone
                if rng.random() < 0.5:
                    marks[(i, (k + 1) % sides)] = tone
    if m.mid:
        # Mid distance: the trunk is a few pixels wide, so its white bark and dark dashes alternate
        # pixel by pixel and flicker. The mid trunk is a little thicker and one plain tone, the
        # average of the bark and its marks, toned down against the dark foliage behind it.
        plain = bark("birch_mid", (0.44, 0.42, 0.39), 0.8)
        tube(m, pts, [r * 1.35 for r in radii], sides, plain, mat_fn=lambda i, k: twig_bark if radii[i] < 0.09 else plain)
    else:
        tube(m, pts, radii, sides, white, mat_fn=lambda i, k: twig_bark if radii[i] < 0.09 else marks.get((i, k), white))
    twigs = [(pts[-1], UP)]
    coarse = 14
    for j in range(coarse // 3, coarse):
        i = j * n // coarse
        for _ in range(2 if j > coarse // 2 else 1):
            d = (horizontal(rng.uniform(0, TAU)) + UP * rng.uniform(0.4, 1.0)).normalized()
            reach = lerp(1.9, 0.8, (j - coarse // 3) / (coarse - coarse // 3)) * rng.uniform(0.8, 1.1)
            grow(m, rng, pts[i], d, reach, 0.045, 1, twig_bark, twigs, spread=0.6, rise=-0.15, children=(2, 3))
    center, crown = bounds_of([p for p, _ in twigs], 0.6)
    hang_cards(m, rng, twigs, leaves, 1.15, 4, center, crown)
    m.occluders.append((center, crown))


def conifer(m, rng, needles, height, snow=None, reach_scale=1.0):
    """Whorls of drooping branches up a straight trunk, each lined with needle sprays. With `snow`,
    upward-facing sprays in the upper crown use the snowy texture."""
    bark_mat = bark("pine", (0.13, 0.07, 0.045))
    trunk = [Vector((0, -0.3, 0)), Vector((0, 1.5, 0)), Vector((0, height * 0.6, 0)), Vector((0, height - 0.3, 0))]
    tube(m, trunk, [0.32, 0.26, 0.15, 0.04], 8, bark_mat)
    center, radii = Vector((0, height * 0.45, 0)), (2.6 * reach_scale, height * 0.55, 2.6 * reach_scale)
    whorls = int(height / 0.75)
    for w in range(whorls):
        t = w / (whorls - 1)
        y = lerp(1.6, height - 0.6, t)
        reach = lerp(2.7, 0.35, t**0.9) * rng.uniform(0.85, 1.1) * reach_scale
        rot = rng.uniform(0, TAU)
        spokes = rng.randint(6, 8)
        for k in range(spokes):
            d = (horizontal(rot + TAU * k / spokes + rng.uniform(-0.25, 0.25)) + DOWN * rng.uniform(0.05, 0.3)).normalized()
            start = Vector((0, y, 0))
            radius = 0.05 * (1.0 - 0.6 * t) + 0.015
            end = branch(m, start, d, reach, radius, bark_mat, rng, segments=2, sides=4, curl=-0.1, tip=0.4)
            steps = max(1, int(reach / 0.55))
            for i in range(steps):
                p = start.lerp(end, (i + 0.7) / steps)
                for flip in (-1, 1):
                    side = d.cross(UP).normalized() * flip
                    up = (d * 0.8 + side * 0.35 + UP * rng.uniform(-0.05, 0.25)).normalized()
                    # Tilt the spray about its own axis so it isn't edge-on from the side.
                    flat = up.cross(UP).normalized()
                    tilt = rng.uniform(0.35, 0.8) * rng.choice((-1, 1))
                    across = flat * math.cos(tilt) + up.cross(flat) * math.sin(tilt)
                    size = 1.3 * (1.0 - 0.4 * t) * rng.uniform(0.85, 1.15)
                    corners = quad(p - up * size * 0.1, up, across, size, size)
                    mat = snow if snow is not None and t > 0.15 and rng.random() < 0.65 else needles
                    m.card(corners, mat, [crown_normal(c, center, radii) * 0.7 + UP * 0.3 for c in corners])
    top = Vector((0, height - 0.9, 0))
    for k in range(3):
        corners = quad(top, UP, horizontal(TAU * k / 3), 1.1, 0.6)
        m.card(corners, snow or needles, [crown_normal(c, center, radii) * 0.5 + UP * 0.5 for c in corners])
    m.occluders.append((center, radii))


def pine(m, rng):
    tint = rng.choice([(0.11, 0.21, 0.12), (0.10, 0.19, 0.13), (0.13, 0.22, 0.11)])
    conifer(m, rng, card_material(f"pine_{tint[1]}", "needles", tint), rng.uniform(10.5, 12.5))


def snow_pine(m, rng):
    needles = card_material("snow_pine", "needles", (0.08, 0.16, 0.11))
    snow = card_material("snow_pine_snowy", "needles_snow")
    conifer(m, rng, needles, rng.uniform(8.5, 11.0), snow=snow, reach_scale=0.85)


def palm(m, rng):
    trunk_mat = bark("palm", (0.30, 0.22, 0.13), 0.85)
    frond = card_material("palm", "palm_frond", (0.32, 0.56, 0.16))
    nut = material("coconut", (0.22, 0.13, 0.05), 0.8)
    height = rng.uniform(6.0, 7.5)
    lean = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.8, 1.8)
    n = 14
    pts = [Vector((0, -0.3, 0)) + lean * (i / n) ** 1.8 + UP * ((height + 0.3) * i / n) for i in range(n + 1)]
    radii = [lerp(0.27, 0.16, i / n) * (1.0 if i % 2 == 0 else 0.86) for i in range(n + 1)]
    tube(m, pts, radii, 7, trunk_mat)
    top = pts[-1]
    count = rng.randint(9, 11)
    for f in range(count):
        a = TAU * f / count + rng.uniform(-0.25, 0.25)
        d = horizontal(a)
        side = UP.cross(d).normalized()
        length = rng.uniform(2.8, 3.6)
        rise = rng.uniform(0.3, 0.65)
        steps = 8
        spine = [top + d * length * (i / steps) + UP * length * (rise * i / steps - 0.85 * (i / steps) ** 2) for i in range(steps + 1)]
        widths = [lerp(0.25, 1.15, math.sin(math.pi * min(1.0, i / steps + 0.1))) * 1.0 for i in range(steps + 1)]
        # Leaflets hang down either side of the rib: tilt the strip's sides downwards.
        sides = [(side + DOWN * 0.15).normalized() for _ in spine]
        normals = [crown_normal(p, top - UP * 0.4, (3.0, 1.5, 3.0)) * 0.6 + UP * 0.4 for p in spine]
        strip_card(m, spine, widths, sides, frond, normals)
    for k in range(3):
        a = TAU * k / 3 + rng.uniform(0, 1)
        blob(m, top + horizontal(a) * 0.22 - UP * 0.25, (0.15, 0.17, 0.15), nut, rng, subdiv=2, amp=0.05)
    m.occluders.append((top - UP * 0.2, (1.6, 0.9, 1.6)))


def willow(m, rng):
    bark_mat = bark("willow", (0.17, 0.13, 0.09))
    leaves = card_material("willow", "leaves_round", (0.42, 0.58, 0.2))
    strand_mat = card_material("willow_strand", "willow_strand", (0.4, 0.56, 0.18))
    trunk = [Vector((0, -0.4, 0)), Vector((0.1, 1.0, 0)), Vector((-0.1, 2.2, 0.15)), Vector((0, 3.0, 0))]
    tube(m, trunk, [0.5, 0.38, 0.33, 0.3], 8, bark_mat, cap_end=False)
    twigs = []
    for b in range(5):
        out = (horizontal(TAU * b / 5 + rng.uniform(-0.3, 0.3)) + UP * 0.9).normalized()
        grow(m, rng, trunk[-1], out, rng.uniform(2.2, 2.8), 0.17, 2, bark_mat, twigs, spread=0.7, rise=0.1)
    center, crown = bounds_of([p for p, _ in twigs], 0.7)
    hang_cards(m, rng, twigs, leaves, 1.3, 3, center, crown)
    # Long strands hanging from the outer twigs.
    for p, _ in twigs:
        out = Vector((p.x - center.x, 0.0, p.z - center.z))
        if out.length < 0.6:
            continue
        out.normalize()
        length = min(rng.uniform(2.2, 3.8), p.y - 0.8)
        if length < 1.0:
            continue
        drift = out * rng.uniform(0.15, 0.4)
        strand = [p + drift * (i / 4) ** 2 - UP * length * i / 4 for i in range(5)]
        across = UP.cross(out).normalized()
        normals = [crown_normal(q, center, crown) * 0.5 + out * 0.5 for q in strand]
        strip_card(m, strand, [0.6] * 5, [across] * 5, strand_mat, normals)
        # A second strand crossing the first, so it doesn't vanish edge-on.
        strip_card(m, strand, [0.6] * 5, [out] * 5, strand_mat, normals)
    m.occluders.append((center, crown))


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


def shrub(m, rng, leaves, stems=(4, 6), reach=(0.55, 0.85), card=0.75, per_twig=6, rise=0.5, spread=1.0):
    """Woody stems from the ground with leaf cards on their twigs. Returns the twigs."""
    stem_mat = bark("shrub", (0.16, 0.12, 0.08))
    twigs = []
    for k in range(rng.randint(*stems)):
        d = (horizontal(rng.uniform(0, TAU)) * spread * rng.uniform(0.4, 1.0) + UP * rise).normalized()
        grow(m, rng, Vector((0, -0.05, 0)), d, rng.uniform(*reach), 0.035, 1, stem_mat, twigs, spread=0.7, rise=0.2)
    center, radii = bounds_of([p for p, _ in twigs], 0.35)
    hang_cards(m, rng, twigs, leaves, card, per_twig, center, radii, outward=0.7, floor=-0.08)
    m.occluders.append((center, radii))
    return twigs


def bush(m, rng):
    texture = rng.choice(["leaves_round", "leaves_oak"])
    tint = rng.choice([(0.25, 0.46, 0.12), (0.3, 0.5, 0.15), (0.22, 0.42, 0.13)])
    shrub(m, rng, card_material(f"bush_{texture}_{tint[0]}", texture, tint))


def berry_bush(m, rng):
    berry = material("berry", (0.55, 0.02, 0.04), 0.35)
    twigs = shrub(m, rng, card_material("berry_bush", "leaves_round", (0.2, 0.4, 0.12)))
    for p, _ in rng.sample(twigs, min(len(twigs), 14)):
        for _ in range(2):
            q = p + random_unit(rng) * 0.12
            blob(m, q, (0.04, 0.04, 0.04), berry, rng, subdiv=1, amp=0.0)


def juniper(m, rng):
    needles = card_material("juniper", "needles", (0.12, 0.2, 0.17))
    shrub(m, rng, needles, stems=(7, 9), reach=(0.6, 0.9), card=0.6, per_twig=6, rise=0.12, spread=1.5)


def dry_shrub(m, rng):
    twig = bark("dry", (0.30, 0.24, 0.16), 0.95)

    thick = 1.6 if m.mid else 1.0

    def limb(start, d, length, radius, depth):
        # Mid version: the finest twigs aren't drawn (thinner than any limb), the rest thicker.
        m.min_radius = 1.0 if depth == 0 else 0.0
        end = branch(m, start, d, length, radius * thick, twig, rng, segments=2, sides=4, curl=0.15, tip=0.5)
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
    tint = rng.choice([(0.26, 0.5, 0.14), (0.3, 0.55, 0.16)])
    mat = card_material(f"fern_{tint[0]}", "fern_frond", tint)
    for f in range(rng.randint(9, 12)):
        a = TAU * f / 10 + rng.uniform(-0.3, 0.3)
        d = horizontal(a)
        side = UP.cross(d).normalized()
        length = rng.uniform(0.75, 1.1)
        steps = 6
        spine = [d * length * 0.85 * (i / steps) + UP * length * (1.5 * i / steps - 0.95 * (i / steps) ** 2) for i in range(steps + 1)]
        widths = [0.38 * length] * (steps + 1)
        normals = [UP * 0.7 + d * 0.3 for _ in spine]
        strip_card(m, spine, widths, [side] * (steps + 1), mat, normals)
    m.root_shade = (0.6, 0.4)


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
    if m.mid:
        # Mid version: separate petals are sub-pixel specks at a distance; one flat disc of the same
        # colour and spread reads the same and stays steady.
        rim = [center + horizontal(rot + TAU * k / 6) * length * 0.8 + UP * length * (lift + cup * 0.6) for k in range(6)]
        for k in range(6):
            tri = [Vector(center), rim[k], rim[(k + 1) % 6]]
            m.face(m.verts(tri), mat, facing=UP)
            m.face(m.verts(tri), mat, facing=-UP)
        return
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
    m.min_radius = 0.0
    if m.mid:
        subdiv = max(1, subdiv - 1)
    b = Vector(base)
    lean = horizontal(rng.uniform(0, TAU)) * stem_h * 0.12
    tube(m, [b - UP * 0.02, b + UP * stem_h * 0.5 + lean * 0.5, b + UP * stem_h + lean], [cap_r * 0.24, cap_r * 0.2, cap_r * 0.22], 6, stem_mat, cap_end=False)
    c = b + UP * stem_h + lean
    blob(m, c, (cap_r, cap_r * 0.62, cap_r), cap_mat, rng, subdiv=subdiv, amp=0.06, floor=c.y - cap_r * 0.08)
    # Spots are tiny bright specks: the mid version leaves them out.
    if spots and not m.mid:
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
    m.min_radius = 0.0
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
    tint = rng.choice([(0.25, 0.48, 0.11), (0.22, 0.44, 0.1), (0.28, 0.5, 0.12)])
    tuft_cards(m, rng, card_material(f"grass_{tint[0]}", "grass_tuft", tint), rng.uniform(0.32, 0.45), 0.55, 3)


def tall_grass(m, rng):
    tuft_cards(m, rng, card_material("tall_grass", "grass_tall", (0.42, 0.58, 0.2)), rng.uniform(0.8, 1.0), 0.85, 4)


def beach_grass(m, rng):
    tuft_cards(m, rng, card_material("beach_grass", "grass_thin", (0.62, 0.62, 0.36)), rng.uniform(0.6, 0.8), 0.8, 4, lean=0.3)


def alpine_grass(m, rng):
    tuft_cards(m, rng, card_material("alpine_grass", "grass_tuft", (0.42, 0.5, 0.22)), rng.uniform(0.2, 0.28), 0.55, 3)


# ---------------------------------------------------------------------------------------------
# Rocks and debris


def rock(m, rng, top_mat=None, top=0.5, size=(0.9, 1.3)):
    a, b = rock_mats()
    rx = rng.uniform(*size)
    ry = rx * rng.uniform(0.55, 0.8)
    rz = rx * rng.uniform(0.7, 1.0)
    blob(m, (0, ry * 0.5, 0), (rx, ry, rz), a, rng, subdiv=4, amp=0.26, freq=1.2, smooth=True, top_mat=top_mat, top=top, mat2=b, mix=0.45)
    m.root_shade = (ry * 0.6, 0.35)


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
        blob(m, c + UP * r * 0.35, (r, r * 0.6, r * 0.85), rng.choice((a, b)), rng, subdiv=2, amp=0.15)


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

# ---------------------------------------------------------------------------------------------
# Desert, savanna, jungle, tundra and arctic


def saguaro(m, rng):
    skin = material("cactus", (0.10, 0.22, 0.08), 0.6)
    height = rng.uniform(4.0, 5.5)
    lean = horizontal(rng.uniform(0, TAU)) * 0.15
    trunk = [Vector((0, -0.3, 0)) + lean * (i / 6) ** 2 + UP * ((height + 0.3) * i / 6) for i in range(7)]
    radius = rng.uniform(0.3, 0.36)
    tube(m, trunk, [radius] * 6 + [radius * 0.95], 16, skin, cap_end=False, ribs=0.1)
    blob(m, trunk[-1], (radius * 0.95, radius * 0.75, radius * 0.95), skin, rng, subdiv=2, amp=0.02)
    for k in range(rng.randint(1, 3)):
        a = rng.uniform(0, TAU)
        out = horizontal(a)
        y = rng.uniform(1.4, height * 0.6)
        reach = rng.uniform(0.5, 0.8)
        rise = rng.uniform(1.0, height - y - 0.4)
        r = radius * 0.75
        base = Vector((0, y, 0))
        elbow = base + out * reach
        arm = [base, base + out * reach * 0.6, elbow + UP * 0.25, elbow + UP * rise * 0.5, elbow + UP * rise]
        tube(m, arm, [r] * 5, 14, skin, cap_end=False, ribs=0.1)
        blob(m, arm[-1], (r * 0.95, r * 0.75, r * 0.95), skin, rng, subdiv=2, amp=0.02)


def barrel_cactus(m, rng):
    skin = material("cactus", (0.10, 0.22, 0.08), 0.6)
    bloom = material("cactus_flower", (0.7, 0.45, 0.04), 0.5)
    for k in range(rng.randint(1, 3)):
        c = horizontal(rng.uniform(0, TAU)) * (0.0 if k == 0 else rng.uniform(0.35, 0.55))
        h = rng.uniform(0.35, 0.7) * (1.0 if k == 0 else 0.7)
        r = h * rng.uniform(0.55, 0.7)
        n = 8
        pts = [c + UP * (-0.1 + (h + 0.1) * i / n) for i in range(n + 1)]
        radii = [max(0.02, r * math.sin(math.pi * (0.25 + 0.75 * i / n)) ** 0.6) for i in range(n + 1)]
        tube(m, pts, radii, 18, skin, ribs=0.12)
        for j in range(5):
            blob(m, pts[-1] + horizontal(TAU * j / 5) * radii[-1] * 0.6, (0.04, 0.03, 0.04), bloom, rng, subdiv=1, amp=0.0)


def prickly_pear(m, rng):
    skin = material("prickly_pear", (0.16, 0.3, 0.1), 0.6)
    fruit = material("prickly_fruit", (0.55, 0.05, 0.12), 0.5)

    def paddle(base, d, size, depth):
        c = base + d * size * 0.9
        side = UP.cross(d)
        side = side.normalized() if side.length > 1e-3 else Vector((1, 0, 0))
        # A flat oval: wide across the paddle, long along it, thin through it.
        basis = (side, d, side.cross(d).normalized())
        blob(m, c, (size * 0.62, size * 0.95, size * 0.18), skin, rng, subdiv=2, amp=0.05, basis=basis)
        if depth == 0:
            for _ in range(rng.randint(0, 2)):
                blob(m, c + d * size * 0.9 + random_unit(rng) * 0.04, (0.05, 0.06, 0.05), fruit, rng, subdiv=1, amp=0.0)
            return
        for _ in range(rng.randint(1, 2)):
            nd = (d + horizontal(rng.uniform(0, TAU)) * 0.7).normalized()
            paddle(c + d * size * 0.7, nd, size * 0.85, depth - 1)

    for _ in range(rng.randint(2, 3)):
        d = (UP + horizontal(rng.uniform(0, TAU)) * 0.5).normalized()
        paddle(Vector((0, -0.05, 0)), d, rng.uniform(0.22, 0.28), 2)


def tumbleweed(m, rng):
    twig = bark("tumbleweed", (0.36, 0.28, 0.17), 0.95)
    r = rng.uniform(0.35, 0.5)
    c = Vector((0, r * 0.85, 0))
    # Twigs wound round the surface of a ball: each runs roughly tangent to it.
    for _ in range(75):
        d = random_unit(rng)
        start = c + d * r * rng.uniform(0.55, 0.95)
        tangent = d.cross(random_unit(rng)).normalized()
        branch(m, start, tangent, r * rng.uniform(0.5, 0.8), 0.01, twig, rng, segments=3, sides=3, curl=-0.3)


def sandstone_rock(m, rng):
    light = material("sandstone_a", (0.48, 0.31, 0.18), 0.85)
    dark = material("sandstone_b", (0.30, 0.16, 0.08), 0.9)
    rx = rng.uniform(0.9, 1.5)
    ry = rx * rng.uniform(0.6, 0.9)
    blob(m, (0, ry * 0.6, 0), (rx, ry, rx * rng.uniform(0.7, 1.0)), light, rng, subdiv=4, amp=0.18, freq=0.9, mat2=dark, bands=0.22)
    m.root_shade = (ry * 0.6, 0.35)


def acacia(m, rng):
    bark_mat = bark("acacia", (0.16, 0.11, 0.07))
    leaves = card_material("acacia", "leaves_round", (0.34, 0.44, 0.14))
    lean = horizontal(rng.uniform(0, TAU)) * rng.uniform(0.3, 0.7)
    split = rng.uniform(2.0, 2.8)
    trunk = [Vector((0, -0.4, 0)), Vector((0, split * 0.5, 0)) + lean * 0.3, Vector((0, split, 0)) + lean]
    tube(m, trunk, [0.32, 0.25, 0.22], 8, bark_mat, cap_end=False)
    twigs = []
    for k in range(rng.randint(3, 4)):
        d = (horizontal(TAU * k / 3.5 + rng.uniform(-0.4, 0.4)) * 0.8 + UP * rng.uniform(0.9, 1.3)).normalized()
        end = branch(m, trunk[-1], d, rng.uniform(2.0, 2.8), 0.14, bark_mat, rng, curl=-0.2)
        # Then out flat: the umbrella.
        for _ in range(rng.randint(2, 3)):
            flat = (horizontal(rng.uniform(0, TAU)) + UP * rng.uniform(-0.05, 0.15)).normalized()
            grow(m, rng, end, flat, rng.uniform(1.4, 2.2), 0.07, 1, bark_mat, twigs, spread=0.5, rise=-0.05, children=(2, 3))
    center, radii = bounds_of([p for p, _ in twigs], 0.6)
    radii = (radii[0], max(radii[1], 0.6), radii[2])
    # Cards lie nearly flat, so the crown reads as a thin, wide canopy.
    for point, direction in twigs:
        for _ in range(5):
            out = Vector((point.x - center.x, 0.0, point.z - center.z))
            out = out.normalized() if out.length > 1e-3 else horizontal(rng.uniform(0, TAU))
            up = (out + random_unit(rng) * 0.5 + UP * 0.1).normalized()
            across = up.cross(UP)
            if across.length < 1e-3:
                continue
            across.normalize()
            tilt = rng.uniform(-0.35, 0.35)
            across = across * math.cos(tilt) + up.cross(across) * math.sin(tilt)
            size = rng.uniform(1.1, 1.5)
            corners = quad(point - up * size * 0.15 + UP * rng.uniform(-0.15, 0.15), up, across, size, size)
            m.card(corners, leaves, [crown_normal(c, center, radii) * 0.5 + UP * 0.5 for c in corners])
    m.occluders.append((center, radii))


def jungle_tree(m, rng):
    bark_mat = bark("jungle", (0.2, 0.16, 0.11))
    leaves = card_material(f"jungle_{rng.randint(0, 1)}", "leaves_round", rng.choice([(0.14, 0.36, 0.08), (0.18, 0.4, 0.1)]))
    height = rng.uniform(14.0, 17.0)
    top = height * 0.68
    pts = [Vector((0, -0.4, 0)), Vector((0, 2.0, 0)), Vector((0.2, top * 0.6, 0.1)), Vector((0, top, 0))]
    tube(m, pts, [0.55, 0.45, 0.38, 0.32], 9, bark_mat, cap_end=False)
    # Buttress roots: flattened fins flaring from the base.
    for k in range(rng.randint(4, 6)):
        d = horizontal(TAU * k / 5 + rng.uniform(-0.3, 0.3))
        fin = [Vector((0, 2.2, 0)) + d * 0.25, d * 0.8 + UP * 0.9, d * 1.6 - UP * 0.15]
        tube(m, fin, [0.3, 0.22, 0.08], 5, bark_mat)
    twigs = []
    for k in range(rng.randint(4, 6)):
        d = (horizontal(TAU * k / 5 + rng.uniform(-0.4, 0.4)) + UP * rng.uniform(0.5, 0.9)).normalized()
        grow(m, rng, pts[-1] - UP * rng.uniform(0, 1.5), d, rng.uniform(2.6, 3.4), 0.2, 2, bark_mat, twigs, rise=0.2, children=(3, 4))
    center, radii = bounds_of([p for p, _ in twigs], 1.0)
    hang_cards(m, rng, twigs, leaves, 2.2, 5, center, radii)
    m.occluders.append((center, radii))


def banana_plant(m, rng):
    stem = material("banana_stem", (0.18, 0.28, 0.1), 0.6)
    leaf_mat = card_material("banana", "leaf_large", (0.3, 0.55, 0.14))
    h = rng.uniform(1.6, 2.4)
    tube(m, [Vector((0, -0.1, 0)), Vector((0, h * 0.5, 0)), Vector((0, h, 0))], [0.14, 0.12, 0.1], 8, stem)
    top = Vector((0, h, 0))
    for k in range(rng.randint(6, 8)):
        a = TAU * k / 7 + rng.uniform(-0.3, 0.3)
        d = horizontal(a)
        length = rng.uniform(1.6, 2.2)
        rise = rng.uniform(0.7, 1.1)
        steps = 6
        spine = [top + d * length * (i / steps) + UP * length * (rise * i / steps - 1.05 * (i / steps) ** 2) for i in range(steps + 1)]
        side = UP.cross(d).normalized()
        widths = [0.62 * length * 0.5] * (steps + 1)
        normals = [UP * 0.6 + d * 0.4 for _ in spine]
        strip_card(m, spine, widths, [(side + DOWN * 0.2).normalized()] * (steps + 1), leaf_mat, normals)
    m.root_shade = (h, 0.3)


def cotton_grass(m, rng):
    tuft_cards(m, rng, card_material("cotton_grass", "grass_cotton", (0.55, 0.62, 0.36)), rng.uniform(0.35, 0.45), 0.6, 3)


def savanna_grass(m, rng):
    tuft_cards(m, rng, card_material("savanna_grass", "grass_tall", (0.78, 0.66, 0.32)), rng.uniform(0.7, 0.95), 0.85, 4, lean=0.25)


def desert_grass(m, rng):
    tuft_cards(m, rng, card_material("desert_grass", "grass_thin", (0.72, 0.6, 0.36)), rng.uniform(0.3, 0.45), 0.55, 3, lean=0.35)


def arctic_shrub(m, rng):
    shrub(m, rng, card_material("arctic_shrub", "needles", (0.26, 0.3, 0.18)), stems=(5, 7), reach=(0.35, 0.55), card=0.45, per_twig=5, rise=0.15, spread=1.5)


def ice_chunk(m, rng):
    ice = material("ice", (0.52, 0.72, 0.9), 0.08)
    for k in range(rng.randint(1, 3)):
        c = horizontal(rng.uniform(0, TAU)) * (0.0 if k == 0 else rng.uniform(0.6, 1.0))
        r = rng.uniform(0.5, 1.0) * (1.0 if k == 0 else 0.6)
        blob(m, c + UP * r * 0.3, (r, r * rng.uniform(0.6, 1.0), r * 0.8), ice, rng, subdiv=2, amp=0.3, freq=0.9, smooth=False)
    m.root_shade = (0.5, 0.2)


def lichen_rock(m, rng):
    rock(m, rng, top_mat=material("lichen", (0.42, 0.4, 0.22), 0.9), top=0.45, size=(0.7, 1.2))



BUILDERS = {
    f.__name__: f
    for f in (
        oak, maple, birch, pine, snow_pine, palm, willow, dead_tree,
        bush, berry_bush, juniper, dry_shrub,
        fern, daisies, poppies, bluebells, alpine_flowers, red_mushroom, brown_mushrooms, cattails,
        grass, tall_grass, beach_grass, alpine_grass,
        boulder, mossy_rock, snow_rock, pebbles, log, stump, driftwood,
        saguaro, barrel_cactus, prickly_pear, tumbleweed, sandstone_rock, acacia, jungle_tree,
        banana_plant, cotton_grass, savanna_grass, desert_grass, arctic_shrub, ice_chunk, lichen_rock,
    )
}  # fmt: skip


# ---------------------------------------------------------------------------------------------
# Impostors: what distant trees are drawn as

IMPOSTOR_SIZE = 512
IMPOSTORS = os.path.join(TEXTURES, "impostors")


def render_impostor(obj, name, bounds):
    """Render three flat-lit orthographic views of `obj` into `impostors/<name>.png`, a 2x2 atlas
    (texture v up): from +z top-left, from +x top-right, from above bottom-left. Returns the views'
    centre and their size in metres (every view is a square of that side)."""
    import numpy as np

    lo, hi = Vector(bounds["min"]), Vector(bounds["max"])
    center = (lo + hi) / 2
    size = max(hi.x - lo.x, hi.y - lo.y, hi.z - lo.z) * 1.02
    scene = bpy.context.scene
    for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    half = IMPOSTOR_SIZE // 2
    scene.render.resolution_x = half
    scene.render.resolution_y = half
    scene.render.film_transparent = True
    scene.view_settings.view_transform = "Standard"
    if scene.world is None:
        scene.world = bpy.data.worlds.new("world")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    # Even white light from every side: the views hold albedo (and a little occlusion), and the game
    # lights them like any other foliage.
    bg.inputs["Color"].default_value = (1.0, 1.0, 1.0, 1.0)
    bg.inputs["Strength"].default_value = 1.0
    for o in list(scene.objects):
        if o.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(o)
    cam_data = bpy.data.cameras.new("impostor")
    cam_data.type = "ORTHO"
    cam_data.ortho_scale = size
    cam_data.clip_end = 1000.0
    cam = bpy.data.objects.new("impostor", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam
    out = np.zeros((IMPOSTOR_SIZE, IMPOSTOR_SIZE, 4), dtype=np.float32)
    # (eye direction, atlas row offset, column offset); Blender pixel rows run bottom-up.
    views = [((0.0, 0.0, 1.0), half, 0), ((1.0, 0.0, 0.0), half, half), ((0.0, 1.0, 0.0), 0, 0)]
    for k, (eye, row, col) in enumerate(views):
        pos = center + Vector(eye) * 100.0
        cam.location = g2b(pos)
        # From above, the image's up is game -z (Blender +y) and its right game +x.
        cam.rotation_euler = (g2b(center) - g2b(pos)).to_track_quat("-Z", "Y").to_euler()
        path = os.path.join(IMPOSTORS, f"_{name}_{k}.png")
        scene.render.filepath = path
        bpy.ops.render.render(write_still=True)
        img = bpy.data.images.load(path)
        px = np.empty(len(img.pixels), dtype=np.float32)
        img.pixels.foreach_get(px)
        out[row : row + half, col : col + half] = px.reshape(half, half, 4)
        bpy.data.images.remove(img)
        os.remove(path)
    # Fill transparent texels with the average opaque colour so filtering never pulls in black.
    covered = out[..., 3] > 0.5
    if covered.any():
        out[~covered, :3] = out[covered, :3].mean(axis=0)
    image = bpy.data.images.new(name, IMPOSTOR_SIZE, IMPOSTOR_SIZE, alpha=True)
    image.pixels.foreach_set(out.ravel())
    image.filepath_raw = os.path.join(IMPOSTORS, f"{name}.png")
    image.file_format = "PNG"
    image.save()
    bpy.data.images.remove(image)
    return center, size


def build_impostor(name, center, size):
    """Three cards showing the impostor views: crossed vertical cards facing +z and +x, and a
    horizontal one through the crown seen from above (without it, trees seen from high up are
    just an X)."""
    m = Mesh(name)
    mat = card_material(name, f"impostors/{name}")
    r = size / 2
    y0, y1 = center.y - r, center.y + r

    def normals(corners, facing):
        # Soft: mostly up and out from the trunk, so the flat cards light like a rounded crown.
        out = []
        for c in corners:
            side = Vector((c.x - center.x, 0.0, c.z - center.z))
            side = side.normalized() if side.length > 1e-3 else facing
            out.append(UP * 0.45 + side * 0.4 + facing * 0.15)
        return out

    front = [Vector((center.x - r, y0, 0)), Vector((center.x + r, y0, 0)), Vector((center.x + r, y1, 0)), Vector((center.x - r, y1, 0))]
    m.card(front, mat, normals(front, Vector((0, 0, 1))), uvs=((0.0, 0.5), (0.5, 0.5), (0.5, 1.0), (0.0, 1.0)))
    # Seen from +x, the image's right edge is towards -z.
    side = [Vector((0, y0, center.z + r)), Vector((0, y0, center.z - r)), Vector((0, y1, center.z - r)), Vector((0, y1, center.z + r))]
    m.card(side, mat, normals(side, Vector((1, 0, 0))), uvs=((0.5, 0.5), (1.0, 0.5), (1.0, 1.0), (0.5, 1.0)))
    # From above: image bottom is towards +z, right towards +x. At the crown's middle height.
    crown = center.y + r * 0.3
    top = [Vector((center.x - r, crown, center.z + r)), Vector((center.x + r, crown, center.z + r)),
           Vector((center.x + r, crown, center.z - r)), Vector((center.x - r, crown, center.z - r))]
    m.card(top, mat, [UP * 0.8 + (c - center).normalized() * 0.2 for c in top], uvs=((0.0, 0.0), (0.5, 0.0), (0.5, 0.5), (0.0, 0.5)))
    m.root_shade = (size * 0.3, 0.3)
    return m.finish()


def burnt(spec, out_dir, only):
    """The burnt stages of every tree, shrub, plant and grass (no previews or far/mid versions)."""
    global BURN, KEEP
    report_path = os.path.join(out_dir, "report_burnt.json")
    report = {}
    for model in spec["model"]:
        if model["kind"] not in ("tree", "shrub", "plant", "grass") or (only and model["id"] not in only):
            continue
        for stage in (1, 2, 3):
            name = f"{model['id']}_burnt{stage}"
            BURN = stage
            KEEP = BURN_KEEP if model["kind"] in ("tree", "shrub") else GRASS_KEEP
            # A small tuft can lose every blade by chance: try another fall until something's left.
            for attempt in range(8):
                bpy.ops.wm.read_factory_settings(use_empty=True)
                MATERIALS.clear()
                BURN_RNG.seed(f"{model['id']}/{stage}/{attempt}")
                m = Mesh(name)
                BUILDERS[model["builder"]](m, random.Random(model["seed"]))
                obj = m.finish()
                report[name] = measure(obj)
                if report[name]["triangles"] > 0:
                    break
            export(obj, os.path.join(out_dir, f"{name}.glb"))
            print(f"PART {name}: {report[name]['triangles']} triangles", flush=True)
    BURN, KEEP = 0, BURN_KEEP
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("PARTS DONE", flush=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    spec_path, out_dir = argv[0], argv[1]
    only = set(argv[2].split(",")) if len(argv) > 2 and argv[2] else None
    with open(spec_path, "rb") as f:
        spec = tomllib.load(f)
    if len(argv) > 2 and argv[2] == "burnt":
        burnt(spec, out_dir, set(argv[3].split(",")) if len(argv) > 3 and argv[3] else None)
        return
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
        if model["kind"] == "tree":
            os.makedirs(IMPOSTORS, exist_ok=True)
            far_name = f"{model['id']}_far"
            center, size = render_impostor(obj, far_name, report[model["id"]])
            far = build_impostor(far_name, center, size)
            export(far, os.path.join(out_dir, f"{far_name}.glb"))
        if model["kind"] in ("tree", "shrub", "grass", "plant"):
            # The mid-distance version: the same model from the same seed, without its twigs and
            # with fewer, larger cards (or coarser grass).
            global BUILDING_MID
            bpy.ops.wm.read_factory_settings(use_empty=True)
            MATERIALS.clear()
            mid = Mesh(f"{model['id']}_mid")
            mid.mid = True
            MID_TEXTURES_USED.clear()
            BUILDING_MID = True
            BUILDERS[model["builder"]](mid, random.Random(model["seed"]))
            BUILDING_MID = False
            mid_obj = mid.finish()
            mid_report = measure(mid_obj)
            mid_path = os.path.join(out_dir, f"{model['id']}_mid.glb")
            # Nothing to drop (cacti, the banana plant): no mid version.
            if mid_report["triangles"] < report[model["id"]]["triangles"] or MID_TEXTURES_USED:
                report[f"{model['id']}_mid"] = mid_report
                export(mid_obj, mid_path)
            elif os.path.exists(mid_path):
                os.remove(mid_path)
        print(f"PART {model['id']}: {report[model['id']]['triangles']} triangles", flush=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print("PARTS DONE", flush=True)


if __name__ == "__main__":
    main()
