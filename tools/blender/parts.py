"""Generate Ascension's building parts in Blender.

Run headless (tools/build_parts.py does this for you):

    blender --background --factory-startup --python tools/blender/parts.py -- <spec.toml> <out dir>

For every tier in the spec and every part (foundation, wall, ceiling) it builds the model from
bevelled boxes and cylinders, exports `<out>/<tier>_<part>.glb`, renders `<out>/previews/<tier>_<part>.png`
and writes `<out>/report.json` (bounding boxes and triangle counts) for validation.

All geometry is authored in *game* coordinates (x right, y up, z towards the viewer, metres) and
converted to Blender's Z-up here, so sizes read the same as in building.toml. Every part stays
inside its spec box; detail is carved inwards, never added outwards.
"""

import json
import math
import os
import random
import sys
import tomllib

import bmesh
import bpy
from mathutils import Matrix, Vector

# ---------------------------------------------------------------------------------------------
# Scene helpers


def game_to_blender(x, y, z):
    # glTF export maps Blender (x, y, z) -> (x, z, -y), so game z is -Blender y.
    return Vector((x, -z, y))


MATERIALS = {}


def material(name, color, roughness, metallic=0.0):
    if name in MATERIALS:
        return MATERIALS[name]
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = (*color, 1.0)
    bsdf.inputs["Roughness"].default_value = roughness
    bsdf.inputs["Metallic"].default_value = metallic
    MATERIALS[name] = mat
    return mat


class Part:
    """Collects the pieces of one model and turns them into a single exported object."""

    def __init__(self, name):
        self.name = name
        self.objects = []

    def box(self, lo, hi, mat, bevel=0.0):
        """Axis-aligned box from game-space corner `lo` to `hi`, optionally chamfered."""
        size = [hi[i] - lo[i] for i in range(3)]
        if min(size) <= 0:
            raise ValueError(f"{self.name}: empty box {lo} {hi}")
        center = [(hi[i] + lo[i]) / 2 for i in range(3)]
        mesh = bpy.data.meshes.new(f"{self.name}_box")
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bm.to_mesh(mesh)
        bm.free()
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.collection.objects.link(obj)
        # Blender axes: x = game x, y = game depth (-z), z = game height (y).
        obj.scale = (size[0], size[2], size[1])
        obj.location = game_to_blender(*center)
        obj.data.materials.append(mat)
        bpy.context.view_layer.objects.active = obj
        obj.select_set(True)
        bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
        obj.select_set(False)
        if bevel > 0:
            width = min(bevel, min(size) * 0.45)
            mod = obj.modifiers.new("bevel", "BEVEL")
            mod.width = width
            mod.segments = 1
            mod.limit_method = "ANGLE"
        self.objects.append(obj)
        return obj

    def cylinder(self, center, radius, length, axis, mat, sides=8):
        """Cylinder along game axis 'x', 'y' or 'z', centred at `center`."""
        mesh = bpy.data.meshes.new(f"{self.name}_cyl")
        bm = bmesh.new()
        bmesh.ops.create_cone(bm, cap_ends=True, segments=sides, radius1=radius, radius2=radius, depth=length)
        bm.to_mesh(mesh)
        bm.free()
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.collection.objects.link(obj)
        # create_cone points along Blender z (= game y).
        if axis == "x":
            obj.rotation_euler = (0.0, math.pi / 2, 0.0)
        elif axis == "z":
            obj.rotation_euler = (math.pi / 2, 0.0, 0.0)
        obj.location = game_to_blender(*center)
        obj.data.materials.append(mat)
        self.objects.append(obj)
        return obj

    def finish(self):
        """Apply modifiers, join into one object, smooth shading by angle. Returns the object."""
        bpy.ops.object.select_all(action="DESELECT")
        for obj in self.objects:
            obj.select_set(True)
        bpy.context.view_layer.objects.active = self.objects[0]
        bpy.ops.object.convert(target="MESH")
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
        obj.name = self.name
        obj.data.name = self.name
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35))
        return obj


# ---------------------------------------------------------------------------------------------
# Tier styles. Each takes the part's size (game metres) and returns a finished Part.
#
# Bounds: foundation x,z in [-w/2, w/2], y in [0, h]; wall x in [-w/2, w/2], y in [0, h],
# z in [-t/2, t/2]; ceiling x,z in [-w/2, w/2], y in [-h, 0].


def spaced(lo, hi, count, gap):
    """Split [lo, hi] into `count` equal spans separated by `gap`."""
    span = (hi - lo - gap * (count - 1)) / count
    return [(lo + i * (span + gap), lo + i * (span + gap) + span) for i in range(count)]


def courses(lo, hi, height, gap):
    count = max(1, round((hi - lo) / height))
    return spaced(lo, hi, count, gap)


def staggered(lo, hi, length, gap, offset):
    """Blocks of about `length` along [lo, hi], the first shifted by `offset` (running bond)."""
    edges = [lo]
    x = lo + (length * 0.5 if offset else length)
    while x < hi - length * 0.3:
        edges.append(x)
        x += length
    edges.append(hi)
    return [(a + (gap / 2 if i > 0 else 0), b - (gap / 2 if i < len(edges) - 2 else 0)) for i, (a, b) in enumerate(zip(edges, edges[1:]))]


class Wood:
    def __init__(self):
        self.frame = material("wood_frame", (0.16, 0.09, 0.045), 0.8)
        self.plank = material("wood_plank", (0.34, 0.21, 0.10), 0.72)
        self.plank_alt = material("wood_plank_alt", (0.30, 0.18, 0.085), 0.75)

    def planks(self, part, axis_ranges, count, along, gap, rng):
        pass

    def foundation(self, part, w, h, d):
        hw, hd = w / 2, d / 2
        part.box((-hw + 0.03, 0.0, -hd + 0.03), (hw - 0.03, h - 0.06, hd - 0.03), self.frame)
        rng = random.Random(1)
        # Deck boards along x.
        for i, (z0, z1) in enumerate(spaced(-hd, hd, 6, 0.012)):
            mat = self.plank if i % 2 == 0 else self.plank_alt
            top = h - rng.uniform(0.0, 0.006)
            part.box((-hw, h - 0.06, z0), (hw, top, z1), mat, bevel=0.008)
        # Corner posts, flush with the outside.
        for sx in (-1, 1):
            for sz in (-1, 1):
                x0, x1 = sorted((sx * hw, sx * (hw - 0.12)))
                z0, z1 = sorted((sz * hd, sz * (hd - 0.12)))
                part.box((x0, 0.0, z0), (x1, h - 0.06, z1), self.frame, bevel=0.012)
        # Side cladding boards between the posts.
        for j, (y0, y1) in enumerate(spaced(0.0, h - 0.06, 3, 0.012)):
            mat = self.plank if j % 2 else self.plank_alt
            for s in (-1, 1):
                part.box((-hw + 0.12, y0, s * hd - (0.03 if s > 0 else 0)), (hw - 0.12, y1, s * hd + (0.03 if s < 0 else 0)), mat, bevel=0.006)
                part.box((s * hw - (0.03 if s > 0 else 0), y0, -hd + 0.12), (s * hw + (0.03 if s < 0 else 0), y1, hd - 0.12), mat, bevel=0.006)

    def wall(self, part, w, h, t):
        hw, ht = w / 2, t / 2
        post = 0.12
        part.box((-hw, 0.0, -ht), (-hw + post, h, ht), self.frame, bevel=0.012)
        part.box((hw - post, 0.0, -ht), (hw, h, ht), self.frame, bevel=0.012)
        part.box((-hw + post, 0.0, -ht), (hw - post, 0.12, ht), self.frame, bevel=0.012)
        part.box((-hw + post, h - 0.12, -ht), (hw - post, h, ht), self.frame, bevel=0.012)
        part.box((-hw + post, h / 2 - 0.04, -ht + 0.015), (hw - post, h / 2 + 0.04, ht - 0.015), self.frame, bevel=0.01)
        rng = random.Random(2)
        for i, (x0, x1) in enumerate(spaced(-hw + post, hw - post, 7, 0.01)):
            mat = self.plank if i % 2 == 0 else self.plank_alt
            inset = rng.uniform(0.035, 0.05)
            part.box((x0, 0.12, -ht + inset), (x1, h - 0.12, ht - inset), mat, bevel=0.006)

    def ceiling(self, part, w, h, d):
        hw, hd = w / 2, d / 2
        deck = 0.06
        rng = random.Random(3)
        for i, (z0, z1) in enumerate(spaced(-hd, hd, 6, 0.012)):
            mat = self.plank if i % 2 == 0 else self.plank_alt
            part.box((-hw, -deck, z0), (hw, -rng.uniform(0.0, 0.005), z1), mat, bevel=0.008)
        for x in (-hw + 0.06, -0.33, 0.33, hw - 0.06):
            part.box((x - 0.06, -h, -hd), (x + 0.06, -deck, hd), self.frame, bevel=0.01)


class Stone:
    def __init__(self):
        self.block = material("stone_block", (0.40, 0.38, 0.35), 0.88)
        self.block_alt = material("stone_block_alt", (0.34, 0.33, 0.31), 0.9)
        self.mortar = material("stone_mortar", (0.22, 0.21, 0.19), 0.95)

    def face_blocks(self, part, rng, ranges_y, length, axis, lo, hi, out0, out1):
        """Running-bond blocks on one face: `axis` is the face's horizontal axis ('x' or 'z')."""
        for j, (y0, y1) in enumerate(ranges_y):
            for k, (a0, a1) in enumerate(staggered(lo, hi, length, 0.016, j % 2)):
                mat = self.block if (j + k) % 3 else self.block_alt
                o0 = out0 + rng.uniform(0.0, 0.006) * (1 if out0 < out1 else -1)
                if axis == "x":
                    part.box((a0, y0, min(o0, out1)), (a1, y1, max(o0, out1)), mat, bevel=0.014)
                else:
                    part.box((min(o0, out1), y0, a0), (max(o0, out1), y1, a1), mat, bevel=0.014)

    def foundation(self, part, w, h, d):
        hw, hd = w / 2, d / 2
        rng = random.Random(4)
        part.box((-hw + 0.02, 0.0, -hd + 0.02), (hw - 0.02, h - 0.02, hd - 0.02), self.mortar)
        rows = courses(0.0, h - 0.06, 0.24, 0.016)
        self.face_blocks(part, rng, rows, 0.5, "x", -hw + 0.06, hw - 0.06, hd, hd - 0.06)
        self.face_blocks(part, rng, rows, 0.5, "x", -hw + 0.06, hw - 0.06, -hd, -hd + 0.06)
        self.face_blocks(part, rng, rows, 0.5, "z", -hd, hd, hw, hw - 0.06)
        self.face_blocks(part, rng, rows, 0.5, "z", -hd, hd, -hw, -hw + 0.06)
        for i, (x0, x1) in enumerate(spaced(-hw, hw, 4, 0.016)):
            for k, (z0, z1) in enumerate(spaced(-hd, hd, 4, 0.016)):
                mat = self.block if (i + k) % 2 else self.block_alt
                part.box((x0, h - 0.06, z0), (x1, h - rng.uniform(0.0, 0.005), z1), mat, bevel=0.012)

    def wall(self, part, w, h, t):
        hw, ht = w / 2, t / 2
        rng = random.Random(5)
        part.box((-hw + 0.01, 0.0, -ht + 0.02), (hw - 0.01, h, ht - 0.02), self.mortar)
        for j, (y0, y1) in enumerate(courses(0.0, h, 0.25, 0.016)):
            for k, (x0, x1) in enumerate(staggered(-hw, hw, 0.5, 0.016, j % 2)):
                mat = self.block if (j + k) % 3 else self.block_alt
                inset = rng.uniform(0.0, 0.008)
                part.box((x0, y0, -ht + inset), (x1, y1, ht - inset), mat, bevel=0.014)

    def ceiling(self, part, w, h, d):
        hw, hd = w / 2, d / 2
        rng = random.Random(6)
        part.box((-hw + 0.02, -h, -hd + 0.02), (hw - 0.02, -0.07, hd - 0.02), self.mortar)
        for i, (x0, x1) in enumerate(spaced(-hw, hw, 4, 0.016)):
            for k, (z0, z1) in enumerate(spaced(-hd, hd, 4, 0.016)):
                mat = self.block if (i + k) % 2 else self.block_alt
                part.box((x0, -0.07, z0), (x1, -rng.uniform(0.0, 0.005), z1), mat, bevel=0.012)


class Concrete:
    def __init__(self):
        self.panel = material("concrete_panel", (0.52, 0.51, 0.48), 0.92)
        self.panel_alt = material("concrete_panel_alt", (0.48, 0.475, 0.45), 0.93)
        self.seam = material("concrete_seam", (0.30, 0.30, 0.29), 0.95)
        self.tie = material("concrete_tie", (0.10, 0.10, 0.10), 0.55)

    def ties(self, part, points, axis, outer, inward):
        """Tie-rod plugs on a face: short dark cylinders flush with the face, recessed into it."""
        for (a, b) in points:
            depth = 0.012
            c = outer - inward * depth / 2
            if axis == "z":
                part.cylinder((a, b, c), 0.022, depth, "z", self.tie)
            else:
                part.cylinder((c, b, a), 0.022, depth, "x", self.tie)

    def foundation(self, part, w, h, d):
        hw, hd = w / 2, d / 2
        part.box((-hw + 0.02, 0.0, -hd + 0.02), (hw - 0.02, h - 0.02, hd - 0.02), self.seam)
        for i, (x0, x1) in enumerate(spaced(-hw, hw, 2, 0.02)):
            for s in (-1, 1):
                mat = self.panel if (i + s) % 2 else self.panel_alt
                z0, z1 = sorted((s * (hd - 0.025), s * (hd - 0.005)))
                part.box((x0, 0.0, z0), (x1, h - 0.025, z1), mat, bevel=0.01)
                x0b, x1b = sorted((s * (hw - 0.025), s * (hw - 0.005)))
                zz0, zz1 = (-hd + 0.03 + i * (hd - 0.02), -hd + 0.03 + i * (hd - 0.02) + (hd - 0.04))
                part.box((x0b, 0.0, zz0), (x1b, h - 0.025, zz1), mat, bevel=0.01)
        for i, (x0, x1) in enumerate(spaced(-hw, hw, 2, 0.02)):
            for k, (z0, z1) in enumerate(spaced(-hd, hd, 2, 0.02)):
                part.box((x0, h - 0.025, z0), (x1, h, z1), self.panel if (i + k) % 2 else self.panel_alt, bevel=0.012)
        pts = [(x, y) for x in (-0.75, -0.25, 0.25, 0.75) for y in (0.3, 0.7)]
        self.ties(part, pts, "z", hd - 0.005, 1)
        self.ties(part, [(-x, y) for x, y in pts], "z", -(hd - 0.005), -1)
        self.ties(part, pts, "x", hw - 0.005, 1)
        self.ties(part, pts, "x", -(hw - 0.005), -1)

    def wall(self, part, w, h, t):
        hw, ht = w / 2, t / 2
        part.box((-0.02, 0.0, -ht + 0.02), (0.02, h, ht - 0.02), self.seam)
        part.box((-hw, h / 2 - 0.01, -ht + 0.02), (hw, h / 2 + 0.01, ht - 0.02), self.seam)
        for i, (x0, x1) in enumerate(spaced(-hw, hw, 2, 0.02)):
            for j, (y0, y1) in enumerate(spaced(0.0, h, 2, 0.02)):
                mat = self.panel if (i + j) % 2 else self.panel_alt
                part.box((x0, y0, -ht + 0.005), (x1, y1, ht - 0.005), mat, bevel=0.012)
        # Plugs stand 5 mm proud of the panels, so they reach the wall's full thickness.
        pts = [(x, y) for x in (-0.75, -0.25, 0.25, 0.75) for y in (0.5, 1.5)]
        self.ties(part, pts, "z", ht, 1)
        self.ties(part, pts, "z", -ht, -1)

    def ceiling(self, part, w, h, d):
        hw, hd = w / 2, d / 2
        part.box((-hw + 0.02, -h, -hd + 0.02), (hw - 0.02, -0.02, hd - 0.02), self.seam)
        for i, (x0, x1) in enumerate(spaced(-hw, hw, 2, 0.02)):
            for k, (z0, z1) in enumerate(spaced(-hd, hd, 2, 0.02)):
                part.box((x0, -h, z0), (x1, 0.0, z1), self.panel if (i + k) % 2 else self.panel_alt, bevel=0.015)


class Steel:
    def __init__(self):
        self.plate = material("steel_plate", (0.36, 0.38, 0.41), 0.42, 0.85)
        self.frame = material("steel_frame", (0.22, 0.235, 0.26), 0.38, 0.9)
        self.rivet = material("steel_rivet", (0.46, 0.47, 0.49), 0.3, 0.95)

    def rivets(self, part, points, axis, outer, inward):
        for (a, b) in points:
            height = 0.012
            c = outer - inward * height / 2
            if axis == "z":
                part.cylinder((a, b, c), 0.014, height, "z", self.rivet, sides=6)
            elif axis == "x":
                part.cylinder((c, b, a), 0.014, height, "x", self.rivet, sides=6)
            else:
                part.cylinder((a, c, b), 0.014, height, "y", self.rivet, sides=6)

    def foundation(self, part, w, h, d):
        hw, hd = w / 2, d / 2
        f = 0.09
        # Plates recessed behind the frame.
        part.box((-hw + 0.03, 0.0, -hd + 0.03), (hw - 0.03, h - 0.03, hd - 0.03), self.plate)
        # Vertical corner posts and top/bottom rings.
        for sx in (-1, 1):
            for sz in (-1, 1):
                x0, x1 = sorted((sx * hw, sx * (hw - f)))
                z0, z1 = sorted((sz * hd, sz * (hd - f)))
                part.box((x0, 0.0, z0), (x1, h, z1), self.frame, bevel=0.008)
        for y0, y1 in ((0.0, f), (h - f, h)):
            for s in (-1, 1):
                z0, z1 = sorted((s * hd, s * (hd - f)))
                part.box((-hw + f, y0, z0), (hw - f, y1, z1), self.frame, bevel=0.008)
                x0, x1 = sorted((s * hw, s * (hw - f)))
                part.box((x0, y0, -hd + f), (x1, y1, hd - f), self.frame, bevel=0.008)
        # Ribs on each side.
        for x in (-0.33, 0.33):
            for s in (-1, 1):
                z0, z1 = sorted((s * hd, s * (hd - 0.05)))
                part.box((x - 0.025, f, z0), (x + 0.025, h - f, z1), self.frame, bevel=0.006)
                x0, x1 = sorted((s * hw, s * (hw - 0.05)))
                part.box((x0, f, x - 0.025), (x1, h - f, x + 0.025), self.frame, bevel=0.006)
        # Deck plate on top and rivets along its edges.
        part.box((-hw + f, h - 0.035, -hd + f), (hw - f, h - 0.012, hd - f), self.plate, bevel=0.004)
        edge = [(-hw + 0.045, z) for z in (-0.6, -0.2, 0.2, 0.6)] + [(hw - 0.045, z) for z in (-0.6, -0.2, 0.2, 0.6)]
        edge += [(x, -hd + 0.045) for x in (-0.6, -0.2, 0.2, 0.6)] + [(x, hd - 0.045) for x in (-0.6, -0.2, 0.2, 0.6)]
        self.rivets(part, edge, "y", h, 1)

    def wall(self, part, w, h, t):
        hw, ht = w / 2, t / 2
        f = 0.09
        part.box((-hw + f, f, -ht + 0.03), (hw - f, h - f, ht - 0.03), self.plate)
        part.box((-hw, 0.0, -ht), (-hw + f, h, ht), self.frame, bevel=0.008)
        part.box((hw - f, 0.0, -ht), (hw, h, ht), self.frame, bevel=0.008)
        part.box((-hw + f, 0.0, -ht), (hw - f, f, ht), self.frame, bevel=0.008)
        part.box((-hw + f, h - f, -ht), (hw - f, h, ht), self.frame, bevel=0.008)
        for y in (h / 3, 2 * h / 3):
            part.box((-hw + f, y - 0.03, -ht + 0.012), (hw - f, y + 0.03, ht - 0.012), self.frame, bevel=0.006)
        pts = [(-hw + 0.045, y) for y in (0.3, 0.7, 1.1, 1.5)] + [(hw - 0.045, y) for y in (0.3, 0.7, 1.1, 1.5)]
        pts += [(x, 0.045) for x in (-0.6, -0.2, 0.2, 0.6)] + [(x, h - 0.045) for x in (-0.6, -0.2, 0.2, 0.6)]
        self.rivets(part, pts, "z", ht, 1)
        self.rivets(part, pts, "z", -ht, -1)

    def ceiling(self, part, w, h, d):
        hw, hd = w / 2, d / 2
        f = 0.09
        part.box((-hw, -0.04, -hd), (hw, -0.012, hd), self.plate, bevel=0.004)
        for s in (-1, 1):
            z0, z1 = sorted((s * hd, s * (hd - f)))
            part.box((-hw, -h, z0), (hw, -0.04, z1), self.frame, bevel=0.008)
            x0, x1 = sorted((s * hw, s * (hw - f)))
            part.box((x0, -h, -hd + f), (x1, -0.04, hd - f), self.frame, bevel=0.008)
        part.box((-0.04, -h, -hd + f), (0.04, -0.04, hd - f), self.frame, bevel=0.006)
        grid = [(x, z) for x in (-0.6, -0.2, 0.2, 0.6) for z in (-0.6, -0.2, 0.2, 0.6)]
        self.rivets(part, grid, "y", 0.0, 1)


STYLES = {"wood": Wood, "stone": Stone, "concrete": Concrete, "steel": Steel}

# ---------------------------------------------------------------------------------------------
# Export, measure, render


def measure(obj):
    """Game-space bounding box and triangle count of an object."""
    mesh = obj.data
    lo = [math.inf] * 3
    hi = [-math.inf] * 3
    for v in mesh.vertices:
        p = obj.matrix_world @ v.co
        g = (p.x, p.z, -p.y)
        for i in range(3):
            lo[i] = min(lo[i], g[i])
            hi[i] = max(hi[i], g[i])
    mesh.calc_loop_triangles()
    return {"min": lo, "max": hi, "triangles": len(mesh.loop_triangles)}


def export(obj, path):
    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.export_scene.gltf(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_apply=True,
        export_normals=True,
        export_materials="EXPORT",
    )


def render_preview(obj, path):
    scene = bpy.context.scene
    for engine in ("BLENDER_EEVEE", "BLENDER_EEVEE_NEXT", "CYCLES"):
        try:
            scene.render.engine = engine
            break
        except TypeError:
            continue
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = 32
    scene.render.resolution_x = 640
    scene.render.resolution_y = 480
    scene.render.film_transparent = False
    if scene.world is None:
        scene.world = bpy.data.worlds.new("world")
    scene.world.use_nodes = True
    bg = scene.world.node_tree.nodes.get("Background")
    bg.inputs["Color"].default_value = (0.55, 0.65, 0.8, 1.0)
    bg.inputs["Strength"].default_value = 0.7

    for o in list(scene.objects):
        if o.type in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(o)
    sun_data = bpy.data.lights.new("sun", "SUN")
    sun_data.energy = 4.0
    sun = bpy.data.objects.new("sun", sun_data)
    sun.rotation_euler = (math.radians(50), 0.0, math.radians(35))
    scene.collection.objects.link(sun)

    # Frame the object's bounding box from a three-quarter view.
    corners = [obj.matrix_world @ Vector(c) for c in obj.bound_box]
    center = sum(corners, Vector()) / 8
    radius = max((c - center).length for c in corners)
    cam_data = bpy.data.cameras.new("cam")
    cam_data.lens = 50
    cam = bpy.data.objects.new("cam", cam_data)
    direction = Vector((1.0, -1.25, 0.8)).normalized()
    cam.location = center + direction * radius * 3.4
    look = (center - cam.location).to_track_quat("-Z", "Y")
    cam.rotation_euler = look.to_euler()
    scene.collection.objects.link(cam)
    scene.camera = cam
    scene.render.filepath = path
    bpy.ops.render.render(write_still=True)


def main():
    argv = sys.argv[sys.argv.index("--") + 1 :]
    spec_path, out_dir = argv[0], argv[1]
    with open(spec_path, "rb") as f:
        spec = tomllib.load(f)
    os.makedirs(os.path.join(out_dir, "previews"), exist_ok=True)
    report = {}
    for tier in spec["tier"]:
        style = STYLES[tier["id"]]
        for part_id, part_spec in spec["parts"].items():
            bpy.ops.wm.read_factory_settings(use_empty=True)
            MATERIALS.clear()
            w, h, d = part_spec["size"]
            name = f"{tier['id']}_{part_id}"
            part = Part(name)
            builder = style()
            getattr(builder, part_id)(part, w, h, d)
            obj = part.finish()
            report[name] = measure(obj)
            export(obj, os.path.join(out_dir, f"{name}.glb"))
            render_preview(obj, os.path.join(out_dir, "previews", f"{name}.png"))
            print(f"PART {name}: {report[name]['triangles']} triangles", flush=True)
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("PARTS DONE", flush=True)


main()
