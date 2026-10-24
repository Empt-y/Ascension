"""Generate Ascension's handheld tools in Blender.

Run headless (tools/build_tools.py does this for you):

    blender --background --factory-startup --python tools/blender/tools.py -- <out dir>

Builds each tool from bevelled boxes, cylinders and spheres, exports `<out>/<name>.glb`, renders
`<out>/previews/<name>.png` and writes `<out>/report.json` (bounds, triangles, and where each
tool's business end is).

Geometry is in *game* coordinates (x right, y up, z forward, metres). Held tools have their grip
at the origin with the hand closed round the y axis and the tool pointing along +z, so the game
can put the origin in the hand. The beacon stands on the ground at the origin.
"""

import json
import math
import os
import sys

import bmesh
import bpy
from mathutils import Euler, Vector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from bl_common import MATERIALS, export, game_to_blender, material, measure, render_preview  # noqa: E402

# Materials: a dark gunmetal body, NMS-ish orange panels, rubber grips, brushed metal, glass, and
# glowing bits (screens, coils, lights).
GUNMETAL = ((0.08, 0.085, 0.095), 0.35, 0.9)
ORANGE = ((0.85, 0.32, 0.06), 0.45, 0.1)
RUBBER = ((0.025, 0.025, 0.028), 0.85, 0.0)
STEEL = ((0.55, 0.56, 0.58), 0.25, 1.0)
WHITE = ((0.78, 0.79, 0.8), 0.4, 0.05)


def mat(name, spec, emission=None):
    color, rough, metal = spec
    return material(name, color, rough, metal, emission)


def glow(name, color, strength=6.0):
    return material(name, tuple(c * 0.2 for c in color), 0.3, 0.0, (color, strength))


class Tool:
    """Collects one tool's pieces, then joins them into a single object."""

    def __init__(self, name):
        self.name = name
        self.objects = []

    def _place(self, mesh, mat_, center, rot):
        obj = bpy.data.objects.new(mesh.name, mesh)
        bpy.context.collection.objects.link(obj)
        obj.location = game_to_blender(*center)
        # Game rotations (degrees about game x, y, z) as Blender's (x, -z, y).
        rx, ry, rz = (math.radians(a) for a in rot)
        obj.rotation_euler = Euler((rx, -rz, ry), "XYZ")
        obj.data.materials.append(mat_)
        self.objects.append(obj)
        return obj

    def box(self, center, size, mat_, bevel=0.0, rot=(0, 0, 0)):
        """A box of game `size` (x, y, z) centred at `center`, rounded off by `bevel`."""
        mesh = bpy.data.meshes.new(f"{self.name}_box")
        bm = bmesh.new()
        bmesh.ops.create_cube(bm, size=1.0)
        bmesh.ops.scale(bm, vec=Vector((size[0], size[2], size[1])), verts=bm.verts)
        bm.to_mesh(mesh)
        bm.free()
        obj = self._place(mesh, mat_, center, rot)
        if bevel > 0:
            m = obj.modifiers.new("bevel", "BEVEL")
            m.width = min(bevel, min(size) * 0.45)
            m.segments = 2
            m.limit_method = "ANGLE"
        return obj

    def cyl(self, center, radius, length, mat_, axis="z", sides=16, radius2=None, rot=(0, 0, 0)):
        """A cylinder (or a cone, with `radius2` at its +axis end) along game `axis`."""
        mesh = bpy.data.meshes.new(f"{self.name}_cyl")
        bm = bmesh.new()
        r2 = radius if radius2 is None else radius2
        bmesh.ops.create_cone(bm, cap_ends=True, segments=sides, radius1=radius, radius2=r2, depth=length)
        bm.to_mesh(mesh)
        bm.free()
        # create_cone runs along Blender z = game y; turn it onto the axis wanted.
        base = {"y": (0, 0, 0), "z": (90, 0, 0), "x": (0, 0, -90)}[axis]
        return self._place(mesh, mat_, center, tuple(b + r for b, r in zip(base, rot)))

    def ring(self, center, radius, thickness, mat_, axis="z"):
        """A torus round game `axis`."""
        bpy.ops.mesh.primitive_torus_add(major_radius=radius, minor_radius=thickness, major_segments=24, minor_segments=6)
        obj = bpy.context.active_object
        obj.location = game_to_blender(*center)
        rot = {"y": (0, 0, 0), "z": (90, 0, 0), "x": (0, 0, -90)}[axis]
        rx, ry, rz = (math.radians(a) for a in rot)
        obj.rotation_euler = Euler((rx, -rz, ry), "XYZ")
        obj.data.materials.append(mat_)
        self.objects.append(obj)
        return obj

    def sphere(self, center, radius, mat_, scale=(1, 1, 1)):
        mesh = bpy.data.meshes.new(f"{self.name}_sph")
        bm = bmesh.new()
        bmesh.ops.create_uvsphere(bm, u_segments=16, v_segments=10, radius=radius)
        bmesh.ops.scale(bm, vec=Vector((scale[0], scale[2], scale[1])), verts=bm.verts)
        bm.to_mesh(mesh)
        bm.free()
        return self._place(mesh, mat_, center, (0, 0, 0))

    def finish(self):
        bpy.ops.object.select_all(action="DESELECT")
        for o in self.objects:
            o.select_set(True)
        bpy.context.view_layer.objects.active = self.objects[0]
        bpy.ops.object.convert(target="MESH")
        bpy.ops.object.join()
        obj = bpy.context.view_layer.objects.active
        obj.name = self.name
        obj.data.name = self.name
        bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35))
        return obj


def scanner():
    """A handheld survey scanner: a tablet held up before the eyes on a short handle, its glowing
    screen towards the user, a sensor bar with an emitter along its far edge."""
    t = Tool("scanner")
    body, grip, metal = mat("scanner_body", WHITE), mat("scanner_grip", RUBBER), mat("scanner_metal", GUNMETAL)
    screen = glow("scanner_screen", (0.15, 0.85, 1.0), 4.0)
    accent = glow("scanner_accent", (1.0, 0.55, 0.1), 5.0)
    tilt = (-20, 0, 0)
    # The tablet: a frame, a bezel and the screen facing back towards the holder.
    t.box((0, 0.1, 0.02), (0.15, 0.11, 0.022), body, bevel=0.01, rot=tilt)
    t.box((0, 0.1, 0.006), (0.135, 0.095, 0.004), metal, bevel=0.003, rot=tilt)
    t.box((0, 0.1, 0.003), (0.122, 0.082, 0.004), screen, bevel=0.002, rot=tilt)
    # Rubber bumpers on the corners.
    for x in (-0.074, 0.074):
        for y in (0.05, 0.15):
            t.box((x, y, 0.02), (0.016, 0.02, 0.028), grip, bevel=0.005, rot=tilt)
    # The sensor bar on top, an emitter and two status lights.
    t.box((0, 0.163, 0.04), (0.1, 0.022, 0.03), metal, bevel=0.006, rot=tilt)
    t.cyl((0, 0.167, 0.058), 0.012, 0.012, screen, axis="z", rot=tilt)
    for x in (-0.035, 0.035):
        t.box((x, 0.168, 0.056), (0.012, 0.006, 0.006), accent, rot=tilt)
    # The handle under it, and its collar.
    t.box((0, 0.0, 0.015), (0.034, 0.1, 0.034), grip, bevel=0.01, rot=(-10, 0, 0))
    t.cyl((0, 0.047, 0.02), 0.024, 0.016, metal, axis="y")
    return t.finish(), {"tip": [0.0, 0.17, 0.065]}


def multitool():
    """The multitool: pistol grip, a chunky receiver with orange panels, a top canister, and a
    barrel ringed with glowing coils."""
    t = Tool("multitool")
    body, paint, grip, steel = mat("mt_body", GUNMETAL), mat("mt_paint", ORANGE), mat("mt_grip", RUBBER), mat("mt_steel", STEEL)
    coil = glow("mt_coil", (0.3, 0.9, 1.0), 5.0)
    # Grip, raked back, and the trigger guard.
    t.box((0, -0.06, -0.01), (0.035, 0.13, 0.045), grip, bevel=0.01, rot=(18, 0, 0))
    t.box((0, -0.035, 0.05), (0.012, 0.06, 0.008), body, rot=(0, 0, 0))
    # Receiver: the body, its painted side panels and the stock.
    t.box((0, 0.025, 0.1), (0.06, 0.07, 0.3), body, bevel=0.012)
    for x in (-0.032, 0.032):
        t.box((x, 0.03, 0.09), (0.006, 0.05, 0.2), paint, bevel=0.004)
    t.box((0, 0.02, -0.1), (0.045, 0.06, 0.13), body, bevel=0.012, rot=(-8, 0, 0))
    t.box((0, 0.005, -0.17), (0.05, 0.08, 0.02), grip, bevel=0.006)
    # A canister along the top, held by two clamps.
    t.cyl((0, 0.075, 0.08), 0.022, 0.2, paint, axis="z")
    for z in (0.02, 0.14):
        t.cyl((0, 0.07, z), 0.026, 0.018, steel, axis="z")
    # Barrel: shroud, three glowing coils, and the emitter.
    t.cyl((0, 0.025, 0.31), 0.022, 0.16, steel, axis="z")
    for z in (0.27, 0.31, 0.35):
        t.ring((0, 0.025, z), 0.026, 0.006, coil, axis="z")
    t.cyl((0, 0.025, 0.4), 0.028, 0.03, body, axis="z", radius2=0.018)
    t.cyl((0, 0.025, 0.418), 0.012, 0.008, coil, axis="z")
    # A sight.
    t.box((0, 0.068, 0.22), (0.02, 0.025, 0.04), body, bevel=0.005)
    return t.finish(), {"muzzle": [0.0, 0.025, 0.425]}


def mode_light(name, color):
    """The multitool's mode light: a lens on its left side, lit in the mode's colour (drawn over
    the multitool, sharing its origin)."""
    t = Tool(name)
    t.box((-0.034, 0.045, 0.16), (0.006, 0.014, 0.03), glow(name, color, 8.0), bevel=0.002)
    t.box((0.034, 0.045, 0.16), (0.006, 0.014, 0.03), glow(name, color, 8.0), bevel=0.002)
    return t.finish(), {}


def manipulator():
    """The terrain manipulator: a squat emitter on a grip, a ring of fins round a glowing
    sphere."""
    t = Tool("manipulator")
    body, paint, grip = mat("tm_body", GUNMETAL), mat("tm_paint", ((0.1, 0.45, 0.35), 0.45, 0.2)), mat("tm_grip", RUBBER)
    core = glow("tm_core", (0.2, 1.0, 0.6), 6.0)
    t.box((0, -0.06, -0.01), (0.035, 0.13, 0.045), grip, bevel=0.01, rot=(15, 0, 0))
    t.box((0, 0.03, 0.06), (0.075, 0.08, 0.22), body, bevel=0.015)
    for x in (-0.04, 0.04):
        t.box((x, 0.03, 0.06), (0.006, 0.06, 0.16), paint, bevel=0.004)
    t.cyl((0, 0.03, 0.2), 0.045, 0.06, paint, axis="z", radius2=0.055)
    for i in range(6):
        a = i * 60.0
        r = math.radians(a)
        t.box((math.cos(r) * 0.055, 0.03 + math.sin(r) * 0.055, 0.25), (0.012, 0.012, 0.06), body, bevel=0.003, rot=(0, 0, a))
    t.sphere((0, 0.03, 0.255), 0.035, core)
    t.ring((0, 0.03, 0.23), 0.05, 0.006, core, axis="z")
    return t.finish(), {"tip": [0.0, 0.03, 0.29]}


def torch():
    """A torch: a long grip and a lamp head, its lens and a ring glowing warm."""
    t = Tool("torch")
    body, grip, steel = mat("torch_body", GUNMETAL), mat("torch_grip", RUBBER), mat("torch_steel", STEEL)
    lamp = glow("torch_lamp", (1.0, 0.75, 0.4), 10.0)
    t.cyl((0, 0.0, 0.0), 0.018, 0.12, grip, axis="y")
    t.cyl((0, 0.09, 0.0), 0.02, 0.06, body, axis="y")
    t.cyl((0, 0.14, 0.0), 0.024, 0.05, steel, axis="y", radius2=0.036)
    t.cyl((0, 0.168, 0.0), 0.034, 0.008, lamp, axis="y")
    t.ring((0, 0.12, 0.0), 0.023, 0.004, lamp, axis="y")
    t.cyl((0, -0.065, 0.0), 0.02, 0.012, steel, axis="y")
    return t.finish(), {"tip": [0.0, 0.175, 0.0]}


def beacon():
    """A waypoint beacon standing on the ground: three legs, a mast, and a glowing head."""
    t = Tool("beacon")
    body, paint, steel = mat("bc_body", GUNMETAL), mat("bc_paint", ORANGE), mat("bc_steel", STEEL)
    light = glow("bc_light", (0.2, 0.9, 1.0), 10.0)
    for i in range(3):
        a = i * 120.0
        r = math.radians(a)
        t.cyl((math.cos(r) * 0.08, 0.1, math.sin(r) * 0.08), 0.012, 0.23, steel, axis="y", sides=8, rot=(0, -a, 25))
    t.cyl((0, 0.18, 0), 0.05, 0.08, body, axis="y")
    t.cyl((0, 0.45, 0), 0.018, 0.5, steel, axis="y")
    t.box((0, 0.35, 0), (0.08, 0.12, 0.05), paint, bevel=0.01)
    t.cyl((0, 0.72, 0), 0.04, 0.05, body, axis="y")
    t.sphere((0, 0.78, 0), 0.045, light)
    t.ring((0, 0.35, 0), 0.06, 0.006, light, axis="y")
    return t.finish(), {"tip": [0.0, 0.8, 0.0]}


def injector():
    """A med injector: a pistol-grip syringe with a glowing green vial."""
    t = Tool("injector")
    body, grip, steel = mat("inj_body", WHITE), mat("inj_grip", RUBBER), mat("inj_steel", STEEL)
    vial = glow("inj_vial", (0.3, 1.0, 0.4), 5.0)
    t.box((0, -0.05, 0.0), (0.03, 0.1, 0.04), grip, bevel=0.01, rot=(10, 0, 0))
    t.box((0, 0.02, 0.05), (0.045, 0.05, 0.14), body, bevel=0.012)
    t.cyl((0, 0.06, 0.05), 0.016, 0.09, vial, axis="z")
    t.cyl((0, 0.02, 0.135), 0.012, 0.04, steel, axis="z", radius2=0.004)
    t.cyl((0, 0.02, 0.16), 0.0015, 0.02, steel, axis="z", sides=6)
    return t.finish(), {"tip": [0.0, 0.02, 0.17]}


TOOLS = {
    "scanner": scanner,
    "multitool": multitool,
    "multitool_light_laser": lambda: mode_light("multitool_light_laser", (0.3, 0.9, 1.0)),
    "multitool_light_rifle": lambda: mode_light("multitool_light_rifle", (1.0, 0.6, 0.15)),
    "multitool_light_flamer": lambda: mode_light("multitool_light_flamer", (1.0, 0.25, 0.05)),
    "manipulator": manipulator,
    "torch": torch,
    "beacon": beacon,
    "injector": injector,
}


def main():
    out = sys.argv[sys.argv.index("--") + 1]
    os.makedirs(os.path.join(out, "previews"), exist_ok=True)
    report = {}
    for name, build in TOOLS.items():
        bpy.ops.wm.read_factory_settings(use_empty=True)
        MATERIALS.clear()
        obj, points = build()
        export(obj, os.path.join(out, f"{name}.glb"))
        report[name] = {**measure(obj), **points}
        # The scanner is seen from the holder's side (its screen); the rest from the front.
        if name == "scanner":
            obj.rotation_euler = (0.0, 0.0, math.pi)
        render_preview(obj, os.path.join(out, "previews", f"{name}.png"))
    with open(os.path.join(out, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    print("PARTS DONE")


if __name__ == "__main__":
    main()
