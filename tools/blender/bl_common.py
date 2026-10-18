"""Blender helpers shared by Ascension's model generators (parts.py, nature.py).

Geometry is authored in game coordinates (x right, y up, z towards the viewer, metres);
`game_to_blender` converts to Blender's Z-up, and the glTF exporter converts back.
"""

import math

import bpy
from mathutils import Vector


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
    options = dict(
        filepath=path,
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_apply=True,
        export_normals=True,
        export_materials="EXPORT",
    )
    if obj.data.color_attributes:
        # Vertex colours carry baked occlusion; the default only exports them when a material uses them.
        options["export_vertex_color"] = "ACTIVE"
    bpy.ops.export_scene.gltf(**options)


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
