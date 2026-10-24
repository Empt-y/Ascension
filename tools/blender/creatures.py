"""Export Quaternius's fish from their .blend files to glTF (tools/build_creatures.py runs this).

    blender --background --factory-startup --python tools/blender/creatures.py -- OUT.glb=IN.blend ...

Each .blend's objects are appended into an empty scene and exported with every action (the
fish's Swim). The fish only exist as .blend and FBX, and the game reads glTF.
"""

import sys

import bpy

for job in sys.argv[sys.argv.index("--") + 1 :]:
    out, src = job.split("=", 1)
    bpy.ops.wm.read_factory_settings(use_empty=True)
    with bpy.data.libraries.load(src, link=False) as (data_from, data_to):
        data_to.objects = data_from.objects
        data_to.actions = data_from.actions
    for o in data_to.objects:
        if o is not None and o.type in ("MESH", "ARMATURE", "EMPTY"):
            bpy.context.scene.collection.objects.link(o)
    bpy.ops.export_scene.gltf(
        filepath=out,
        export_format="GLB",
        export_yup=True,
        export_apply=False,
        export_animations=True,
        export_animation_mode="ACTIONS",
        export_skins=True,
        export_materials="EXPORT",
    )
    print("EXPORTED", out, flush=True)
print("FISH DONE", flush=True)
