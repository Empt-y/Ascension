"""Put Quaternius's CC0 creatures and characters into the game's assets.

    python tools/build_creatures.py [--src DIR] [--blender PATH]

Reads the downloaded packs (default C:\\Users\\theas\\Downloads\\quaternius: `animals/`,
`fish/`, and the Universal Base Characters and Universal Animation Library zips unpacked under
`x/`) and writes:

  assets/models/creatures/<kind>.glb    the animals (as they are) and the fish (exported from
                                        their .blend files by tools/blender/creatures.py)
  assets/models/player/body.glb, hair.glb, eyebrows.glb   the player's body and its attachments
  assets/models/player/animations_1.glb, animations_2.glb  the Universal Animation Libraries
  assets/models/{creatures,player}/CREDITS.md

Skinned models are repacked, never re-exported: a Blender round trip can change a skeleton's rest
orientations, and the animation libraries play on the body by bone name, so their rest poses must
stay exactly as authored. Repacking only gathers a .gltf's buffers and images into one .glb,
drops the normal and roughness maps (the game uses base colour only) and shrinks textures to at
most 1024 pixels.
"""

import argparse
import base64
import io
import json
import os
import shutil
import struct
import subprocess
import sys

from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CREATURES = os.path.join(ROOT, "assets", "models", "creatures")
PLAYER = os.path.join(ROOT, "assets", "models", "player")
DEFAULT_SRC = r"C:\Users\theas\Downloads\quaternius"
DEFAULT_BLENDER = r"C:\Users\theas\tools\blender-5.2.2\blender.exe"
MAX_TEXTURE = 1024

UBC = "x/Universal Base Characters[Standard]/Universal Base Characters[Standard]"
UAL1 = "x/Universal Animation Library[Standard]/Universal Animation Library[Standard]/Unreal-Godot/UAL1_Standard.glb"
UAL2 = "x/Universal Animation Library 2[Standard]/Universal Animation Library 2[Standard]/Unreal-Godot/UAL2_Standard.glb"

# Game kind -> file in animals/ (wild animals only: the dogs wait for taming).
ANIMALS = {
    "wolf": "Wolf.gltf",
    "fox": "Fox.gltf",
    "stag": "Stag.glb",
    "deer": "Deer.gltf",
    "horse": "Horse.gltf",
    "horse_white": "Horse_White.gltf",
    "bull": "Bull.gltf",
    "cow": "Cow.gltf",
    "donkey": "Donkey.glb",
    "alpaca": "Alpaca.gltf",
}
# Game kind -> .blend in fish/.
FISH = {
    "fish": "Fish1.blend",
    "fish_2": "Fish2.blend",
    "fish_3": "Fish3.blend",
    "dolphin": "Dolphin.blend",
    "manta": "Manta_ray.blend",
    "shark": "Shark.blend",
    "whale": "Whale.blend",
}
PLAYER_PARTS = {
    "body": f"{UBC}/Base Characters/Godot - UE/Superhero_Male_FullBody.gltf",
    "hair": f"{UBC}/Hairstyles/Rigged to Head Bone/glTF (Godot -Unreal)/Hair_SimpleParted.gltf",
    "eyebrows": f"{UBC}/Hairstyles/Rigged to Head Bone/glTF (Godot -Unreal)/Eyebrows_Regular.gltf",
}


def read_gltf(path):
    """A .gltf or .glb as (json, list of buffers' bytes)."""
    data = open(path, "rb").read()
    base = os.path.dirname(path)
    if data[:4] == b"glTF":
        n = struct.unpack("<I", data[12:16])[0]
        doc = json.loads(data[20 : 20 + n])
        rest = data[20 + n :]
        bin_chunk = rest[8 : 8 + struct.unpack("<I", rest[:4])[0]] if rest else b""
        buffers = []
        for b in doc.get("buffers", []):
            buffers.append(bin_chunk if "uri" not in b else load_uri(b["uri"], base))
        return doc, buffers, base
    doc = json.loads(data)
    return doc, [load_uri(b["uri"], base) for b in doc.get("buffers", [])], base


def load_uri(uri, base):
    if uri.startswith("data:"):
        return base64.b64decode(uri.split(",", 1)[1])
    return open(os.path.join(base, uri.replace("%20", " ")), "rb").read()


def pack(src, out, keep_images=True):
    """Repack `src` as one .glb at `out`: base colour textures only, shrunk; everything else as is."""
    doc, buffers, base = read_gltf(src)
    used = set()
    for m in doc.get("materials", []):
        for key in ("normalTexture", "occlusionTexture", "emissiveTexture"):
            m.pop(key, None)
        pbr = m.get("pbrMetallicRoughness", {})
        pbr.pop("metallicRoughnessTexture", None)
        if "baseColorTexture" in pbr:
            if keep_images:
                used.add(doc["textures"][pbr["baseColorTexture"]["index"]]["source"])
            else:
                pbr.pop("baseColorTexture")
    # Every buffer view's bytes, re-laid into one binary chunk.
    blob = bytearray()
    views = []
    for v in doc.get("bufferViews", []):
        data = buffers[v["buffer"]][v.get("byteOffset", 0) : v.get("byteOffset", 0) + v["byteLength"]]
        while len(blob) % 4:
            blob.append(0)
        views.append({**v, "buffer": 0, "byteOffset": len(blob)})
        blob += data
    images = []
    for i, img in enumerate(doc.get("images", [])):
        if i not in used:
            # Unused now: a 1x1 stand-in keeps every index valid.
            raw = one_pixel()
        elif "uri" in img:
            raw = shrink(load_uri(img["uri"], base))
        else:
            v = doc["bufferViews"][img["bufferView"]]
            b = buffers[v["buffer"]]
            raw = shrink(b[v.get("byteOffset", 0) : v.get("byteOffset", 0) + v["byteLength"]])
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(raw)})
        blob += raw
        images.append({"name": img.get("name", f"image{i}"), "mimeType": "image/png", "bufferView": len(views) - 1})
    while len(blob) % 4:
        blob.append(0)
    doc["bufferViews"] = views
    if images:
        doc["images"] = images
    doc["buffers"] = [{"byteLength": len(blob)}]
    write_glb(out, doc, bytes(blob))


def one_pixel():
    b = io.BytesIO()
    Image.new("RGBA", (1, 1), (255, 255, 255, 255)).save(b, "PNG")
    return b.getvalue()


def shrink(raw):
    im = Image.open(io.BytesIO(raw))
    if max(im.size) > MAX_TEXTURE:
        scale = MAX_TEXTURE / max(im.size)
        im = im.resize((round(im.width * scale), round(im.height * scale)), Image.LANCZOS)
    b = io.BytesIO()
    im.save(b, "PNG", optimize=True)
    return b.getvalue()


def write_glb(path, doc, blob):
    js = json.dumps(doc, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    with open(path, "wb") as f:
        f.write(struct.pack("<4sII", b"glTF", 2, total))
        f.write(struct.pack("<I4s", len(js), b"JSON") + js)
        f.write(struct.pack("<I4s", len(blob), b"BIN\x00") + blob)


def summary(path):
    doc, _, _ = read_gltf(path)
    anims = sorted({a.get("name", "").split("|")[-1] for a in doc.get("animations", [])})
    joints = len(doc["skins"][0]["joints"]) if doc.get("skins") else 0
    return f"{os.path.getsize(path) / 1e6:5.1f} MB, {joints} joints, {len(anims)} animations"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=DEFAULT_SRC)
    ap.add_argument("--blender", default=os.environ.get("BLENDER", DEFAULT_BLENDER))
    args = ap.parse_args()
    src = args.src
    os.makedirs(CREATURES, exist_ok=True)
    os.makedirs(PLAYER, exist_ok=True)
    made, missing = [], []
    for kind, f in ANIMALS.items():
        path = os.path.join(src, "animals", f)
        if not os.path.exists(path):
            missing.append(f)
            continue
        out = os.path.join(CREATURES, f"{kind}.glb")
        pack(path, out, keep_images=False)
        made.append((kind, out))
    fish = {k: os.path.join(src, "fish", f) for k, f in FISH.items() if os.path.exists(os.path.join(src, "fish", f))}
    missing += [f for k, f in FISH.items() if k not in fish]
    if fish:
        script = os.path.join(ROOT, "tools", "blender", "creatures.py")
        jobs = [f"{os.path.join(CREATURES, k + '.glb')}={p}" for k, p in fish.items()]
        cmd = [args.blender, "--background", "--factory-startup", "--python", script, "--", *jobs]
        result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
        if result.returncode != 0 or "FISH DONE" not in result.stdout:
            print(result.stdout[-3000:], result.stderr[-3000:])
            sys.exit("Blender failed on the fish")
        made += [(k, os.path.join(CREATURES, k + ".glb")) for k in fish]
    for name, rel in PLAYER_PARTS.items():
        out = os.path.join(PLAYER, f"{name}.glb")
        pack(os.path.join(src, rel), out)
        made.append((f"player {name}", out))
    for n, rel in ((1, UAL1), (2, UAL2)):
        out = os.path.join(PLAYER, f"animations_{n}.glb")
        shutil.copyfile(os.path.join(src, rel), out)
        made.append((f"animations {n}", out))
    for name, path in made:
        print(f"{name:16} {summary(path)}")
    if missing:
        print("not downloaded yet (Google Drive quota):", ", ".join(missing))
    credits = (
        "# Credits\n\nModels and animations by Quaternius (https://quaternius.com), CC0 1.0 (public\n"
        "domain: no credit needed, given anyway). Put here by tools/build_creatures.py.\n\n"
    )
    with open(os.path.join(CREATURES, "CREDITS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(credits + "- Ultimate Animated Animal Pack: https://quaternius.com/packs/ultimateanimatedanimals.html\n")
        f.write("- Animated Fish Pack: https://quaternius.com/packs/animatedfish.html\n")
        f.write("- Stag and donkey via Poly Pizza (Quaternius, CC0): https://poly.pizza/m/tQdzbZ1Cmw, https://poly.pizza/m/qmX6nhnvp7\n")
    with open(os.path.join(PLAYER, "CREDITS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(credits + "- Universal Base Characters (Standard): https://quaternius.com/packs/universalbasecharacters.html\n")
        f.write("- Universal Animation Library 1 and 2 (Standard): https://quaternius.com/packs/universalanimationlibrary.html,\n")
        f.write("  https://quaternius.com/packs/universalanimationlibrary2.html\n")


if __name__ == "__main__":
    main()
