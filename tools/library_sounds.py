"""Put free library recordings into assets/sounds_library under the game's sound names, to compare
with the generated ones (tools/sounds.py). In the game, F8 switches between the two sets; a sound
the libraries don't have stays generated.

    python tools/library_sounds.py <downloads dir>

The downloads dir holds what was fetched and unpacked (see the sources below): `kenney/impact.zip`,
`jcsounds/extracted/Nature Ambient Pack Vol 1 - mp3/...`, the OpenGameArt files in `oga/` and
Mixkit's in `mixkit/` (by id). Writes
assets/sounds_library/CREDITS.md with each sound's source and licence.
"""

import os
import shutil
import sys
import zipfile

import numpy as np
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "sounds_library")
JC = "jcsounds/extracted/Nature Ambient Pack Vol 1 - mp3"

KENNEY = ("Kenney, Impact Sounds", "CC0 1.0", "https://kenney.nl/assets/impact-sounds")
JCSOUNDS = ("JC Sounds, Nature Ambient Pack Vol 1", "CC-BY 4.0 (credit: JC Sounds)", "https://opengameart.org/node/180349")
QUBODUP = ("qubodup, Synthesized explosion", "CC0 1.0", "https://opengameart.org/content/synthesized-explosion")
BARADARI = ("Michel Baradari, Rumble/explosion", "CC-BY 3.0 (credit: Michel Baradari)", "https://opengameart.org/content/rumbleexplosion")
FANTOZZI = ("Fantozzi, Footsteps (Grass/Sand & Stone)", "CC0 1.0", "https://opengameart.org/content/fantozzis-footsteps-grasssand-stone")
CONGUS = (
    "congusbongus, Footsteps on different surfaces (water: from EminYILDIRIM and swuing on freesound.org)",
    "CC-BY 3.0 (credit: congusbongus, EminYILDIRIM, swuing)",
    "https://opengameart.org/content/footsteps-on-different-surfaces",
)
RUBBERDUCK = ("rubberduck, 40 CC0 water / splash / slime SFX", "CC0 1.0", "https://opengameart.org/content/40-cc0-water-splash-slime-sfx")
# Mixkit Sound Effects Free License: free in games, no credit needed, but the files mustn't be
# handed out on their own or "with source files" (fine inside the game; if this repository is ever
# published, fetch these instead of committing them).
MIXKIT = ("Mixkit, volcano sound effects", "Mixkit Sound Effects Free License (no credit needed)", "https://mixkit.co/free-sound-effects/volcano/")
BRET = ("bretbernhoft, Rumbling Sound Effects", "CC0 1.0", "https://opengameart.org/content/rumbling-sound-effects")
BMACZERO = ("BMacZero, Bubble Sound Effects", "CC0 1.0", "https://opengameart.org/content/bubble-sound-effects")
DKLON = ("dklon, Big Explosion", "CC-BY 3.0 (credit: dklon)", "https://opengameart.org/content/big-explosion")
KENNEY_RPG = ("Kenney, RPG Audio", "CC0 1.0", "https://kenney.nl/assets/rpg-audio")
TINYWORLDS = (
    "TinyWorlds, Different steps on wood, stone, leaves, gravel and mud",
    "CC0 1.0",
    "https://opengameart.org/content/different-steps-on-wood-stone-leaves-gravel-and-mud",
)
KDD = "oga/x_[kdd]DifferentSteps_0"
# Other footsteps on soft ground to audition in the game (Shift+F8 cycles them, in place of
# step_grass): steps/<set>/step_grass_<n>.
STEP_SETS = {
    "congusbongus": (CONGUS, [f"oga/x_footsteps_0/footsteps/grass/{n}.ogg" for n in range(9)]),
    "tinyworlds": (TINYWORLDS, [f"{KDD}/leaves01.ogg", f"{KDD}/leaves02.ogg", f"{KDD}/mud02.ogg"]),
}
# ...and Kenney's RPG footsteps, from their zip.
RPG_STEPS = [f"Audio/footstep0{n}.ogg" for n in range(10)]

# Game name -> (source, list of files); one-shots get numbered variations, loops keep one file.
KENNEY_SHOTS = {
    "step_grass": "footstep_grass",
    "step_snow": "footstep_snow",
    "step_hard": "footstep_concrete",
    "dig_hard": "impactMining",
    "dig_soft": "impactSoft_heavy",
    "place": "impactSoft_medium",
    "hurt": "impactPunch_heavy",
    "land": "impactGeneric_light",
}
LOOPS = {
    "wind": (JCSOUNDS, f"{JC}/Weather/AMB_Nature Pack Vol 1_Weather_Mountain Peak.mp3"),
    "rain": (JCSOUNDS, f"{JC}/Weather/AMB_Nature Pack Vol 1_Weather_Light Rain.mp3"),
    "river": (JCSOUNDS, f"{JC}/Water Enviroments/AMB_Nature Pack Vol 1_Water Enviroments_Mountain Stream.mp3"),
    "lake": (JCSOUNDS, f"{JC}/Water Enviroments/AMB_Nature Pack Vol 1_Water Enviroments_Peaceful Lake .mp3"),
    "fire": (JCSOUNDS, f"{JC}/Fire & Elemental/AMB_Nature Pack Vol 1_ Fire & Elemental_Large Bonfire.mp3"),
    "underwater": (BMACZERO, "oga/bubbles-loop1-amp.wav"),
}
# Loops cut from a long recording (seconds in, seconds long), the ends crossfaded so it loops.
CUTS = {
    "rumble": (BRET, "oga/rumbling17.wav", 20.0, 30.0),
    # "Volcano eruption with lava flow" and "Bubbling volcano lava".
    "roar": (MIXKIT, "mixkit/2443.wav", 10.0, 40.0),
    "lava": (MIXKIT, "mixkit/2440.wav", 5.0, 30.0),
}
# Numbered variations from loose files.
FILES = {
    "step_sand": (FANTOZZI, [f"oga/fantozzi/Fantozzi-footsteps/ogg/Fantozzi-Sand{s}{n}.ogg" for s in "LR" for n in (1, 2, 3)]),
    "step_water": (CONGUS, [f"oga/x_footsteps_0/footsteps/water/{n}.ogg" for n in range(5)]),
    "splash": (RUBBERDUCK, [f"oga/x_water-splash-slime-sfx/splash_{n:02}.ogg" for n in range(1, 9)]),
    "blast": (DKLON, ["oga/tnt.mp3", "oga/DeathFlash.flac"]),
    # Lava popping and hissing now and then (bubble, hiss, big bubble burst).
    "lava_pop": (MIXKIT, ["mixkit/2445.wav", "mixkit/2447.wav", "mixkit/2448.wav"]),
}
SINGLES = {
    "bomb_land_0": (QUBODUP, "oga/synthetic_explosion_1.flac"),
    "boom_0": (BARADARI, "oga/rumble.flac"),
    # "Volcano lava and ash burst".
    "bomb_land_1": (MIXKIT, "mixkit/2449.wav"),
}


def main():
    src = sys.argv[1]
    if os.path.isdir(OUT):
        shutil.rmtree(OUT)
    os.makedirs(OUT)
    credits = {}
    with zipfile.ZipFile(os.path.join(src, "kenney", "impact.zip")) as z:
        for name, prefix in KENNEY_SHOTS.items():
            files = sorted(f for f in z.namelist() if f.split("/")[-1].startswith(prefix + "_") and f.endswith(".ogg"))
            for n, f in enumerate(files):
                with z.open(f) as data, open(os.path.join(OUT, f"{name}_{n}.ogg"), "wb") as out:
                    shutil.copyfileobj(data, out)
            credits.setdefault(KENNEY, []).append(f"{name} ({len(files)}): {prefix}_*")
    for name, (source, path) in LOOPS.items():
        ext = os.path.splitext(path)[1]
        shutil.copyfile(os.path.join(src, path), os.path.join(OUT, name + ext))
        credits.setdefault(source, []).append(f"{name}: {os.path.basename(path)}")
    for name, (source, files) in FILES.items():
        for n, path in enumerate(files):
            shutil.copyfile(os.path.join(src, path), os.path.join(OUT, f"{name}_{n}{os.path.splitext(path)[1]}"))
        credits.setdefault(source, []).append(f"{name} ({len(files)}): " + ", ".join(os.path.basename(f) for f in files))
    for name, (source, path, start, seconds) in CUTS.items():
        rate, x = wavfile.read(os.path.join(src, path))
        # (Its level doesn't matter: the cut is normalised below.)
        x = x.astype(np.float64)
        a, n, f = int(start * rate), int(seconds * rate), int(1.5 * rate)
        piece = x[a : a + n + f]
        w = np.linspace(0, np.pi / 2, f)[:, None] if piece.ndim > 1 else np.linspace(0, np.pi / 2, f)
        out = piece[:n].copy()
        out[:f] = piece[:f] * np.sin(w) + piece[n : n + f] * np.cos(w)
        out /= max(np.max(np.abs(out)), 1e-9) / 0.9
        wavfile.write(os.path.join(OUT, f"{name}.wav"), rate, (out * 32767).astype(np.int16))
        credits.setdefault(source, []).append(f"{name}: {seconds:.0f} s of {os.path.basename(path)} from {start:.0f} s, looped")
    for name, (source, path) in SINGLES.items():
        ext = os.path.splitext(path)[1]
        shutil.copyfile(os.path.join(src, path), os.path.join(OUT, name + ext))
        credits.setdefault(source, []).append(f"{name}: {os.path.basename(path)}")
    for set_name, (source, files) in STEP_SETS.items():
        os.makedirs(os.path.join(OUT, "steps", set_name))
        for n, path in enumerate(files):
            shutil.copyfile(os.path.join(src, path), os.path.join(OUT, "steps", set_name, f"step_grass_{n}.ogg"))
        credits.setdefault(source, []).append(f"steps/{set_name} ({len(files)}): " + ", ".join(os.path.basename(f) for f in files))
    os.makedirs(os.path.join(OUT, "steps", "kenney_rpg"))
    with zipfile.ZipFile(os.path.join(src, "kenney", "rpg.zip")) as z:
        for n, f in enumerate(RPG_STEPS):
            with z.open(f) as data, open(os.path.join(OUT, "steps", "kenney_rpg", f"step_grass_{n}.ogg"), "wb") as out:
                shutil.copyfileobj(data, out)
    credits.setdefault(KENNEY_RPG, []).append(f"steps/kenney_rpg ({len(RPG_STEPS)}): footstep00-09")
    with open(os.path.join(OUT, "CREDITS.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write("# Library sounds\n\nRecordings from free libraries, under the game's sound names (made by\n")
        f.write("tools/library_sounds.py). Everything else in the game is generated by tools/sounds.py.\n\n")
        for (title, licence, url), uses in credits.items():
            f.write(f"## {title}\n\nLicence: {licence}. Source: {url}\n\n")
            f.writelines(f"- {u}\n" for u in uses)
            f.write("\n")
    files = [os.path.join(d, f) for d, _, fs in os.walk(OUT) for f in fs]
    print(f"{len(files)} files, {sum(map(os.path.getsize, files)) / 1e6:.1f} MB in {OUT}")


if __name__ == "__main__":
    main()
