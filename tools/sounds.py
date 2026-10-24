"""Generate Ascension's sound effects: every sound is made here from filtered noise, tones and
envelopes (no recordings), the way tools/blender makes the models.

    python tools/sounds.py [--only NAME ...]

Writes 16-bit WAV files into assets/sounds: one-shots as `<name>_<n>.wav` (variations, picked at
random in the game), loops as `<name>.wav` (seamless: the end is crossfaded into the start). Each
sound is deterministic from its name, so rebuilding changes nothing unless its recipe does. Prints
each file's length, peak and loudness, and writes assets/sounds/report.json.
"""

import argparse
import hashlib
import json
import os
import sys

import numpy as np
from scipy import signal
from scipy.io import wavfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "sounds")
RATE = 44100


# ---------------------------------------------------------------------------------------------
# Building blocks


def rng_for(name):
    return np.random.default_rng(int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "little"))


def t_of(seconds):
    return np.arange(int(seconds * RATE)) / RATE


def white(rng, seconds):
    return rng.standard_normal(int(seconds * RATE))


def pink(rng, seconds):
    """1/f noise (equal energy per octave): the base of wind, water and roars."""
    n = int(seconds * RATE)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / RATE)
    f[0] = f[1]
    out = np.fft.irfft(spectrum / np.sqrt(f), n)
    return out / np.std(out)


def brown(rng, seconds):
    """1/f^2 noise: deep rumbles."""
    n = int(seconds * RATE)
    spectrum = np.fft.rfft(rng.standard_normal(n))
    f = np.fft.rfftfreq(n, 1 / RATE)
    f[0] = f[1]
    out = np.fft.irfft(spectrum / f, n)
    return out / np.std(out)


def band(x, lo=None, hi=None, order=4):
    """Low-pass (`hi`), high-pass (`lo`) or band-pass (both), in Hz."""
    if lo and hi:
        sos = signal.butter(order, [lo, hi], "bandpass", fs=RATE, output="sos")
    elif hi:
        sos = signal.butter(order, hi, "lowpass", fs=RATE, output="sos")
    else:
        sos = signal.butter(order, lo, "highpass", fs=RATE, output="sos")
    return signal.sosfilt(sos, x)


def sweep_band(x, cutoff, order=2):
    """Low-pass with a cutoff (Hz) that changes per sample (block-wise, 256 samples)."""
    out = np.zeros_like(x)
    zi = None
    for start in range(0, len(x), 256):
        c = float(np.clip(cutoff[min(start, len(cutoff) - 1)], 40.0, RATE * 0.45))
        sos = signal.butter(order, c, "lowpass", fs=RATE, output="sos")
        if zi is None:
            zi = signal.sosfilt_zi(sos) * 0.0
        out[start : start + 256], zi = signal.sosfilt(sos, x[start : start + 256], zi=zi)
    return out


def env(seconds, attack, decay, curve=1.0):
    """An attack then an exponential-ish decay: the shape of a hit."""
    t = t_of(seconds)
    a = np.clip(t / max(attack, 1e-4), 0.0, 1.0)
    d = np.exp(-np.maximum(t - attack, 0.0) / decay) ** curve
    return a * d


def slow(rng, seconds, rate, depth, smooth=4):
    """A slowly wandering control signal around 1 (gusts, swells): smoothed random steps at
    about `rate` per second."""
    n = int(seconds * RATE)
    points = max(int(seconds * rate) + 3, 4)
    knots = 1.0 + depth * rng.uniform(-1, 1, points)
    for _ in range(smooth):
        knots = np.convolve(knots, [0.25, 0.5, 0.25], mode="same")
    return np.interp(np.linspace(0, points - 1, n), np.arange(points), knots)


def chirp(seconds, f0, f1, decay):
    """A tone gliding from f0 to f1 Hz as it dies away (bubbles, blorps)."""
    t = t_of(seconds)
    f = f0 + (f1 - f0) * (1 - np.exp(-t / (seconds * 0.4)))
    phase = 2 * np.pi * np.cumsum(f) / RATE
    return np.sin(phase) * np.exp(-t / decay)


def scatter(rng, seconds, rate, grain, amp=(0.3, 1.0)):
    """Short `grain()` sounds dropped at random (Poisson, `rate` per second)."""
    out = np.zeros(int(seconds * RATE))
    count = rng.poisson(rate * seconds)
    for _ in range(count):
        g = grain()
        start = rng.integers(0, max(len(out) - 1, 1))
        end = min(start + len(g), len(out))
        out[start:end] += g[: end - start] * rng.uniform(*amp)
    return out


def click(rng, seconds, lo, hi):
    return band(white(rng, seconds), lo, hi, 2) * env(seconds, 0.0005, seconds / 4)


def loop(make, seconds, fade=1.0):
    """A seamless loop: make `seconds + fade`, then crossfade the extra into the start."""
    x = make(seconds + fade)
    n, f = int(seconds * RATE), int(fade * RATE)
    if x.ndim == 1:
        x = x[:, None]
    head, tail = x[:f], x[n : n + f]
    w = np.linspace(0, np.pi / 2, f)[:, None]
    out = x[:n].copy()
    out[:f] = head * np.sin(w) + tail * np.cos(w)
    return out.squeeze()


def stereo(make):
    """Two decorrelated channels of the same recipe (wide ambience)."""
    return lambda seconds: np.stack([make(seconds, 0), make(seconds, 1)], axis=1)


def fade_out(x, seconds=0.01):
    n = min(int(seconds * RATE), len(x))
    x = x.copy()
    x[len(x) - n :] *= np.linspace(1, 0, n) if x.ndim == 1 else np.linspace(1, 0, n)[:, None]
    return x


def to_peak(x, db):
    return x / (np.max(np.abs(x)) + 1e-9) * 10 ** (db / 20)


def to_rms(x, db, ceiling=-1.0):
    x = x / (np.sqrt(np.mean(x**2)) + 1e-9) * 10 ** (db / 20)
    peak = np.max(np.abs(x))
    limit = 10 ** (ceiling / 20)
    return x * (limit / peak) if peak > limit else x


# ---------------------------------------------------------------------------------------------
# One-shots: each recipe makes one variation from its own random stream.


def dig_soft(rng):
    """A spade into soil, sand or snow: a muffled crunch with grit in it."""
    s = 0.32
    body = band(pink(rng, s), 120, 1800) * env(s, 0.004, 0.07)
    grit = scatter(rng, s, 90, lambda: click(rng, 0.006, 1500, 6000)) * env(s, 0.01, 0.1)
    thud = band(white(rng, s), None, 220) * env(s, 0.002, 0.05) * 2.5
    return to_peak(body + 0.5 * grit + thud, -3)


def dig_hard(rng):
    """A pick into rock: a sharp chip, a knock, and a few stones trickling."""
    s = 0.55
    chip = click(rng, s, 1800, 7000) * 3.0
    knock = chirp(s, rng.uniform(170, 230), 120, 0.05) * 0.8
    trickle = scatter(rng, s, 30, lambda: click(rng, 0.012, 900, 5000)) * env(s, 0.06, 0.18)
    return to_peak(chip * env(s, 0.0003, 0.012) + knock + 0.45 * trickle, -3)


def place(rng):
    """Packing a lump of material down: a soft, low thump."""
    s = 0.3
    thump = band(white(rng, s), None, 350) * env(s, 0.003, 0.06)
    tone = chirp(s, rng.uniform(90, 120), 55, 0.06) * 0.6
    pat = band(pink(rng, s), 300, 2500) * env(s, 0.002, 0.03) * 0.3
    return to_peak(thump + tone + pat, -4)


def step(rng, lo, hi, decay, grit_rate, grit_band, thud=0.8, seconds=0.22):
    """A footstep: heel then toe (two bursts a few hundredths apart), with grit for crunch."""
    s = seconds
    out = np.zeros(int(s * RATE))
    for offset, gain in ((0.0, 1.0), (rng.uniform(0.03, 0.06), 0.6)):
        n = int(offset * RATE)
        burst = band(pink(rng, s), lo, hi) * env(s, 0.003, decay)
        low = band(white(rng, s), None, 200) * env(s, 0.002, 0.03) * thud
        out[n:] += (burst + low)[: len(out) - n] * gain
    if grit_rate:
        out += 0.5 * scatter(rng, s, grit_rate, lambda: click(rng, 0.005, *grit_band)) * env(s, 0.005, decay * 1.5)
    return to_peak(fade_out(out), -6)


def step_grass(rng):
    return step(rng, 700, 4500, 0.035, 160, (2000, 7000), thud=0.5)


def step_hard(rng):
    return step(rng, 1200, 6000, 0.012, 40, (2500, 8000), thud=1.4, seconds=0.18)


def step_sand(rng):
    return step(rng, 400, 2600, 0.05, 220, (1500, 5000), thud=0.4, seconds=0.26)


def step_snow(rng):
    return step(rng, 600, 3000, 0.06, 420, (1200, 3500), thud=0.3, seconds=0.3)


def step_gravel(rng):
    return step(rng, 900, 5000, 0.03, 300, (1500, 6000), thud=0.7, seconds=0.25)


def steam(rng):
    """Lava meeting water: a sharp hiss of steam, dying away."""
    s = 1.6
    hiss = band(white(rng, s), 2500, 11000, 2) * env(s, 0.01, 0.35) * slow(rng, s, 6, 0.3)
    spit = scatter(rng, s, 40, lambda: click(rng, 0.01, 1500, 6000)) * env(s, 0.01, 0.3)
    return to_peak(fade_out(hiss + 0.4 * spit, 0.1), -3)


def step_water(rng):
    s = 0.4
    slosh = band(pink(rng, s), 300, 2500) * env(s, 0.01, 0.1)
    bubbles = scatter(rng, s, 25, lambda: chirp(0.05, rng.uniform(500, 900), rng.uniform(1200, 2000), 0.015)) * env(s, 0.02, 0.15)
    return to_peak(fade_out(slosh + 0.5 * bubbles), -6)


def thump(rng):
    """A soft, low thump: the body's weight landing, layered under every footstep and landing."""
    s = 0.25
    tone = chirp(s, rng.uniform(85, 105), rng.uniform(50, 60), 0.05)
    body = band(white(rng, s), None, 180) * env(s, 0.006, 0.04) * 1.5
    return to_peak(fade_out(tone * env(s, 0.006, 0.08) + body), -3)


def land(rng):
    """Landing from a jump or a fall: a heavy body thump."""
    s = 0.4
    thump = band(white(rng, s), None, 260) * env(s, 0.002, 0.07) * 2.0
    scuff = band(pink(rng, s), 500, 3000) * env(s, 0.004, 0.05) * 0.4
    return to_peak(thump + scuff, -3)


def hurt(rng):
    """Taking a hit: a dull blow."""
    s = 0.35
    blow = band(white(rng, s), None, 500) * env(s, 0.001, 0.05) * 2.0
    tone = chirp(s, 160, 70, 0.07)
    return to_peak(blow + tone, -3)


def splash(rng):
    """Falling into water: a burst of spray, then bubbles rising."""
    s = 1.0
    burst = band(pink(rng, s), 250, 5000) * env(s, 0.005, 0.12)
    bubbles = scatter(rng, s, 40, lambda: chirp(0.08, rng.uniform(300, 600), rng.uniform(900, 1800), 0.025)) * env(s, 0.08, 0.3)
    return to_peak(fade_out(burst + 0.6 * bubbles, 0.05), -3)


def bomb_land(rng):
    """A lava bomb hitting the ground: a deep crump and rocks spraying."""
    s = 1.6
    crump = band(brown(rng, s), None, 260) * env(s, 0.004, 0.25)
    crack = click(rng, s, 800, 6000) * env(s, 0.0005, 0.02) * 2.0
    debris = scatter(rng, s, 60, lambda: click(rng, 0.015, 900, 5000)) * env(s, 0.05, 0.4)
    return to_peak(fade_out(crump + crack + 0.4 * debris, 0.1), -2)


def boom(rng):
    """A volcano's eruption bursting out: a deep boom rolling away into rumble."""
    s = 5.0
    hit = band(brown(rng, s), None, 180) * env(s, 0.01, 0.5)
    roll = band(brown(rng, s), 20, 120) * env(s, 0.3, 1.6) * slow(rng, s, 3, 0.4)
    crackle = scatter(rng, s, 30, lambda: click(rng, 0.02, 300, 3000)) * env(s, 0.02, 0.8)
    return to_peak(fade_out(hit + 0.8 * roll + 0.3 * crackle, 0.4), -1)


def blast(rng):
    """A lateral blast's shockwave arriving: a sharp crack, a boom that thumps the chest, and the
    mountain coming apart for seconds after."""
    s = 10.0
    crack = band(white(rng, s), 200, 9000, 2) * env(s, 0.0005, 0.03) * 3.0
    hit = band(brown(rng, s), None, 120) * env(s, 0.005, 0.8) * 1.5
    roll = band(brown(rng, s), 18, 90) * env(s, 0.5, 3.5) * slow(rng, s, 2, 0.5)
    debris = scatter(rng, s, 25, lambda: band(brown(rng, 0.3), None, 400) * env(0.3, 0.01, 0.08)) * env(s, 0.5, 2.5)
    return to_peak(fade_out(crack + hit + roll + 0.5 * debris, 1.0), -0.5)


# ---------------------------------------------------------------------------------------------
# Loops


def wind(seconds, ch):
    rng = rng_for(f"wind{ch}")
    gust = slow(rng, seconds, 0.35, 0.55)
    base = sweep_band(pink(rng, seconds), 250 + 900 * gust**2) * gust
    # A faint whistle that comes and goes with the gusts.
    whistle = band(white(rng, seconds), 650, 760, 2) * np.clip(gust - 1.1, 0, None) * 6
    return base + whistle


def rain(seconds, ch):
    rng = rng_for(f"rain{ch}")
    hiss = band(white(rng, seconds), 1500, 12000, 2) * 0.35
    drops = scatter(rng, seconds, 900, lambda: click(rng, 0.004, 2000, 9000), (0.1, 1.0))
    patter = scatter(rng, seconds, 60, lambda: chirp(0.02, rng.uniform(1500, 3000), 900, 0.006), (0.2, 0.6))
    return hiss + drops + 0.5 * patter


def river(seconds):
    rng = rng_for("river")
    rush = band(pink(rng, seconds), 150, 2500) * slow(rng, seconds, 1.5, 0.2)
    burble = scatter(rng, seconds, 70, lambda: chirp(0.05, rng.uniform(350, 800), rng.uniform(900, 2200), 0.015), (0.2, 0.8))
    return rush + 0.6 * burble


def lake(seconds):
    rng = rng_for("lake")
    swell = np.clip(slow(rng, seconds, 0.4, 0.8, smooth=2), 0.2, None)
    lap = band(pink(rng, seconds), 200, 1400) * swell**2
    splashes = scatter(rng, seconds, 6, lambda: band(pink(rng, 0.2), 400, 3000) * env(0.2, 0.01, 0.05), (0.2, 0.6))
    return lap + 0.5 * splashes


def lava(seconds):
    rng = rng_for("lava")
    churn = band(brown(rng, seconds), None, 180) * slow(rng, seconds, 0.8, 0.3)
    blorps = scatter(rng, seconds, 5, lambda: chirp(0.35, rng.uniform(55, 80), rng.uniform(140, 220), 0.1), (0.4, 1.0))
    hiss = band(white(rng, seconds), 3000, 9000, 2) * 0.06 * slow(rng, seconds, 0.5, 0.6)
    pops = scatter(rng, seconds, 4, lambda: click(rng, 0.02, 600, 4000), (0.3, 1.0))
    return churn + 1.5 * blorps + hiss + 0.5 * pops


def fire(seconds):
    rng = rng_for("fire")
    roar = band(brown(rng, seconds), 40, 400) * slow(rng, seconds, 1.2, 0.3) * 0.6
    crackle = scatter(rng, seconds, 35, lambda: click(rng, 0.008, 1500, 9000), (0.1, 1.0))
    snaps = scatter(rng, seconds, 3, lambda: click(rng, 0.03, 600, 5000), (0.5, 1.0)) * 1.5
    hiss = band(white(rng, seconds), 4000, 10000, 2) * 0.05
    return roar + crackle + snaps + hiss


def rumble(seconds):
    rng = rng_for("rumble")
    return band(brown(rng, seconds), 18, 75) * slow(rng, seconds, 0.6, 0.5)


def roar(seconds):
    rng = rng_for("roar")
    jet = band(pink(rng, seconds), 120, 900) * slow(rng, seconds, 0.7, 0.35)
    deep = band(brown(rng, seconds), 20, 150) * slow(rng, seconds, 0.5, 0.4) * 1.5
    crackle = scatter(rng, seconds, 20, lambda: click(rng, 0.02, 400, 3000), (0.2, 1.0))
    return jet + deep + 0.4 * crackle


def underwater(seconds):
    rng = rng_for("underwater")
    murk = band(brown(rng, seconds), 30, 350) * slow(rng, seconds, 0.4, 0.3)
    bubbles = scatter(rng, seconds, 3, lambda: chirp(0.1, rng.uniform(250, 500), rng.uniform(700, 1100), 0.03), (0.2, 0.6))
    return murk + 0.4 * bubbles


# name: (recipe, variations) for one-shots; (recipe, seconds, loudness dB) for loops.
ONE_SHOTS = {
    "dig_soft": (dig_soft, 4),
    "dig_hard": (dig_hard, 4),
    "place": (place, 3),
    "step_grass": (step_grass, 5),
    "step_hard": (step_hard, 5),
    "step_sand": (step_sand, 5),
    "step_snow": (step_snow, 5),
    "step_water": (step_water, 4),
    "step_gravel": (step_gravel, 5),
    "steam": (steam, 3),
    "thump": (thump, 4),
    "land": (land, 2),
    "hurt": (hurt, 3),
    "splash": (splash, 3),
    "bomb_land": (bomb_land, 3),
    "boom": (boom, 2),
    "blast": (blast, 1),
}
LOOPS = {
    "wind": (stereo(wind), 12.0, -20),
    "rain": (stereo(rain), 8.0, -22),
    "river": (river, 8.0, -20),
    "lake": (lake, 10.0, -22),
    "lava": (lava, 10.0, -18),
    "fire": (fire, 8.0, -20),
    "rumble": (rumble, 10.0, -16),
    "roar": (roar, 10.0, -16),
    "underwater": (underwater, 8.0, -22),
}


def write(path, x):
    pcm = np.clip(x, -1.0, 1.0)
    wavfile.write(path, RATE, (pcm * 32767).astype(np.int16))


def stats(x):
    return {
        "seconds": round(len(x) / RATE, 3),
        "peak_db": round(float(20 * np.log10(np.max(np.abs(x)) + 1e-12)), 1),
        "rms_db": round(float(20 * np.log10(np.sqrt(np.mean(x**2)) + 1e-12)), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="just these sounds")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    report_path = os.path.join(OUT, "report.json")
    report = json.load(open(report_path)) if os.path.exists(report_path) and args.only else {}
    for name, (recipe, count) in ONE_SHOTS.items():
        if args.only and name not in args.only:
            continue
        for n in range(count):
            x = recipe(rng_for(f"{name}_{n}"))
            write(os.path.join(OUT, f"{name}_{n}.wav"), x)
            report[f"{name}_{n}"] = stats(x)
    for name, (recipe, seconds, loudness) in LOOPS.items():
        if args.only and name not in args.only:
            continue
        x = to_rms(loop(recipe, seconds), loudness)
        write(os.path.join(OUT, f"{name}.wav"), x)
        report[name] = stats(x) | {"loop": True}
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)
    for name, s in sorted(report.items()):
        print(f"{name:14} {s['seconds']:6.2f} s  peak {s['peak_db']:6.1f} dB  rms {s['rms_db']:6.1f} dB")
    bad = [n for n, s in report.items() if s["peak_db"] > -0.1 or s["rms_db"] < -45]
    if bad:
        sys.exit(f"clipping or near-silent: {bad}")


if __name__ == "__main__":
    main()
