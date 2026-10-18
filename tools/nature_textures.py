"""Draw the alpha-cutout foliage textures used by the nature models.

    python tools/nature_textures.py

Writes assets/textures/nature/*.png: clusters of leafy twigs or needle sprays on a transparent
background, drawn at 4x and downsampled for clean edges. Colours are kept nearly neutral (light,
slightly green) so one texture can serve several species: each model's material tints it. The
clusters fan out from the bottom centre of the texture, where a card attaches to its twig.
"""

import math
import os
import random

import numpy as np
from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "assets", "textures", "nature")
SIZE = 512
SCALE = 4
S = SIZE * SCALE
TAU = 2 * math.pi
TWIG = (115, 95, 75, 255)


def shade(rng, base, spread):
    v = rng.uniform(-spread, spread)
    return tuple(max(0, min(255, int(c * (1.0 + v)))) for c in base)


def leaf_polygon(base, angle, length, width, lobes=0, lobe_depth=0.0, steps=40):
    """Outline of a leaf from `base` along `angle`: a pointed ellipse, optionally lobed."""
    ax, ay = math.cos(angle), math.sin(angle)
    px, py = -ay, ax
    left, right = [], []
    for i in range(steps + 1):
        t = i / steps
        w = width * math.sin(math.pi * t**0.85) ** 0.9
        if lobes:
            w *= 1.0 - lobe_depth * (0.5 - 0.5 * math.cos(TAU * lobes * t))
        x, y = base[0] + ax * length * t, base[1] + ay * length * t
        left.append((x + px * w, y + py * w))
        right.append((x - px * w, y - py * w))
    return left + right[::-1]


def twig(draw, rng, start, end, width, bend=0.12):
    """A slightly curved, tapering twig. Returns its points."""
    (x0, y0), (x1, y1) = start, end
    nx, ny = -(y1 - y0), x1 - x0
    sway = rng.uniform(-bend, bend)
    pts = []
    for i in range(21):
        t = i / 20
        off = math.sin(math.pi * t) * sway
        pts.append((x0 + (x1 - x0) * t + nx * off, y0 + (y1 - y0) * t + ny * off))
    for i in range(20):
        draw.line([pts[i], pts[i + 1]], fill=TWIG, width=max(1, int(width * (1.0 - 0.7 * i / 20))))
    return pts


def leafy_twig(draw, rng, start, end, leaves, length, width, lobes, lobe_depth, base_color):
    stem = twig(draw, rng, start, end, S * 0.012)
    heading = math.atan2(end[1] - start[1], end[0] - start[0])
    for i in range(leaves):
        t = 0.15 + 0.85 * i / max(1, leaves - 1)
        bx, by = stem[min(int(t * 20), 19)]
        side = 1 if i % 2 == 0 else -1
        angle = heading + side * rng.uniform(0.6, 1.1) * (1.0 - 0.55 * t)
        if i == leaves - 1:
            angle = heading + rng.uniform(-0.15, 0.15)
        size = rng.uniform(0.8, 1.1) * (0.8 + 0.3 * math.sin(math.pi * min(1.0, t + 0.2)))
        L, W = length * S * size, width * S * size
        color = shade(rng, base_color, 0.15)
        poly = leaf_polygon((bx, by), angle, L, W, lobes, lobe_depth)
        draw.polygon(poly, fill=color + (255,))
        # Darker half on one side of the midrib, and the midrib itself.
        tip = (bx + math.cos(angle) * L, by + math.sin(angle) * L)
        draw.polygon(poly[: len(poly) // 2] + [tip], fill=tuple(int(c * 0.85) for c in color) + (255,))
        rib_end = (bx + math.cos(angle) * L * 0.9, by + math.sin(angle) * L * 0.9)
        draw.line([(bx, by), rib_end], fill=tuple(min(255, int(c * 1.08)) for c in color) + (255,), width=max(2, int(W * 0.06)))


def leaf_cluster(rng, twigs, leaves, length, width, lobes=0, lobe_depth=0.0, base_color=(205, 225, 175)):
    """Several leafy twigs fanning up from the bottom centre."""
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    base = (S * 0.5, S * 0.99)
    for k in range(twigs):
        a = -math.pi / 2 + (k - (twigs - 1) / 2) * rng.uniform(0.45, 0.6)
        reach = S * rng.uniform(0.62, 0.78) * (1.0 if k == twigs // 2 else 0.9)
        end = (base[0] + math.cos(a) * reach, base[1] + math.sin(a) * reach)
        leafy_twig(draw, rng, base, end, leaves, length, width, lobes, lobe_depth, base_color)
    return img


def needle_spray(rng, base_color=(200, 220, 190)):
    """A flat fan of needled branchlets, like the end of a spruce bough."""
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    main = twig(draw, rng, (S * 0.5, S * 0.99), (S * 0.5, S * 0.03), S * 0.014, bend=0.04)
    branches = [main]
    for i in range(10):
        k = 2 + int(i * 1.6)
        side = 1 if i % 2 == 0 else -1
        bx, by = main[k]
        reach = S * rng.uniform(0.3, 0.42) * (1.0 - 0.05 * i)
        a = -math.pi / 2 + side * rng.uniform(0.75, 1.05)
        branches.append(twig(draw, rng, (bx, by), (bx + math.cos(a) * reach, by + math.sin(a) * reach), S * 0.007))
    for b in branches:
        for k in range(0, 20):
            x, y = b[k]
            dx, dy = b[k + 1][0] - x, b[k + 1][1] - y
            d = math.hypot(dx, dy) or 1.0
            dx, dy = dx / d, dy / d
            for side in (-1, 1):
                for j in range(4):
                    a = math.atan2(dy, dx) + side * rng.uniform(0.45, 0.85)
                    n = S * rng.uniform(0.05, 0.075) * (1.0 - 0.35 * k / 20)
                    ox, oy = x + dx * j * d / 4, y + dy * j * d / 4
                    end = (ox + math.cos(a) * n, oy + math.sin(a) * n)
                    draw.line([(ox, oy), end], fill=shade(rng, base_color, 0.2) + (255,), width=int(S * 0.0075))
    return img


def box_blur(a, r):
    """Mean over a (2r+1)^2 window (edges clamped), for float arrays."""
    pad = np.pad(a, r, mode="edge")
    c = np.cumsum(np.cumsum(pad, axis=0), axis=1)
    c = np.pad(c, ((1, 0), (1, 0)))
    n = 2 * r + 1
    return (c[n:, n:] - c[:-n, n:] - c[n:, :-n] + c[:-n, :-n]) / (n * n)


def finish(img, name):
    small = img.resize((SIZE, SIZE), Image.LANCZOS)
    a = np.asarray(small).astype(np.float32)
    alpha = a[..., 3] / 255.0
    # Bleed colour outwards into transparent texels so filtering never pulls in black.
    covered = alpha > 0.5
    rgb = a[..., :3].copy()
    for radius in (2, 6, 16, 48):
        weight = covered.astype(np.float32)
        summed = np.stack([box_blur(rgb[..., c] * weight, radius) for c in range(3)], axis=-1)
        w = box_blur(weight, radius)[..., None]
        fill = ~covered & (w[..., 0] > 1e-3)
        rgb[fill] = (summed / np.maximum(w, 1e-6))[fill]
        covered = covered | fill
    a[..., :3] = rgb
    out = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")
    os.makedirs(OUT, exist_ok=True)
    out.save(os.path.join(OUT, f"{name}.png"))
    print(f"wrote {name}: {100 * (alpha > 0.5).mean():.0f}% covered")


def maple_leaf(draw, rng, base, angle, size, color):
    """A palmate leaf: five pointed lobes fanning from the stalk end."""
    dark = tuple(int(c * 0.86) for c in color) + (255,)
    for k, (off, scale) in enumerate([(-1.25, 0.6), (-0.62, 0.85), (0.0, 1.0), (0.62, 0.85), (1.25, 0.6)]):
        a = angle + off * 0.8
        poly = leaf_polygon(base, a, size * scale, size * scale * 0.32)
        draw.polygon(poly, fill=(color if k % 2 == 0 else tuple(int(c * 0.93) for c in color)) + (255,))
        draw.line([base, (base[0] + math.cos(a) * size * scale * 0.85, base[1] + math.sin(a) * size * scale * 0.85)], fill=dark, width=max(2, int(size * 0.025)))


def maple_cluster(rng, base_color=(215, 210, 170)):
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    base = (S * 0.5, S * 0.99)
    for k in range(3):
        a = -math.pi / 2 + (k - 1) * rng.uniform(0.45, 0.6)
        reach = S * rng.uniform(0.6, 0.75)
        end = (base[0] + math.cos(a) * reach, base[1] + math.sin(a) * reach)
        stem = twig(draw, rng, base, end, S * 0.012)
        for i in range(5):
            t = 0.25 + 0.75 * i / 4
            bx, by = stem[min(int(t * 20), 19)]
            side = 1 if i % 2 == 0 else -1
            la = a + side * rng.uniform(0.5, 0.9) * (1.0 - 0.5 * t)
            stalk = S * 0.05
            sx, sy = bx + math.cos(la) * stalk, by + math.sin(la) * stalk
            draw.line([(bx, by), (sx, sy)], fill=TWIG, width=int(S * 0.005))
            maple_leaf(draw, rng, (sx, sy), la, S * rng.uniform(0.13, 0.17), shade(rng, base_color, 0.15))
    return img


def snowy_spray(rng):
    """The needle spray with clumps of snow lying along the top of its branchlets."""
    img = needle_spray(rng, base_color=(48, 82, 58))
    draw = ImageDraw.Draw(img)
    for _ in range(70):
        x, y = rng.uniform(0.15, 0.85) * S, rng.uniform(0.05, 0.9) * S
        if img.getpixel((int(x), int(y)))[3] < 128:
            continue
        r = S * rng.uniform(0.012, 0.03)
        draw.ellipse([x - r * 1.6, y - r, x + r * 1.6, y + r * 0.6], fill=(250, 252, 255, 255))
    return img


def frond(rng, leaflet_len, leaflet_w, count, droop, lobes=0, base_color=(205, 225, 175), taper_base=0.25):
    """A pinnate frond along the texture's vertical axis (base at the bottom): a midrib with paired
    leaflets, longest in the middle."""
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    x0 = S * 0.5
    draw.line([(x0, S * 0.995), (x0, S * 0.02)], fill=TWIG, width=int(S * 0.012))
    for i in range(count):
        t = (i + 0.5) / count
        y = S * (0.98 - 0.95 * t)
        envelope = math.sin(math.pi * min(1.0, taper_base + (1 - taper_base) * t)) ** 0.7
        for side in (-1, 1):
            a = -math.pi / 2 + side * (math.pi / 2 - droop) + rng.uniform(-0.06, 0.06)
            L = S * leaflet_len * envelope * rng.uniform(0.9, 1.05)
            W = S * leaflet_w * (0.6 + 0.4 * envelope)
            color = shade(rng, base_color, 0.12)
            poly = leaf_polygon((x0, y), a, L, W, lobes, 0.35 if lobes else 0.0)
            draw.polygon(poly, fill=color + (255,))
            tip = (x0 + math.cos(a) * L, y + math.sin(a) * L)
            draw.polygon(poly[: len(poly) // 2] + [tip], fill=tuple(int(c * 0.86) for c in color) + (255,))
    return img


def strand(rng, base_color=(210, 225, 170)):
    """A hanging willow strand: a thin twig (top of the texture) with narrow leaves along it."""
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    pts = twig(draw, rng, (S * 0.5, S * 0.01), (S * 0.5 + rng.uniform(-0.04, 0.04) * S, S * 0.99), S * 0.008, bend=0.05)
    for k in range(1, 20):
        x, y = pts[k]
        for j in range(3):
            side = rng.choice((-1, 1))
            a = math.pi / 2 + side * rng.uniform(0.25, 0.6)
            L = S * rng.uniform(0.12, 0.18)
            draw.polygon(leaf_polygon((x, y + j * S * 0.015), a, L, L * 0.16), fill=shade(rng, base_color, 0.14) + (255,))
    return img


def tuft(rng, blades, height, width, base_color=(195, 220, 160), heads=0, spread=0.35):
    """Grass blades rising from the bottom of the texture, curving outwards; darker at the base."""
    img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    for _ in range(blades):
        x0 = S * (0.5 + rng.uniform(-spread, spread))
        lean = rng.uniform(-0.35, 0.35) + (x0 / S - 0.5) * 0.8
        h = S * height * rng.uniform(0.55, 1.0)
        w = S * width * rng.uniform(0.7, 1.3)
        color = shade(rng, base_color, 0.18)
        steps = 12
        left, right = [], []
        for i in range(steps + 1):
            t = i / steps
            x = x0 + lean * h * t * t
            y = S * 0.999 - h * t
            half = w * (1 - t) ** 0.8 / 2
            left.append((x - half, y))
            right.append((x + half, y))
        draw.polygon(left + right[::-1], fill=color + (255,))
    for _ in range(heads):
        x0 = S * (0.5 + rng.uniform(-spread * 0.6, spread * 0.6))
        h = S * height * rng.uniform(0.85, 1.05)
        lean = rng.uniform(-0.15, 0.15)
        top = (x0 + lean * h, S - h)
        draw.line([(x0, S * 0.999), top], fill=(150, 160, 110, 255), width=int(S * 0.004))
        draw.ellipse([top[0] - S * 0.012, top[1] - S * 0.002, top[0] + S * 0.012, top[1] + S * 0.07], fill=(200, 180, 130, 255))
    # Darken towards the roots (self-shadowing inside the tuft).
    a = np.asarray(img).astype(np.float32)
    rows = np.arange(S, dtype=np.float32)[:, None]
    a[..., :3] *= (0.6 + 0.4 * np.clip((S - rows) / (S * 0.45), 0.0, 1.0))[..., None]
    return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8), "RGBA")


def main():
    finish(leaf_cluster(random.Random(1), 3, 9, 0.17, 0.05, lobes=3, lobe_depth=0.45), "leaves_oak")
    finish(leaf_cluster(random.Random(2), 3, 11, 0.13, 0.045), "leaves_round")
    finish(needle_spray(random.Random(3)), "needles")
    finish(maple_cluster(random.Random(4)), "leaves_maple")
    finish(snowy_spray(random.Random(5)), "needles_snow")
    finish(frond(random.Random(6), 0.46, 0.014, 44, 0.8), "palm_frond")
    finish(frond(random.Random(7), 0.22, 0.05, 18, 0.25, lobes=4, taper_base=0.1), "fern_frond")
    finish(strand(random.Random(8)), "willow_strand")
    finish(tuft(random.Random(9), 70, 0.95, 0.035), "grass_tuft")
    finish(tuft(random.Random(10), 45, 0.98, 0.028, base_color=(200, 215, 150), heads=6), "grass_tall")
    finish(tuft(random.Random(11), 40, 0.95, 0.018, base_color=(225, 220, 165), spread=0.3), "grass_thin")


if __name__ == "__main__":
    main()
