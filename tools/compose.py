#!/usr/bin/env python3
"""Compose an avatar from the sliced layers in public/set/axp.

  python tools/compose.py --seed alice --out alice.png
  python tools/compose.py --sheet 48 --out sheet.png      # a grid to review the space

Deterministic: the same seed always gives the same avatar, so a contributor's
principal id is a stable portrait. Layers (bottom to top): background, subject
(mannequin head with hair/hat, or a villager face, or an alternative portrait),
face accessory aligned by skin box, headwear anchored to the top of the head.
Skin tone is varied by seed.
"""
from __future__ import annotations

import argparse
import colorsys
import hashlib
import json
import random
from pathlib import Path

import io
import re

import cairosvg
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SET = ROOT / "public" / "set" / "axp"
PARTS = ROOT / "public" / "avatar" / "part"
SIZE = 512
# The notion-avatar parts live in a 1080x1080 master space; the face shape
# occupies this box. Features are placed by mapping it onto a head's skin box.
NOTION_FACE = (220, 367, 784, 871)
NOTION_COUNTS = {"eyebrows": 16, "eyes": 14, "nose": 14, "mouth": 20, "beard": 16}
INK = (43, 38, 34)

SKIN = np.array([[154, 97, 55], [109, 68, 43], [196, 121, 64], [138, 88, 47], [99, 66, 33], [176, 110, 60]])


def load_manifest() -> dict:
    return json.loads((SET / "manifest.json").read_text())


def rng_for(seed: str) -> random.Random:
    digest = hashlib.sha256(seed.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def open_layer(entry: dict) -> Image.Image:
    return Image.open(SET / entry["file"]).convert("RGBA")


def skin_mask(rgba: np.ndarray, tolerance: float = 40.0) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.int32)
    best = np.full(rgb.shape[:2], 1e9)
    for colour in SKIN:
        best = np.minimum(best, np.sqrt(((rgb - colour) ** 2).sum(axis=2)))
    return (best < tolerance) & (rgba[:, :, 3] > 40)


def retone(image: Image.Image, hue_shift: float, value_scale: float, sat_scale: float) -> Image.Image:
    """Shift the mannequin's skin: hue in degrees, value and saturation multipliers.
    Only skin-coloured pixels change, so hair, hats and glasses keep their colours."""
    arr = np.array(image).astype(np.float32)
    mask = skin_mask(arr.astype(np.uint8))
    if not mask.any():
        return image
    rgb = arr[mask][:, :3] / 255.0
    hsv = np.array([colorsys.rgb_to_hsv(*px) for px in rgb])
    hsv[:, 0] = (hsv[:, 0] + hue_shift / 360.0) % 1.0
    hsv[:, 1] = np.clip(hsv[:, 1] * sat_scale, 0, 1)
    hsv[:, 2] = np.clip(hsv[:, 2] * value_scale, 0, 1)
    out = np.array([colorsys.hsv_to_rgb(*px) for px in hsv]) * 255.0
    arr[mask, :3] = out
    return Image.fromarray(arr.astype(np.uint8))


def fit(image: Image.Image, height: int) -> tuple[Image.Image, float]:
    scale = height / image.height
    return image.resize((max(1, round(image.width * scale)), height), Image.LANCZOS), scale


def paste_centered(canvas: Image.Image, layer: Image.Image, cx: float, bottom: float) -> tuple[int, int]:
    x = round(cx - layer.width / 2)
    y = round(bottom - layer.height)
    canvas.alpha_composite(layer, (x, y))
    return x, y


_svg_meta_cache: dict[str, tuple[int, int, float, float] | None] = {}


def notion_part(kind: str, index: int) -> tuple[Path, int, int, float, float] | None:
    """A part's file, pixel size and master-space position. Parts either carry
    their master offset in a top-level translate or are drawn on the full 1080
    canvas; anything else cannot be placed and is skipped."""
    path = PARTS / kind / f"{kind}-{index}.svg"
    key = str(path)
    if key not in _svg_meta_cache:
        text = path.read_text()
        w = int(float(re.search(r'width="([\d.]+)(?:px)?"', text).group(1)))
        h = int(float(re.search(r'height="([\d.]+)(?:px)?"', text).group(1)))
        m = re.search(r"translate\((-?[\d.]+),\s*(-?[\d.]+)\)", text)
        if m:
            _svg_meta_cache[key] = (w, h, -float(m.group(1)), -float(m.group(2)))
        elif w == 1080 and h == 1080:
            _svg_meta_cache[key] = (w, h, 0.0, 0.0)
        else:
            _svg_meta_cache[key] = None
    meta = _svg_meta_cache[key]
    return (path, *meta) if meta else None


def usable_parts(kind: str) -> list[int]:
    return [i for i in range(NOTION_COUNTS[kind]) if notion_part(kind, i)]


def notion_pass(canvas: Image.Image, skin_box: list[float], rng: random.Random) -> None:
    """Draw a face onto a featureless mannequin with the notion-avatar line parts:
    eyebrows, eyes, nose, mouth, sometimes a beard. Black strokes are re-inked
    to the workspace's warm charcoal."""
    skin_w = skin_box[2] - skin_box[0]
    scale = skin_w / (NOTION_FACE[2] - NOTION_FACE[0])
    face_cx = (skin_box[0] + skin_box[2]) / 2
    face_cy = (skin_box[1] + skin_box[3]) / 2 + skin_w * 0.02
    master_cx = (NOTION_FACE[0] + NOTION_FACE[2]) / 2
    master_cy = (NOTION_FACE[1] + NOTION_FACE[3]) / 2
    kinds = ["eyebrows", "eyes", "nose", "mouth"] + (["beard"] if rng.random() < 0.12 else [])
    for kind in kinds:
        options = usable_parts(kind)
        if not options:
            continue
        path, w, h, x, y = notion_part(kind, rng.choice(options))
        # the notion face is taller than the round mannequin; beards get pulled in
        k = 0.8 if kind == "beard" else 1.0
        out_w, out_h = max(1, round(w * scale * k)), max(1, round(h * scale * k))
        png = cairosvg.svg2png(url=str(path), output_width=out_w, output_height=out_h)
        layer = Image.open(io.BytesIO(png)).convert("RGBA")
        arr = np.array(layer)
        dark = arr[:, :, 3] > 0
        arr[dark, :3] = INK
        layer = Image.fromarray(arr)
        px = round(face_cx + (x - master_cx) * scale * k)
        py = round(face_cy + (y - master_cy) * scale * k)
        canvas.alpha_composite(layer, (px, py))


def compose(seed: str, manifest: dict) -> Image.Image:
    rng = rng_for(seed)
    canvas = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))

    # background (skipped for transparent portraits, e.g. the family photo)
    if manifest["bg"]:
        bg = open_layer(rng.choice(manifest["bg"])).resize((SIZE, SIZE), Image.LANCZOS)
        canvas.alpha_composite(bg)
        # a soft vignette so the subject sits in the frame
        yy, xx = np.mgrid[0:SIZE, 0:SIZE]
        r = np.sqrt(((xx - SIZE / 2) / (SIZE / 2)) ** 2 + ((yy - SIZE / 2) / (SIZE / 2)) ** 2)
        vignette = Image.fromarray((np.clip((r - 0.75) / 0.6, 0, 1) * 110).astype(np.uint8))
        shade = Image.new("RGBA", (SIZE, SIZE), (30, 40, 30, 255))
        shade.putalpha(vignette)
        canvas.alpha_composite(shade)
    else:
        rng.random()  # keep the seed's draw sequence identical with or without a background

    family = rng.random()
    if family < 0.7 and manifest["head"]:
        subject_kind = "head"
        entry = rng.choice(manifest["head"])
        subject = open_layer(entry)
        subject = retone(
            subject,
            hue_shift=rng.uniform(-8, 10),
            value_scale=rng.choice([0.62, 0.78, 0.9, 1.0, 1.08, 1.18]),
            sat_scale=rng.uniform(0.85, 1.1),
        )
        skin = entry["skin"]
    elif family < 0.92 and manifest["base"]:
        subject_kind = "base"
        entry = rng.choice(manifest["base"])
        subject = open_layer(entry)
        skin = None
    else:
        subject_kind = "alt"
        entry = rng.choice(manifest["alt"] or manifest["base"])
        subject = open_layer(entry)
        skin = None

    target_h = round(SIZE * (0.74 if subject_kind == "head" else 0.7))
    subject, scale = fit(subject, target_h)
    sx, sy = paste_centered(canvas, subject, SIZE / 2, SIZE * 0.9)

    if subject_kind == "head" and skin:
        # skin box in canvas space
        skin_box = [sx + skin[0] * scale, sy + skin[1] * scale, sx + skin[2] * scale, sy + skin[3] * scale]
        skin_w = skin_box[2] - skin_box[0]
        skin_h = skin_box[3] - skin_box[1]
        costume = entry.get("skin_fraction", 1) < 0.2  # a full mask or helmet: no face to draw
        if not costume:
            notion_pass(canvas, skin_box, rng)
        if not costume and manifest["glasses"] and rng.random() < 0.45:
            g = rng.choice(manifest["glasses"])
            tskin = manifest["glasses_template"]["skin"]
            gscale = skin_w / (tskin[2] - tskin[0])
            layer = open_layer(g)
            layer = layer.resize((max(1, round(layer.width * gscale)), max(1, round(layer.height * gscale))), Image.LANCZOS)
            canvas.alpha_composite(
                layer,
                (round(skin_box[0] + g["offset"][0] * gscale), round(skin_box[1] + g["offset"][1] * gscale)),
            )
        if not costume and manifest["topper"] and rng.random() < 0.28:
            t = rng.choice(manifest["topper"])
            layer = open_layer(t)
            kind = t["kind"]
            width = skin_w * (1.45 if kind == "helmet" else 1.15 if kind in ("hat", "costume") else 1.05)
            layer = layer.resize((max(1, round(width)), max(1, round(layer.height * width / layer.width))), Image.LANCZOS)
            if kind == "helmet":
                bottom = skin_box[1] + skin_h * 0.95
            elif kind == "ornament":
                bottom = skin_box[1] + skin_h * 0.42
            elif kind == "mask":
                bottom = skin_box[1] + skin_h * 0.8
            else:
                bottom = skin_box[1] + skin_h * 0.3
            paste_centered(canvas, layer, (skin_box[0] + skin_box[2]) / 2, bottom)
    elif manifest["topper"] and rng.random() < 0.3:
        # a hat on a villager or portrait: sit it on the top third of the subject
        t = rng.choice([x for x in manifest["topper"] if x["kind"] in ("hat", "ornament")] or manifest["topper"])
        layer = open_layer(t)
        width = subject.width * 0.8
        layer = layer.resize((max(1, round(width)), max(1, round(layer.height * width / layer.width))), Image.LANCZOS)
        paste_centered(canvas, layer, SIZE / 2, sy + subject.height * 0.3)
    return canvas


def sheet(count: int, manifest: dict, seed: str) -> Image.Image:
    cols = 8
    rows = (count + cols - 1) // cols
    cell = 160
    out = Image.new("RGB", (cols * cell, rows * cell), "#f6f7f4")
    for i in range(count):
        avatar = compose(f"{seed}-{i}", manifest).resize((cell - 8, cell - 8), Image.LANCZOS)
        out.paste(avatar, ((i % cols) * cell + 4, (i // cols) * cell + 4), avatar)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default="axp")
    parser.add_argument("--out", default="avatar.png")
    parser.add_argument("--sheet", type=int, default=0, help="render N avatars in a grid instead of one")
    parser.add_argument("--transparent", action="store_true", help="no background: for the family photo")
    args = parser.parse_args()
    manifest = load_manifest()
    if args.transparent:
        manifest = {**manifest, "bg": []}
    image = sheet(args.sheet, manifest, args.seed) if args.sheet else compose(args.seed, manifest)
    image.save(args.out)
    print(args.out)


if __name__ == "__main__":
    main()
