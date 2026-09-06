#!/usr/bin/env python3
"""Slice the raw sheets in public/set/tmp into structured generative inputs.

Output: public/set/axp/<layer>/<layer>-<n>.png plus public/set/axp/manifest.json.

Layers (bottom to top when composing):
  bg       square backgrounds                         from bgs.png (15x15 grid)
  base     villager faces, full-bleed icons           from base.png (22x22 grid)
  alt      alternative portrait bases                 from alt_base.png (black grid)
  head     the mannequin head wearing hair or a hat   from hats2.png (teal grid) and hats.png (packed)
  glasses  face accessories with the mannequin removed from onface.png (aligned grid)
  topper   isolated headwear icons                    from mwsc1.zip (Cap*.png)

Every head and glasses entry records the mannequin's skin bounding box so a
composer can align layers from different sheets: put a glasses layer on a head
by mapping one skin box onto the other.

Usage: python tools/slice.py [--zip path/to/mwsc1.zip]
"""
from __future__ import annotations

import argparse
import json
import os
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

ROOT = Path(__file__).resolve().parent.parent
TMP = ROOT / "public" / "set" / "tmp"
OUT = ROOT / "public" / "set" / "axp"

# The mannequin's skin, sampled from the sheets: a warm tan with darker shading.
SKIN = np.array([[154, 97, 55], [109, 68, 43], [196, 121, 64], [138, 88, 47], [99, 66, 33], [176, 110, 60]])


def load(name: str) -> np.ndarray:
    return np.array(Image.open(TMP / name).convert("RGBA"))


def skin_mask(rgba: np.ndarray, tolerance: float = 34.0) -> np.ndarray:
    rgb = rgba[:, :, :3].astype(np.int32)
    alpha = rgba[:, :, 3] > 128
    best = np.full(rgb.shape[:2], 1e9)
    for colour in SKIN:
        dist = np.sqrt(((rgb - colour) ** 2).sum(axis=2))
        best = np.minimum(best, dist)
    return (best < tolerance) & alpha


def bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    ys, xs = np.where(mask)
    if not len(xs):
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def trim(rgba: np.ndarray, pad: int = 2) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop to the alpha bounding box. Returns the crop and its (x, y) offset."""
    box = bbox(rgba[:, :, 3] > 8)
    if not box:
        return rgba, (0, 0)
    x0, y0, x1, y1 = box
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(rgba.shape[1], x1 + pad), min(rgba.shape[0], y1 + pad)
    return rgba[y0:y1, x0:x1], (x0, y0)


def save(layer: str, index: int, rgba: np.ndarray) -> str:
    folder = OUT / layer
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{layer}-{index}.png"
    Image.fromarray(rgba).save(folder / name, optimize=True)
    return f"{layer}/{name}"


def grid_cells(rgba: np.ndarray, lines_x: list[int], lines_y: list[int]):
    for row, (y0, y1) in enumerate(zip(lines_y, lines_y[1:])):
        for col, (x0, x1) in enumerate(zip(lines_x, lines_x[1:])):
            yield row, col, rgba[y0 + 1 : y1, x0 + 1 : x1]


def line_positions(mask_1d: np.ndarray, size: int) -> list[int]:
    """Collapse runs of separator pixels into single line positions, with edges."""
    idx = np.where(mask_1d)[0]
    lines: list[int] = []
    for i in idx:
        if lines and i - lines[-1] <= 2:
            lines[-1] = int(i)
        else:
            lines.append(int(i))
    if not lines or lines[0] > 4:
        lines.insert(0, -1)
    if lines[-1] < size - 4:
        lines.append(size)
    return lines


def slice_backgrounds(manifest: dict) -> None:
    sheet = load("bgs.png")
    alpha = sheet[:, :, 3] > 8
    xs = line_positions(alpha.sum(axis=0) == 0, sheet.shape[1])
    ys = line_positions(alpha.sum(axis=1) == 0, sheet.shape[0])
    count = 0
    for row, col, cell in grid_cells(sheet, xs, ys):
        if (cell[:, :, 3] > 8).mean() < 0.9:
            continue
        crop, _ = trim(cell, pad=0)
        manifest["bg"].append({"file": save("bg", count, crop), "source": f"bgs.png r{row} c{col}"})
        count += 1


def slice_bases(manifest: dict) -> None:
    sheet = load("base.png")
    alpha = sheet[:, :, 3] > 8
    xs = line_positions(alpha.sum(axis=0) == 0, sheet.shape[1])
    ys = line_positions(alpha.sum(axis=1) == 0, sheet.shape[0])
    count = 0
    for row, col, cell in grid_cells(sheet, xs, ys):
        if (cell[:, :, 3] > 8).sum() < 400:
            continue
        crop, _ = trim(cell)
        manifest["base"].append({"file": save("base", count, crop), "source": f"base.png r{row} c{col}"})
        count += 1


def slice_alt_bases(manifest: dict) -> None:
    sheet = load("alt_base.png")
    rgb = sheet[:, :, :3].astype(int)
    black = rgb.sum(axis=2) < 30
    xs = line_positions(black.mean(axis=0) > 0.5, sheet.shape[1])
    ys = line_positions(black.mean(axis=1) > 0.5, sheet.shape[0])
    count = 0
    for row, col, cell in grid_cells(sheet, xs, ys):
        if cell.size == 0:
            continue
        # cells are drawn on a flat grey; key it out and skip empty cells
        grey = (np.abs(cell[:, :, :3].astype(int) - 100).sum(axis=2) < 24) | (cell[:, :, :3].astype(int).sum(axis=2) < 30)
        keyed = cell.copy()
        keyed[:, :, 3] = np.where(grey, 0, cell[:, :, 3])
        if (keyed[:, :, 3] > 8).mean() < 0.15:
            continue
        crop, _ = trim(keyed)
        manifest["alt"].append({"file": save("alt", count, crop), "source": f"alt_base.png r{row} c{col}"})
        count += 1


def head_entry(layer: str, index: int, cell: np.ndarray, source: str) -> dict | None:
    crop, (ox, oy) = trim(cell)
    skin = skin_mask(crop)
    box = bbox(skin)
    if not box or (box[2] - box[0]) < 20:
        return None
    return {
        "file": save(layer, index, crop),
        "source": source,
        "size": [int(crop.shape[1]), int(crop.shape[0])],
        "skin": list(box),
    }


def slice_heads(manifest: dict) -> None:
    count = 0
    # hats2: teal grid lines every 129px
    sheet = load("hats2.png")
    rgb = sheet[:, :, :3].astype(int)
    teal = (np.abs(rgb[:, :, 0]) < 20) & (np.abs(rgb[:, :, 1] - 128) < 25) & (np.abs(rgb[:, :, 2] - 128) < 25) & (sheet[:, :, 3] > 200)
    xs = line_positions(teal.mean(axis=0) > 0.6, sheet.shape[1])
    ys = line_positions(teal.mean(axis=1) > 0.6, sheet.shape[0])
    for row, col, cell in grid_cells(sheet, xs, ys):
        cell = cell.copy()
        # drop any grid-line remnants
        t = (np.abs(cell[:, :, 0].astype(int)) < 20) & (np.abs(cell[:, :, 1].astype(int) - 128) < 25) & (np.abs(cell[:, :, 2].astype(int) - 128) < 25)
        cell[:, :, 3] = np.where(t, 0, cell[:, :, 3])
        if (cell[:, :, 3] > 8).sum() < 800:
            continue
        entry = head_entry("head", count, cell, f"hats2.png r{row} c{col}")
        if entry:
            manifest["head"].append(entry)
            count += 1
    # hats: packed sheet, connected components on alpha
    sheet = load("hats.png")
    labels, n = ndimage.label(sheet[:, :, 3] > 8)
    for label in range(1, n + 1):
        ys_, xs_ = np.where(labels == label)
        if len(xs_) < 1500:
            continue
        x0, x1, y0, y1 = xs_.min(), xs_.max() + 1, ys_.min(), ys_.max() + 1
        cell = sheet[y0:y1, x0:x1].copy()
        cell[:, :, 3] = np.where(labels[y0:y1, x0:x1] == label, cell[:, :, 3], 0)
        entry = head_entry("head", count, cell, f"hats.png component {label}")
        if entry:
            manifest["head"].append(entry)
            count += 1


def slice_glasses(manifest: dict) -> None:
    """onface.png: accessories drawn on one mannequin. Each head is a connected
    component; aligning them on the skin box and taking the per-pixel median gives
    the bare head, and each tile minus that template is the accessory alone.
    Accessory offsets are recorded relative to the skin box so a composer can
    place them on any head by mapping skin boxes."""
    sheet = load("onface.png")
    labels, n = ndimage.label(sheet[:, :, 3] > 8)
    crops = []
    for label in range(1, n + 1):
        ys_, xs_ = np.where(labels == label)
        if len(xs_) < 1500:
            continue
        x0, x1, y0, y1 = xs_.min(), xs_.max() + 1, ys_.min(), ys_.max() + 1
        cell = sheet[y0:y1, x0:x1].copy()
        cell[:, :, 3] = np.where(labels[y0:y1, x0:x1] == label, cell[:, :, 3], 0)
        skin = bbox(skin_mask(cell))
        if not skin or (skin[2] - skin[0]) < 40:
            continue
        crops.append((cell, skin))
    # canvas: every crop placed so its skin box centre-x / bottom-y coincide
    margin = 40
    sw = int(np.median([s[2] - s[0] for _, s in crops]))
    sh = int(np.median([s[3] - s[1] for _, s in crops]))
    W, H = sw + 2 * margin, sh + 2 * margin
    anchor = (W // 2, margin + sh)  # skin centre-x, skin bottom
    stack = []
    for cell, skin in crops:
        canvas = np.zeros((H, W, 4), dtype=np.uint8)
        cx = (skin[0] + skin[2]) // 2
        dx, dy = anchor[0] - cx, anchor[1] - skin[3]
        ys0, xs0 = max(0, dy), max(0, dx)
        h = min(cell.shape[0], H - ys0)
        w = min(cell.shape[1], W - xs0)
        src_y, src_x = max(0, -dy), max(0, -dx)
        canvas[ys0 : ys0 + h - src_y, xs0 : xs0 + w - src_x] = cell[src_y:h, src_x:w]
        stack.append(canvas)
    arr = np.stack(stack).astype(np.int16)
    template = np.median(arr, axis=0).astype(np.int16)
    template_skin = bbox(skin_mask(template.astype(np.uint8)))
    manifest["glasses_template"] = {"size": [int(W), int(H)], "skin": list(template_skin) if template_skin else None}
    save("glasses", 999, template.astype(np.uint8))  # for inspection; not a layer
    count = 0
    for canvas in arr:
        colour_diff = np.abs(canvas[:, :, :3] - template[:, :, :3]).sum(axis=2)
        alpha_diff = np.abs(canvas[:, :, 3] - template[:, :, 3])
        tile = canvas.astype(np.uint8)
        # differs from the bare head, and is not itself skin (edge slivers from
        # sub-pixel misalignment are skin-coloured)
        mask = ((colour_diff > 80) | (alpha_diff > 80)) & (tile[:, :, 3] > 20) & ~skin_mask(tile, 46)
        mask = ndimage.binary_opening(mask, iterations=1)
        labels_, n_ = ndimage.label(mask)
        if n_:
            sizes = ndimage.sum(mask, labels_, range(1, n_ + 1))
            keep = np.isin(labels_, [i + 1 for i, size in enumerate(sizes) if size >= 40])
            mask &= keep
        mask = ndimage.binary_dilation(mask, iterations=1)
        if mask.sum() < 80:
            continue  # a bare head
        acc = canvas.astype(np.uint8).copy()
        acc[:, :, 3] = np.where(mask, acc[:, :, 3], 0)
        crop, (ox, oy) = trim(acc, pad=1)
        manifest["glasses"].append(
            {
                "file": save("glasses", count, crop),
                "source": "onface.png",
                "size": [int(crop.shape[1]), int(crop.shape[0])],
                # offset of the crop's top-left from the skin box's top-left, in template pixels
                "offset": [int(ox - template_skin[0]), int(oy - template_skin[1])] if template_skin else [int(ox), int(oy)],
            }
        )
        count += 1


def slice_toppers(manifest: dict, archive: Path) -> None:
    kinds = {
        "CapHat": "hat",
        "CapFullface": "helmet",
        "CapOrnament": "ornament",
        "CapMask": "mask",
        "CapCostume": "costume",
        "CapKnit": "hat",
        "CapHelmet": "helmet",
    }
    count = 0
    with zipfile.ZipFile(archive) as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            name = os.path.basename(info.filename)
            if "__MACOSX" in info.filename or not name.lower().endswith(".png") or not name.startswith("Cap"):
                continue
            kind = next((v for k, v in kinds.items() if name.startswith(k)), None)
            if not kind:
                continue
            with zf.open(info) as fh:
                rgba = np.array(Image.open(fh).convert("RGBA"))
            crop, _ = trim(rgba, pad=1)
            manifest["topper"].append(
                {
                    "file": save("topper", count, crop),
                    "source": name,
                    "kind": kind,
                    "size": [int(crop.shape[1]), int(crop.shape[0])],
                }
            )
            count += 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", default=str(ROOT / "mwsc1.zip"))
    args = parser.parse_args()
    manifest: dict = {"bg": [], "base": [], "alt": [], "head": [], "glasses": [], "topper": []}
    slice_backgrounds(manifest)
    slice_bases(manifest)
    slice_alt_bases(manifest)
    slice_heads(manifest)
    slice_glasses(manifest)
    if Path(args.zip).exists():
        slice_toppers(manifest, Path(args.zip))
    manifest["layers"] = ["bg", "base|alt|head", "glasses", "topper"]
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    for layer in ("bg", "base", "alt", "head", "glasses", "topper"):
        print(f"{layer:8} {len(manifest[layer])}")


if __name__ == "__main__":
    main()
