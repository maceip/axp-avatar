# AXP avatar set

Generative inputs sliced from the sheets in `public/set/tmp` and `mwsc1.zip` by
`tools/slice.py`, composed into avatars by `tools/compose.py`. Everyone in the
AXP family photo is drawn from this set, so every portrait shares one style.

## Layers (bottom to top)

| Layer     | Count | From                         | What it is                                                    |
| --------- | ----- | ---------------------------- | ------------------------------------------------------------- |
| `bg`      | 209   | `bgs.png` (15×15 grid)       | square backgrounds                                            |
| `head`    | 1065  | `hats2.png`, `hats.png`      | the featureless mannequin head wearing hair or a hat          |
| `base`    | 479   | `base.png` (22×22 grid)      | villager faces, used whole as an alternative subject          |
| `alt`     | 108   | `alt_base.png`               | alternative portraits, grey key removed                       |
| `glasses` | 90    | `onface.png`                 | face accessories with the mannequin subtracted                |
| `topper`  | 53    | `mwsc1.zip` `Cap*.png`       | isolated headwear: hat, helmet, ornament, mask, costume       |
| notion    | —     | `public/avatar/part`         | eyebrows, eyes, nose, mouth, beard: the notion pass           |

`manifest.json` lists every file with its source cell. Heads carry `skin`
(the mannequin's skin bounding box) and `skin_fraction`; glasses carry an
`offset` from the skin box of the bare-head template (`glasses/glasses-999.png`,
kept for inspection). Two layers from different sheets are aligned by mapping
one skin box onto the other, which is how glasses land on eyes and hats land
on heads without any per-item tuning.

## How the composer uses them

`compose(seed)` is deterministic: a contributor's principal id is a stable
portrait.

1. Background (unless `--transparent`), with a soft vignette.
2. Subject: 70% a mannequin `head`, 22% a villager `base`, 8% an `alt` portrait.
3. On a mannequin head: skin retoned by seed (hue ±8°, six lightness steps),
   then the **notion pass**: eyebrows, eyes, nose, mouth and sometimes a beard
   from the notion-avatar line parts, scaled so the notion face box
   (220,367)–(784,871) in the 1080 master space fits the head's skin box, and
   re-inked to warm charcoal. Heads with `skin_fraction < 0.2` are full masks
   or helmets and get no face.
4. `glasses` (45%) aligned by skin box; a `topper` (28%) sat on the crown,
   helmets over the whole head, flower ornaments lower.
5. Villager and alt subjects get a hat 30% of the time and no drawn face.

```sh
python tools/slice.py                              # regenerate the set (~20 s)
python tools/compose.py --seed alice --out alice.png
python tools/compose.py --seed alice --transparent # for the family photo
python tools/compose.py --sheet 48 --out sheet.png # review the space
```

Python 3.11+, `pip install pillow numpy scipy cairosvg`.

## Where this plugs into AXP

The family photo (branch `exp/onboarding-family-photo` in the AXP repo) shows
every portrait posted to the project's `family-photo` session, in join order.
An agent's onboarding task is to make its portrait and post it; this composer
is the "our tool" option in that task: `compose.py --seed <principal>
--transparent` gives a 512×512 PNG under 1 MB to upload with `_axp/blobPut`.
The big empty group scene goes through the photo's `scene` prop when it lands
in this repo.

## Provenance

The sheets and the icon archive are Nintendo (Animal Crossing) and Bandai
Namco (Dragon Ball) renders. They are fine as internal prototype inputs and as
a target for the style; anything we publish or ship should replace them with
our own drawings in the same slots. The pipeline does not care where the
pixels come from: keep the folder names and the manifest shape and the
composer keeps working. The notion-avatar parts are MIT (Mayandev).
