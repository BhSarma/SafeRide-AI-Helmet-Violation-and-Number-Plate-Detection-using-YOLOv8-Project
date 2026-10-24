"""
prepare_data.py - merge helmet_plate + extra_data/* into ONE clean YOLO dataset.

Run from the project folder:
    python prepare_data.py                  # DRY RUN: shows class mapping + counts, writes nothing
    python prepare_data.py --go             # builds merged_data/
Options:
    --skip dataset3 dataset4   leave out datasets (folder names)
    --cap 4                    keep at most N near-identical images per group
    --negatives hard_neg       folder of images with NO objects (pedestrians, cyclists, ...)
    --out merged_data  --extra extra_data  --base helmet_plate

What it does
  1. Reads every dataset's data.yaml and maps its class names to:
        0 helmet | 1 license_plate | 2 motorcyclist
     "without/no helmet" classes are dropped (a rider with no helmet box = violation).
  2. Finds near-duplicate images (frames of the same video etc.) and groups them.
  3. Re-splits train/valid/test BY GROUP, so a test image never has a twin in train.
"""
import argparse
import json
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
TARGET = ["helmet", "license_plate", "motorcyclist"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_DIRS = {"train": "train", "valid": "valid", "val": "valid", "test": "test"}


def norm(s):
    return "".join(ch for ch in str(s).lower() if ch.isalnum())


def map_class(name):
    """Return 'helmet' | 'license_plate' | 'motorcyclist' | 'NOHELMET' | None (unknown)."""
    n = norm(name)
    if "helmet" in n and any(k in n for k in ("without", "no", "non", "not", "bare", "head")):
        return "NOHELMET"
    if "plate" in n or "licen" in n:
        return "license_plate"
    if "helmet" in n:
        return "helmet"
    if any(k in n for k in ("rider", "motorcycl", "motorbike", "bike", "twowheel", "scooter")):
        return "motorcyclist"
    return None


def read_names(ds_dir):
    y = ds_dir / "data.yaml"
    if not y.exists():
        return None
    names = yaml.safe_load(y.read_text()).get("names", [])
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names)


def parse_label(path, cmap):
    """Return (list of (cls_id, cx, cy, w, h), n_original_lines)."""
    out, n = [], 0
    if not path.exists():
        return out, 0
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 5:
            continue
        n += 1
        try:
            c = int(float(p[0]))
            v = [float(x) for x in p[1:]]
        except ValueError:
            continue
        tgt = cmap.get(c)
        if tgt not in TARGET:
            continue
        if len(v) == 4:
            cx, cy, w, h = v
        else:  # polygon -> box
            xs, ys = v[0::2], v[1::2]
            x1, x2, y1, y2 = min(xs), max(xs), min(ys), max(ys)
            cx, cy, w, h = (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1
        x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2)
        x2, y2 = min(1.0, cx + w / 2), min(1.0, cy + h / 2)
        w, h = x2 - x1, y2 - y1
        if w < 0.002 or h < 0.002:
            continue
        out.append((TARGET.index(tgt), (x1 + x2) / 2, (y1 + y2) / 2, w, h))
    return out, n


def dhash(path):
    im = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    s = cv2.resize(im, (9, 8), interpolation=cv2.INTER_AREA).astype(np.int16)
    bits = (s[:, 1:] > s[:, :-1]).flatten()
    return int(np.packbits(bits).view(np.uint64)[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--go", action="store_true", help="actually build the merged dataset")
    ap.add_argument("--base", default="helmet_plate")
    ap.add_argument("--extra", default="extra_data")
    ap.add_argument("--out", default="merged_data")
    ap.add_argument("--skip", nargs="*", default=[])
    ap.add_argument("--cap", type=int, default=4)
    ap.add_argument("--thr", type=int, default=6, help="near-duplicate hash distance (of 64)")
    ap.add_argument("--negatives", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    base, extra, out = ROOT / a.base, ROOT / a.extra, ROOT / a.out
    datasets = []
    if base.exists():
        datasets.append(base)
    if extra.exists():
        datasets += sorted(p for p in extra.iterdir() if p.is_dir())
    datasets = [d for d in datasets if d.name not in a.skip]
    if not datasets:
        sys.exit("No datasets found. Check --base / --extra.")

    # ---------- 1. class mapping + collect items ----------
    items = []  # dict(img, lbl, ds, boxes, orig)
    print("=" * 70, "\nCLASS MAPPING (check this!)\n" + "=" * 70)
    for d in datasets:
        names = read_names(d)
        if names is None:
            print(f"[{d.name}] no data.yaml -> SKIPPED")
            continue
        cmap, print_rows = {}, []
        for i, nme in enumerate(names):
            t = map_class(nme)
            cmap[i] = t
            print_rows.append(f"    {i}: {nme!r:22s} -> {t if t else 'DROPPED (unknown)'}")
        print(f"[{d.name}]  classes: {len(names)}")
        print("\n".join(print_rows))
        cnt, n_img, dropped_img = Counter(), 0, 0
        for sp, spn in SPLIT_DIRS.items():
            idir, ldir = d / sp / "images", d / sp / "labels"
            if not idir.exists():
                continue
            for img in sorted(p for p in idir.iterdir() if p.suffix.lower() in IMG_EXT):
                boxes, orig = parse_label(ldir / (img.stem + ".txt"), cmap)
                if orig > 0 and not boxes:
                    dropped_img += 1      # had labels, but none usable -> ambiguous, skip
                    continue
                items.append(dict(img=img, ds=d.name, boxes=boxes))
                n_img += 1
                for b in boxes:
                    cnt[TARGET[b[0]]] += 1
        print(f"    images kept: {n_img} (skipped {dropped_img} with only unusable labels)")
        print("    labels:", {k: cnt.get(k, 0) for k in TARGET})
        if cnt.get("motorcyclist", 0) == 0:
            print("    WARNING: no rider labels -> riders in this dataset are unlabeled. "
                  f"Consider: --skip {d.name}")
        print()

    if not items:
        sys.exit("Nothing to merge.")

    # ---------- 2. near-duplicate groups ----------
    print("Hashing images for duplicate detection ...")
    ok = []
    for it in items:
        h = dhash(it["img"])
        if h is not None:
            it["h"] = h
            ok.append(it)
    items = ok
    n = len(items)
    H = np.array([it["h"] for it in items], dtype=np.uint64)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n - 1):
        x = np.ascontiguousarray(H[i + 1:] ^ H[i])
        dist = np.unpackbits(x.view(np.uint8)).reshape(-1, 64).sum(1)
        for j in np.nonzero(dist <= a.thr)[0]:
            ra, rb = find(i), find(i + 1 + int(j))
            if ra != rb:
                parent[rb] = ra
    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(items[i])
    kept_groups = []
    for g in groups.values():
        g.sort(key=lambda it: -len(it["boxes"]))
        kept_groups.append(g[: a.cap])
    n_kept = sum(len(g) for g in kept_groups)
    print(f"images: {n} -> {n_kept} after near-duplicate cap ({len(kept_groups)} groups)")

    # ---------- 3. split by group ----------
    rnd = random.Random(a.seed)
    rnd.shuffle(kept_groups)
    target = {"train": 0.80, "valid": 0.10, "test": 0.10}
    splits = {k: [] for k in target}
    for g in kept_groups:
        deficit = {k: target[k] * n_kept - len(splits[k]) for k in target}
        splits[max(deficit, key=deficit.get)].extend(g)

    # negatives
    negs = []
    if a.negatives:
        nd = ROOT / a.negatives
        negs = sorted(p for p in nd.iterdir() if p.suffix.lower() in IMG_EXT) if nd.exists() else []
        print(f"hard negatives: {len(negs)} images (added to train, no labels)")

    print("\nFINAL SPLITS")
    for k, v in splits.items():
        c = Counter(TARGET[b[0]] for it in v for b in it["boxes"])
        extra_n = len(negs) if k == "train" else 0
        print(f"  {k:6s}: {len(v) + extra_n:5d} images  labels={ {t: c.get(t, 0) for t in TARGET} }")
    per_ds = Counter(it["ds"] for v in splits.values() for it in v)
    print("  per dataset after cleaning:", dict(per_ds))

    if not a.go:
        print("\nDRY RUN only - nothing written. Re-run with --go to build", a.out)
        return

    # ---------- 4. write ----------
    if out.exists():
        if not a.force:
            sys.exit(f"{out} already exists. Delete it or use --force.")
        shutil.rmtree(out)
    for k, v in splits.items():
        (out / k / "images").mkdir(parents=True)
        (out / k / "labels").mkdir(parents=True)
        for it in v:
            stem = f"{it['ds']}_{it['img'].stem}"
            shutil.copy2(it["img"], out / k / "images" / (stem + it["img"].suffix.lower()))
            lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in it["boxes"]]
            (out / k / "labels" / (stem + ".txt")).write_text("\n".join(lines))
    for p in negs:
        stem = f"neg_{p.stem}"
        shutil.copy2(p, out / "train" / "images" / (stem + p.suffix.lower()))
        (out / "train" / "labels" / (stem + ".txt")).write_text("")
    (out / "data.yaml").write_text(yaml.safe_dump(
        {"path": ".", "train": "train/images", "val": "valid/images", "test": "test/images",
         "nc": 3, "names": TARGET}, sort_keys=False))
    (out / "report.json").write_text(json.dumps(
        {"datasets": [d.name for d in datasets], "images": {k: len(v) for k, v in splits.items()},
         "near_dup_cap": a.cap}, indent=2))
    print(f"\nDONE -> {out}\nTrain with:  python main.py all --data {a.out} --model yolov8m.pt --imgsz 960")


if __name__ == "__main__":
    main()
