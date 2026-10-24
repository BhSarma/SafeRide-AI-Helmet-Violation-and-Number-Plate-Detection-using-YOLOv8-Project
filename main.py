"""
SafeRide AI - Helmet Violation & Number Plate Detection (YOLOv8)

Folder layout expected (next to this file):
    helmet_plate/   (train/ valid/ test/ each with images/ + labels/, data.yaml)
    test_image_1.jpg, test_image_2.jpg, test_video.mp4   (optional samples)

Usage:
    python main.py all                         # verify -> train (hard 5h cap) -> test eval -> sample predictions
    python main.py train --model yolov8m.pt    # train only
    python main.py train --resume              # continue an interrupted run
    python main.py eval                        # metrics on the TEST split
    python main.py predict --source test_image_1.jpg
    python main.py predict --source test_video.mp4
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import cv2
import yaml

ROOT = Path(__file__).resolve().parent
DATASET_DIR = ROOT / "helmet_plate"
RUNTIME_YAML = ROOT / "data_runtime.yaml"
RUNS_DIR = ROOT / "runs"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
BEST_WEIGHTS = MODELS_DIR / "best.pt"

CLASS_NAMES = ["helmet", "license_plate", "motorcyclist"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VID_EXT = {".mp4", ".avi", ".mov", ".mkv"}

GREEN, RED, YELLOW = (0, 200, 0), (0, 0, 255), (0, 215, 255)


# --------------------------------------------------------------------------- #
# Data preparation
# --------------------------------------------------------------------------- #
def verify_dataset():
    """Check the YOLO folder structure and write a path-safe data yaml."""
    if not DATASET_DIR.exists():
        sys.exit(f"[ERROR] Dataset folder not found: {DATASET_DIR}\n"
                 f"Put the 'helmet_plate' folder (train/valid/test) next to main.py.")
    print("=== Dataset check ===")
    for split in ("train", "valid", "test"):
        img_dir, lbl_dir = DATASET_DIR / split / "images", DATASET_DIR / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            sys.exit(f"[ERROR] Missing {img_dir} or {lbl_dir}")
        imgs = [p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT]
        lbls = {p.stem for p in lbl_dir.glob("*.txt")}
        missing = sum(1 for p in imgs if p.stem not in lbls)
        print(f"{split:6s}: {len(imgs):5d} images, {len(lbls):5d} labels, {missing} images without label")
        if not imgs:
            sys.exit(f"[ERROR] No images in {img_dir}")

    cfg = {
        "path": str(DATASET_DIR),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": len(CLASS_NAMES),
        "names": CLASS_NAMES,
    }
    RUNTIME_YAML.write_text(yaml.safe_dump(cfg, sort_keys=False))
    print(f"Wrote {RUNTIME_YAML.name}\n")


def get_device():
    import torch
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
        return 0
    print("[WARNING] No CUDA GPU detected - training on CPU will be extremely slow.")
    return "cpu"


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train(args):
    from ultralytics import YOLO
    verify_dataset()
    device = get_device()

    if args.resume:
        ckpts = sorted(RUNS_DIR.glob("*/weights/last.pt"), key=lambda p: p.stat().st_mtime)
        if not ckpts:
            sys.exit("[ERROR] Nothing to resume (no runs/*/weights/last.pt).")
        print(f"Resuming from {ckpts[-1]}")
        model = YOLO(str(ckpts[-1]))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=str(RUNTIME_YAML),
            time=args.hours,          # HARD wall-clock cap in hours (overrides epochs, LR schedule fits the time)
            epochs=300,               # upper bound only; 'time' decides
            imgsz=args.imgsz,
            batch=args.batch,         # -1 = auto-fit to GPU memory
            device=device,
            workers=args.workers,
            cache=args.cache,
            project=str(RUNS_DIR),
            name="saferide",
            patience=args.patience,
            hsv_v=0.6,                # stronger brightness variation (night robustness)
            cos_lr=True,
            close_mosaic=15,
            seed=0,
            plots=True,
            verbose=True,
        )

    best = Path(model.trainer.best)
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy2(best, BEST_WEIGHTS)
    print(f"\nBest weights saved to: {BEST_WEIGHTS}")
    return BEST_WEIGHTS


# --------------------------------------------------------------------------- #
# Evaluation (TEST split)
# --------------------------------------------------------------------------- #
def evaluate(weights, imgsz=640):
    from ultralytics import YOLO
    verify_dataset()
    model = YOLO(str(weights))
    m = model.val(
        data=str(RUNTIME_YAML),
        split="test",
        imgsz=imgsz,
        batch=8,
        device=get_device(),
        workers=0,
        plots=True,
        project=str(RESULTS_DIR),
        name="test_eval",
        exist_ok=True,
    )
    overall = {
        "mAP@50": float(m.box.map50),
        "mAP@50-95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
    }
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        per_class[m.names[int(c)]] = {
            "precision": float(m.box.p[i]),
            "recall": float(m.box.r[i]),
            "mAP@50": float(m.box.ap50[i]),
            "mAP@50-95": float(m.box.ap[i]),
        }
    print("\n=== TEST SET RESULTS ===")
    for k, v in overall.items():
        print(f"{k:10s}: {v:.4f}")
    for name, d in per_class.items():
        print(f"  {name:14s} P={d['precision']:.3f} R={d['recall']:.3f} "
              f"mAP50={d['mAP@50']:.3f} mAP50-95={d['mAP@50-95']:.3f}")
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "test_metrics.json").write_text(
        json.dumps({"overall": overall, "per_class": per_class}, indent=2))
    print(f"Saved {RESULTS_DIR / 'test_metrics.json'}")
    return overall


# --------------------------------------------------------------------------- #
# Violation logic + drawing
# --------------------------------------------------------------------------- #
def parse_dets(result):
    dets = []
    for b in result.boxes:
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        dets.append((int(b.cls[0]), float(b.conf[0]), x1, y1, x2, y2))
    return dets


def rider_has_helmet(rider, helmets):
    """A rider is helmeted if a helmet centre lies inside the rider box, upper half."""
    _, _, rx1, ry1, rx2, ry2 = rider
    head_limit = ry1 + 0.5 * (ry2 - ry1)
    for _, _, hx1, hy1, hx2, hy2 in helmets:
        cx, cy = (hx1 + hx2) / 2, (hy1 + hy2) / 2
        if rx1 <= cx <= rx2 and ry1 <= cy <= head_limit:
            return True
    return False


def annotate(img, dets):
    helmets = [d for d in dets if d[0] == 0]
    plates = [d for d in dets if d[0] == 1]
    riders = [d for d in dets if d[0] == 2]
    violations = 0

    def box(d, color, label):
        _, conf, x1, y1, x2, y2 = d
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(img, text, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for r in riders:
        if rider_has_helmet(r, helmets):
            box(r, GREEN, "Rider: helmet OK")
        else:
            violations += 1
            box(r, RED, "NO HELMET")
    for h in helmets:
        box(h, GREEN, "helmet")
    for p in plates:
        box(p, YELLOW, "plate")

    stats = {"riders": len(riders), "violations": violations, "plates": len(plates)}
    banner = f"Riders: {stats['riders']}  No-helmet: {violations}  Plates: {stats['plates']}"
    cv2.putText(img, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, banner, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    return img, stats


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def predict_image(model, path, conf, out_dir):
    img = cv2.imread(str(path))
    if img is None:
        print(f"[WARN] Cannot read {path}")
        return None
    res = model.predict(img, conf=conf, imgsz=640, verbose=False)[0]
    out, stats = annotate(img, parse_dets(res))
    out_path = out_dir / f"{Path(path).stem}_detected.jpg"
    cv2.imwrite(str(out_path), out)
    print(f"{Path(path).name}: {stats} -> {out_path}")
    return stats


def predict_video(model, path, conf, out_dir):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out_path = out_dir / f"{Path(path).stem}_detected.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n, worst = 0, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        res = model.predict(frame, conf=conf, imgsz=640, verbose=False)[0]
        out, stats = annotate(frame, parse_dets(res))
        worst = max(worst, stats["violations"])
        writer.write(out)
        n += 1
        if n % 100 == 0:
            print(f"  {n}/{total} frames")
    cap.release()
    writer.release()
    print(f"{Path(path).name}: {n} frames, max simultaneous no-helmet riders={worst} -> {out_path}")


def predict(weights, source, conf):
    from ultralytics import YOLO
    model = YOLO(str(weights))
    out_dir = RESULTS_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    src = Path(source)
    if src.is_dir():
        for p in sorted(src.iterdir()):
            if p.suffix.lower() in IMG_EXT:
                predict_image(model, p, conf, out_dir)
    elif src.suffix.lower() in VID_EXT:
        predict_video(model, src, conf, out_dir)
    elif src.suffix.lower() in IMG_EXT:
        predict_image(model, src, conf, out_dir)
    else:
        sys.exit(f"[ERROR] Unsupported source: {source}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    global DATASET_DIR
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", default="yolov8s.pt", help="yolov8s.pt (default) or yolov8m.pt if GPU >= 8GB")
    common.add_argument("--hours", type=float, default=5.0, help="hard training time cap in hours")
    common.add_argument("--imgsz", type=int, default=640)
    common.add_argument("--batch", type=int, default=-1, help="-1 = auto")
    common.add_argument("--workers", type=int, default=4, help="set 0 if Windows dataloader errors")
    common.add_argument("--cache", default=False, help="'ram' to cache images (faster, ~2GB RAM)")
    common.add_argument("--resume", action="store_true")
    common.add_argument("--data", default=None, help="dataset folder (default: helmet_plate), e.g. merged_data")
    common.add_argument("--patience", type=int, default=60, help="early-stopping patience (epochs)")
    common.add_argument("--weights", default=str(BEST_WEIGHTS))
    common.add_argument("--conf", type=float, default=0.25)

    ap = argparse.ArgumentParser(description="SafeRide AI")
    sub = ap.add_subparsers(dest="cmd")
    sub.add_parser("all", parents=[common])
    sub.add_parser("train", parents=[common])
    sub.add_parser("eval", parents=[common])
    p = sub.add_parser("predict", parents=[common])
    p.add_argument("--source", required=True)
    args = ap.parse_args(["all"] if len(sys.argv) == 1 else None)
    if getattr(args, "data", None):
        DATASET_DIR = (ROOT / args.data).resolve()

    if args.cmd == "all":
        best = train(args)
        evaluate(best, args.imgsz)
        for s in ("test_image_1.jpg", "test_image_2.jpg"):
            if (ROOT / s).exists():
                predict(best, ROOT / s, args.conf)
    elif args.cmd == "train":
        train(args)
    elif args.cmd == "eval":
        evaluate(args.weights, args.imgsz)
    elif args.cmd == "predict":
        predict(args.weights, args.source, args.conf)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()