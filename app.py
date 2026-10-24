"""
SafeRide AI - Streamlit website
Run:  streamlit run app.py
Needs best.pt in the same folder as this file (or in a 'models' folder).
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
WEIGHT_CANDIDATES = [ROOT / "best.pt", ROOT / "models" / "best.pt"]

# Class ids must match training: 0 helmet, 1 license_plate, 2 motorcyclist
GREEN, RED, YELLOW, ORANGE = (0, 200, 0), (0, 0, 255), (0, 215, 255), (0, 165, 255)  # BGR

IMGSZ = 640              # model was trained at 640; larger sizes made detection worse
VIDEO_IMGSZ = 960
MIN_RIDER_W = 40         # ignore tiny rider boxes (pixels)
MIN_RIDER_H = 60
MIN_VIOL_FRAMES = 8      # video: frames a rider must lack a helmet before it counts
DARK_LEVEL = 90          # average brightness (0-255) below this = night image

st.set_page_config(page_title="SafeRide AI", page_icon="🏍️", layout="wide")


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    for p in WEIGHT_CANDIDATES:
        if p.exists():
            return YOLO(str(p))
    return None


# --------------------------------------------------------------------------- #
# Night enhancement
# --------------------------------------------------------------------------- #
def is_dark(bgr):
    return float(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).mean()) < DARK_LEVEL


def brighten(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


def prepare(bgr):
    """Brighten dark (night) images before detection; daytime images unchanged."""
    return brighten(bgr) if is_dark(bgr) else bgr


# --------------------------------------------------------------------------- #
# Detection parsing + violation logic
# --------------------------------------------------------------------------- #
def parse_dets(result, thresholds):
    """Return (cls, conf, x1, y1, x2, y2, track_id) using per-class thresholds."""
    dets = []
    for b in result.boxes:
        c = int(b.cls[0])
        p = float(b.conf[0])
        if p < thresholds.get(c, 0.4):
            continue
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0]]
        tid = int(b.id[0]) if getattr(b, "id", None) is not None else None
        dets.append((c, p, x1, y1, x2, y2, tid))
    return dets


def valid_rider(d, min_aspect):
    """Filters out pedestrians (tall + narrow) and tiny boxes."""
    w, h = d[4] - d[2], d[5] - d[3]
    if w < MIN_RIDER_W or h < MIN_RIDER_H:
        return False
    return (w / h) >= min_aspect


def center_in_box(d, r, mx=0.0, my=0.0):
    cx, cy = (d[2] + d[4]) / 2, (d[3] + d[5]) / 2
    bw, bh = r[4] - r[2], r[5] - r[3]
    return (r[2] - mx * bw <= cx <= r[4] + mx * bw and
            r[3] - my * bh <= cy <= r[5] + my * bh)


def in_head_zone(d, r):
    """Helmet centre must be inside the rider box, in the upper 60% (head area)."""
    cx, cy = (d[2] + d[4]) / 2, (d[3] + d[5]) / 2
    return (r[2] <= cx <= r[4] and
            r[3] <= cy <= r[3] + 0.6 * (r[5] - r[3]))


def rider_has_helmet(rider, helmets):
    """Helmet centre must lie inside the rider box, upper 60%."""
    rx1, ry1, rx2, ry2 = rider[2], rider[3], rider[4], rider[5]
    head_limit = ry1 + 0.6 * (ry2 - ry1)
    for h in helmets:
        cx, cy = (h[2] + h[4]) / 2, (h[3] + h[5]) / 2
        if rx1 <= cx <= rx2 and ry1 <= cy <= head_limit:
            return True
    return False


def annotate(img, dets, min_aspect, state=None):
    """
    state=None  -> single image: violation is immediate.
    state=dict  -> video: a rider must lack a helmet for MIN_VIOL_FRAMES frames.
    """
    all_riders = [d for d in dets if d[0] == 2]
    riders = [r for r in all_riders if valid_rider(r, min_aspect)]
    # keep only helmets in the head zone of a rider (drops headlights etc.)
    helmets = [d for d in dets if d[0] == 0 and any(in_head_zone(d, r) for r in riders)]
    plates = [d for d in dets if d[0] == 1 and any(center_in_box(d, r, 0.15, 0.15) for r in riders)]
    violations = 0

    def box(d, color, label):
        conf, x1, y1, x2, y2 = d[1], d[2], d[3], d[4], d[5]
        p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
        cv2.rectangle(img, p1, p2, color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (p1[0], p1[1] - th - 6), (p1[0] + tw + 4, p1[1]), color, -1)
        cv2.putText(img, text, (p1[0] + 2, p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    for r in riders:
        has_helmet = rider_has_helmet(r, helmets)
        tid = r[6]

        if state is None or tid is None:
            confirmed, pending = (not has_helmet), False
        else:
            if has_helmet:
                state["seen"].add(tid)
            elif tid in state["seen"]:
                has_helmet = True   # seen with helmet earlier, keep it
            cnt = state["count"].get(tid, 0)
            cnt = cnt + 1 if not has_helmet else max(0, cnt - 1)
            state["count"][tid] = cnt
            confirmed = cnt >= MIN_VIOL_FRAMES
            pending = (not has_helmet) and not confirmed
            if confirmed:
                state["confirmed"].add(tid)

        if confirmed:
            violations += 1
            box(r, RED, "NO HELMET")
        elif pending:
            box(r, ORANGE, "checking...")
        else:
            box(r, GREEN, "Rider: helmet OK")

    for h in helmets:
        box(h, GREEN, "helmet")
    for p in plates:
        box(p, YELLOW, "plate")

    return img, {"riders": len(riders), "violations": violations, "plates": len(plates)}


# --------------------------------------------------------------------------- #
# Plate second pass (zoom into each rider to find small / night plates)
# --------------------------------------------------------------------------- #
def plate_second_pass(model, work, dets, plate_conf):
    """Re-scan each rider crop to find plates the full-image pass missed."""
    H, W = work.shape[:2]
    extra = []
    for r in [d for d in dets if d[0] == 2]:
        x1, y1 = int(max(r[2], 0)), int(max(r[3], 0))
        x2, y2 = int(min(r[4], W)), int(min(r[5], H))
        if x2 - x1 < 40 or y2 - y1 < 60:
            continue
        crop = work[y1:y2, x1:x2]
        res = model.predict(crop, conf=0.05, imgsz=IMGSZ, verbose=False)[0]
        for b in res.boxes:
            if int(b.cls[0]) != 1:
                continue
            p = float(b.conf[0])
            if p < plate_conf:
                continue
            bx1, by1, bx2, by2 = [float(v) for v in b.xyxy[0]]
            extra.append((1, p, bx1 + x1, by1 + y1, bx2 + x1, by2 + y1, None))
    return extra


def merge_plates(dets, extra):
    plates = [d for d in dets if d[0] == 1]
    for e in extra:
        cx, cy = (e[2] + e[4]) / 2, (e[3] + e[5]) / 2
        if not any(p[2] <= cx <= p[4] and p[3] <= cy <= p[5] for p in plates):
            dets.append(e)
    return dets


# --------------------------------------------------------------------------- #
# Detection entry points
# --------------------------------------------------------------------------- #
def detect_image(model, bgr, th, min_aspect):
    work = prepare(bgr)                       # brightened if night image
    res = model.predict(work, conf=0.05, imgsz=IMGSZ, verbose=False)[0]
    dets = parse_dets(res, th)
    dets = merge_plates(dets, plate_second_pass(model, work, dets, th[1]))
    return annotate(bgr.copy(), dets, min_aspect, state=None)   # draw on the original


def detect_video_frame(model, bgr, th, min_aspect, state, first):
    work = prepare(bgr)
    res = model.track(work, conf=0.05, imgsz=VIDEO_IMGSZ, persist=not first,
                      tracker="bytetrack.yaml", verbose=False)[0]
    return annotate(bgr.copy(), parse_dets(res, th), min_aspect, state=state)


def show_stats(stats):
    c1, c2, c3 = st.columns(3)
    c1.metric("Riders", stats["riders"])
    c2.metric("No-helmet violations", stats["violations"])
    c3.metric("Plates detected", stats["plates"])
    if stats["violations"] > 0:
        st.error(f"🚨 {stats['violations']} helmet violation(s) detected")
    elif stats["riders"] > 0:
        st.success("✅ All riders are wearing helmets")
    else:
        st.info("No motorcyclist detected")


# --------------------------------------------------------------------------- #
# Image / video handlers
# --------------------------------------------------------------------------- #
def handle_image(model, pil_img, th, min_aspect):
    bgr = cv2.cvtColor(np.array(pil_img.convert("RGB")), cv2.COLOR_RGB2BGR)
    out, stats = detect_image(model, bgr, th, min_aspect)
    left, right = st.columns(2)
    left.subheader("Original")
    left.image(pil_img, use_container_width=True)
    right.subheader("Detection")
    right.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
    show_stats(stats)

    ok, buf = cv2.imencode(".jpg", out)
    if ok:
        st.download_button("⬇️ Download result image", buf.tobytes(),
                           "saferide_result.jpg", "image/jpeg")


def handle_video(model, uploaded, th, min_aspect):
    tmp_dir = Path(tempfile.mkdtemp())
    src = tmp_dir / "input.mp4"
    src.write_bytes(uploaded.read())

    cap = cv2.VideoCapture(str(src))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    w, h = int(cap.get(3)), int(cap.get(4))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    raw_out = tmp_dir / "raw.mp4"
    writer = cv2.VideoWriter(str(raw_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    state = {"count": {}, "confirmed": set(), "seen": set()}
    bar = st.progress(0.0, text="Processing video...")
    n = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        out, _ = detect_video_frame(model, frame, th, min_aspect, state, first=(n == 0))
        writer.write(out)
        n += 1
        if n % 5 == 0:
            bar.progress(min(n / total, 1.0), text=f"Processing frame {n}/{total}")
    cap.release()
    writer.release()
    bar.empty()

        # Browsers can't play mp4v; convert to H.264.
    final = raw_out
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        ffmpeg_exe = shutil.which("ffmpeg")
    if ffmpeg_exe:
        h264 = tmp_dir / "result.mp4"
        r = subprocess.run([ffmpeg_exe, "-y", "-i", str(raw_out), "-vcodec", "libx264",
                            "-pix_fmt", "yuv420p", str(h264)], capture_output=True)
        if r.returncode == 0 and h264.exists():
            final = h264

    st.success(f"Done: {n} frames. Unique no-helmet riders confirmed: {len(state['confirmed'])}")
    data = final.read_bytes()
    st.video(data)
    st.download_button("⬇️ Download result video", data, "saferide_result.mp4", "video/mp4")


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
st.title("🏍️ SafeRide AI")
st.caption("Helmet violation & number plate detection using YOLOv8")

model = load_model()
if model is None:
    st.error("best.pt not found. Put it next to app.py (or in a 'models' folder) and refresh.")
    st.stop()

with st.sidebar:
    st.header("Settings")
    rider_conf = st.slider("Motorcyclist confidence", 0.20, 0.95, 0.50, 0.05)
    helmet_conf = st.slider("Helmet confidence", 0.10, 0.95, 0.35, 0.05)
    plate_conf = st.slider("Plate confidence", 0.05, 0.95, 0.30, 0.05)
    min_aspect = st.slider("Min rider width/height (filters pedestrians)", 0.30, 1.00, 0.55, 0.05)
    mode = st.radio("Input", ["Upload image", "Upload video", "Camera snapshot"])
    st.markdown("---")
    st.markdown("**Colors**  \n🟢 helmet OK  \n🔴 NO HELMET  \n🟠 checking (video)  \n🟡 number plate")

thresholds = {0: helmet_conf, 1: plate_conf, 2: rider_conf}

if mode == "Upload image":
    f = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
    if f:
        handle_image(model, Image.open(f), thresholds, min_aspect)
elif mode == "Upload video":
    f = st.file_uploader("Choose a video", type=["mp4", "avi", "mov", "mkv"])
    if f and st.button("Run detection", type="primary"):
        handle_video(model, f, thresholds, min_aspect)
else:
    snap = st.camera_input("Take a photo")
    if snap:
        handle_image(model, Image.open(snap), thresholds, min_aspect)