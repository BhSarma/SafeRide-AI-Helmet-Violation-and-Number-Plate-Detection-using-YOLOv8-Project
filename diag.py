import cv2
from ultralytics import YOLO

IMG = "night.jpg"   # change this if your image has a different name

model = YOLO("best.pt")
print("Classes:", model.names)
img = cv2.imread(IMG)


def brighten(bgr):
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


tests = {"original": img, "brightened": brighten(img)}
for name, im in tests.items():
    for sz in (640, 960, 1280):
        r = model.predict(im, conf=0.01, imgsz=sz, verbose=False)[0]
        plates = [round(float(b.conf[0]), 3) for b in r.boxes if int(b.cls[0]) == 1]
        print(name, sz, "plate confidences:", plates)