from ultralytics import YOLO

model = YOLO("best.pt")
results = model.predict("test_image_1.jpg", show=True, save=True, conf=0.4)