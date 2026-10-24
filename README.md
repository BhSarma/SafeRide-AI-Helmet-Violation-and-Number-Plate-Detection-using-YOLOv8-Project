# 🏍️ SafeRide AI: Helmet Violation & Number Plate Detection using YOLOv8

SafeRide AI detects motorcycle riders, helmets and number plates in road images and videos. A rider without a helmet is flagged as a violation, and the number plate of that rider is detected so the vehicle can be identified. A **Streamlit** website lets anyone upload an image or video, or take a camera snapshot, and see the results.

The model is a **YOLOv8m** detector trained on a merged dataset of about 2,800 labelled images. It works best on **rear-view footage**, where the camera is behind the vehicles and the number plates are visible.

---

## Features

- Detects 3 classes: `helmet`, `license_plate`, `motorcyclist`
- Flags riders without a helmet (**NO HELMET**) and shows the number plate of the violating rider
- Works on **images**, **videos** and **camera snapshots**
- Separate confidence sliders for riders, helmets and plates
- Optional bicycle filter (uses a small extra YOLOv8n model) to reduce false alarms
- Re-scan step that zooms in on a violating rider to find a missed plate
- Video mode confirms a violation over several frames, which reduces one-frame false alarms
- Plate evidence: the clearest plate crop of each violating rider
- Download buttons for the annotated image and video

Colour code: 🟢 helmet OK, 🔴 NO HELMET, 🟡 number plate, ⚪ still checking (video).

---

## Results

Evaluated on a held-out **test set of 282 images (1,380 labelled objects)** that the model never saw during training.

| Class | Precision | Recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| **All** | **0.986** | **0.973** | **0.986** | **0.818** |
| helmet | 0.983 | 0.953 | 0.973 | 0.772 |
| license_plate | 0.976 | 0.978 | 0.990 | 0.763 |
| motorcyclist | 0.998 | 0.988 | 0.995 | 0.918 |

An earlier model, trained on a smaller single-source dataset, reached about 0.954 mAP50 and 0.679 mAP50-95 on its own validation set. It missed cases from other sources, so more varied data was merged and a larger model was trained. The two scores come from different data, so they are not a strict like-for-like comparison.

Sample outputs are in the [`sample_results/`](sample_results/) folder.

---

## Training summary

| Item | Value |
|---|---|
| Model | YOLOv8m (pretrained `yolov8m.pt`) |
| Image size | 960 |
| Training time | 7 hours (190 epochs, planned automatically to fit the time limit) |
| Hardware | Kaggle GPU (Tesla T4) |
| Data split | 2,256 train / 282 validation / 282 test images |
| Classes | 0 `helmet`, 1 `license_plate`, 2 `motorcyclist` |

The final dataset was built by cleaning and merging several public datasets with `prepare_data.py` (class mapping check, label cleaning and a train/valid/test split). The data itself is **not** included in this repository.

**Dataset sources:** `[add the names and links of your datasets here]`

---

## Project structure

```
.
├── app.py              # Streamlit website
├── best.pt             # trained model (final)
├── requirements.txt    # Python packages
├── prepare_data.py     # builds the merged dataset
├── main.py             # training / evaluation / prediction script
├── test.py             # quick test of the model on one image
├── test_samples/       # sample image and video for testing
├── sample_results/     # example detection results
└── README.md
```

---

## How to run

**1. Create and activate a virtual environment (first time only)**

Windows:
```
python -m venv .venv
.venv\Scripts\activate
```
Mac / Linux:
```
python3 -m venv .venv
source .venv/bin/activate
```

**2. Install the packages (first time only)**
```
pip install -r requirements.txt
```

**3. Start the website**
```
streamlit run app.py
```
The browser opens at `http://localhost:8501`. Choose **Upload image**, **Upload video** or **Camera snapshot** in the sidebar.

Keep `best.pt` in the same folder as `app.py`. The bicycle filter downloads a small model (`yolov8n.pt`) automatically the first time it is used, so an internet connection is needed once.

**Quick test without the website**
```
python test.py
```

---

## Limitations

1. **Camera view (main limitation):** The model is trained on rear-view images, so it works best with a camera behind the vehicles. On front-view video, helmet detection still works well, but number plates are usually not visible, because most Indian two-wheelers carry only a rear plate.
2. **No plate reading:** The system detects the number plate but does not read the plate number (no OCR).
3. **Rain:** Heavy-rain footage was not tested.

The scores above are measured on test images from the same sources as the training data, so results on completely new cameras may differ.

---

## Future work

- Add front-view to the training data
- Read the plate number with OCR
- Add full object tracking so each rider is counted once in a video
- Count each person on a bike separately

---

## Tech stack

Python, Ultralytics YOLOv8, OpenCV, PyTorch, Streamlit, Kaggle (training).

## Author

BH.Sarma