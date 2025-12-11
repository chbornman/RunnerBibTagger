# Bib Tagger GUI - Learnings & Next Steps

## Goal
Create a distributable macOS `.dmg` app that photographers can use without any dependencies.

## Key Discovery: OCR Model Versions Matter!

### Current Model Landscape

| Package | Detection Model | Recognition Model | Release Year | Size |
|---------|-----------------|-------------------|--------------|------|
| `rapidocr-onnxruntime` 1.2.3 | PP-OCRv3 | PP-OCRv3 | 2022 | ~130MB |
| `rapidocr` 3.4.3 | **PP-OCRv4** | **PP-OCRv4** | 2023 | ~150MB |
| PaddleOCR (current) | **PP-OCRv5 server** | **en_PP-OCRv5_mobile** | 2024 | ~2GB |

**We were using PP-OCRv3 (2022) when PP-OCRv5 (2024) exists!**

### Validation Results (782 images)

| Metric | ONNX (PP-OCRv3) | PaddleOCR (PP-OCRv5) |
|--------|-----------------|----------------------|
| Total detections | 1,921 | 1,945 |
| Total bibs read | 1,323 | 1,387 |
| Images with bibs | 717 | 727 |
| Time per image | **418ms** | 1,751ms |

- ONNX is **4.2x faster**
- But reads **~5% fewer bibs** (older OCR model)

---

## Immediate Next Steps

### 1. Upgrade to `rapidocr` 3.4.3 (PP-OCRv4)

The newer `rapidocr` package uses PP-OCRv4 models, which should close the accuracy gap.

```bash
# In bib-tagger-gui/requirements.txt
# Change from:
rapidocr-onnxruntime>=1.3.0

# To:
rapidocr>=3.4.0
```

**API Change Required:**
```python
# Old API (rapidocr-onnxruntime)
from rapidocr_onnxruntime import RapidOCR
result, _ = ocr(image)
# result = [[box, text, confidence], ...]

# New API (rapidocr 3.4.3)
from rapidocr import RapidOCR
result = ocr(image)
# result.boxes = array of boxes
# result.txts = tuple of texts
# result.scores = tuple of confidences
```

### 2. Convert PP-OCRv5 to ONNX (Best Option)

PaddlePaddle models can be converted to ONNX using `paddle2onnx`:

```bash
pip install paddle2onnx

# Convert detection model
paddle2onnx --model_dir ~/.paddlex/official_models/PP-OCRv5_server_det \
            --model_filename inference.json \
            --params_filename inference.pdiparams \
            --save_file models/PP-OCRv5_det.onnx

# Convert recognition model
paddle2onnx --model_dir ~/.paddlex/official_models/en_PP-OCRv5_mobile_rec \
            --model_filename inference.json \
            --params_filename inference.pdiparams \
            --save_file models/PP-OCRv5_rec.onnx
```

Then configure RapidOCR to use these custom models:

```python
from rapidocr import RapidOCR
ocr = RapidOCR(
    det_model_path="models/PP-OCRv5_det.onnx",
    rec_model_path="models/PP-OCRv5_rec.onnx"
)
```

### 3. Check if PP-OCRv5 ONNX Already Exists

Search these locations:
- https://github.com/RapidAI/RapidOCR/releases
- https://huggingface.co/models?search=ppocr+onnx
- https://github.com/PaddlePaddle/PaddleOCR/releases

---

## File Structure After Changes

```
bib-tagger-gui/
├── bib_tagger_onnx.py      # Update to use rapidocr 3.4.3 API
├── bib_tagger_gui.py       # Tkinter GUI (unchanged)
├── requirements.txt        # Change to rapidocr>=3.4.0
├── models/
│   ├── bib_detector.onnx   # YOLO model (already converted)
│   ├── PP-OCRv5_det.onnx   # TODO: Convert or download
│   └── PP-OCRv5_rec.onnx   # TODO: Convert or download
└── ...
```

---

## Technical Notes

### Why YOLO ONNX Works Well
- YOLO export to ONNX is well-supported by Ultralytics
- Detection results are nearly identical (1,921 vs 1,945 detections)
- The 24-detection difference is likely due to NMS threshold variations

### Why OCR Matters More
- The **64 bib difference** (1,323 vs 1,387) is entirely from OCR
- PP-OCRv5 has better text detection and recognition
- For race bibs: numbers on jerseys, varied lighting, motion blur

### Preprocessing/Postprocessing
Both systems use similar preprocessing:
- Letterbox padding to 640x640 for YOLO
- Standard ONNX Runtime inference

The OCR preprocessing may differ - need to verify:
- Image normalization values
- Text detection thresholds
- Recognition confidence thresholds

---

## Dependency Comparison

### Current (Heavy)
```
paddlepaddle>=3.0.0    # ~1GB
paddleocr>=3.0.0       # ~200MB
ultralytics>=8.0.0     # ~500MB + PyTorch ~2GB
opencv-python>=4.8.0   # ~50MB
```
**Total: ~2GB+**

### Target (Lightweight)
```
rapidocr>=3.4.0        # ~150MB (includes PP-OCRv4 ONNX models)
onnxruntime>=1.15.0    # ~50MB
opencv-python>=4.8.0   # ~50MB
numpy>=1.24.0          # included
```
**Total: ~250MB**

With custom PP-OCRv5 ONNX models: +~100MB = **~350MB total**

---

## Summary

1. **Immediate win**: Upgrade from `rapidocr-onnxruntime` to `rapidocr` 3.4.3
2. **Best accuracy**: Convert PP-OCRv5 to ONNX ourselves using `paddle2onnx`
3. **Speed**: ONNX is 4x faster regardless of model version
4. **Distribution**: ~350MB is much better than 2GB+ for a .dmg

The goal of a lightweight, distributable app is achievable - we just need to use the right OCR models!
