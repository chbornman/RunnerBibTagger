# Bib Tagger - Technical Learnings

## Architecture Decision: ONNX-Only Stack

We chose ONNX Runtime over PyTorch/PaddlePaddle for distribution size:

| Stack | Size |
|-------|------|
| PyTorch + PaddlePaddle | ~2GB+ |
| **ONNX Runtime** | **~250MB** |

This makes the app practical for distribution as a standalone macOS `.app` or `.dmg`.

## OCR Model Selection

### Model Versions Tested

| Package | OCR Model | Year | Accuracy |
|---------|-----------|------|----------|
| `rapidocr-onnxruntime` 1.2.3 | PP-OCRv3 | 2022 | Baseline |
| **`rapidocr` 3.4.3** | **PP-OCRv4** | 2023 | **+5% bibs read** |
| PaddleOCR (native) | PP-OCRv5 | 2024 | +10% but 4x slower |

We use `rapidocr` 3.4.3 which bundles PP-OCRv4 ONNX models internally. This provides a good balance of accuracy and speed without external model files.

### RapidOCR 3.4.3 API

```python
from rapidocr import RapidOCR
ocr = RapidOCR()
result = ocr(image)
# result.boxes - list of polygon coordinates
# result.txts - tuple of recognized text strings
# result.scores - tuple of confidence scores
```

## Height Filtering for Secondary Numbers

Race bibs often have multiple numbers:
- **Main bib number** (large)
- Gear check numbers (small)
- Timing chip codes (small)
- Wave/corral indicators (small)

The 50% height threshold in `read_bib_number()` keeps only text regions at least half as tall as the largest detected text. This reliably filters secondary numbers without needing to train a separate classifier.

## Performance Benchmarks

On a test set of 782 race photos:

| Metric | Value |
|--------|-------|
| Average time per image | ~420ms |
| Detection rate | 92% of images with visible bibs |
| OCR accuracy | ~95% on clear bibs |

## Future Improvements

### PP-OCRv5 Conversion
PaddlePaddle models can be converted to ONNX for better accuracy:

```bash
pip install paddle2onnx
paddle2onnx --model_dir PP-OCRv5_det --save_file models/PP-OCRv5_det.onnx
```

This would provide PP-OCRv5 accuracy with ONNX speed, but requires maintaining custom model files.

### Training Data
The bib detection model was trained on public Roboflow datasets. See README.md for full attribution.
