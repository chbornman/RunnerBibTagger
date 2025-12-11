# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bib Tagger is a desktop application for race photographers that detects bib numbers in photos and writes them to IPTC metadata. The goal is a lightweight, distributable macOS `.dmg` app that doesn't require Python installation.

## Architecture

```
Image → YOLO ONNX (bib regions) → RapidOCR PP-OCRv4 (text) → exiftool (metadata)
```

### Key Files

- **`main.py`** - Single entry point for both GUI and CLI modes
- **`gui.py`** - Tkinter GUI application (`BibTaggerApp` class)
  - Two-column layout: controls/log on left, preview on right
  - Options: CSV, preview, save debug, metadata, subfolders
  - Image navigation with Previous/Next and arrow keys
- **`bib_tagger.py`** - Core inference engine (`BibTagger` class)
  - YOLO ONNX for bib detection
  - RapidOCR 3.4.3 for OCR (bundles PP-OCRv4 models internally)
  - 50% height threshold filtering to reject secondary numbers
- **`models/bib_detector.onnx`** - Custom-trained YOLO model (~36MB)
- **`tests/`** - Pytest test suite
  - `conftest.py` - Fixtures and test image paths
  - `test_bib_tagger.py` - Core detection tests
  - `test_cli.py` - CLI flag tests
  - `test_gui_options.py` - GUI option behavior tests
- **`tests/fixtures/images/`** - Resized test images (~5MB total)

### Why ONNX?

The project uses ONNX Runtime instead of PyTorch/PaddlePaddle to minimize distribution size:
- ONNX stack: ~250MB total
- PyTorch + PaddlePaddle: ~2GB+

RapidOCR 3.4.3 bundles PP-OCRv4 ONNX models internally at `site-packages/rapidocr/models/`.

## Common Commands

```bash
# Launch GUI
python main.py

# CLI - process folder
python main.py photos/ --csv --debug

# CLI - single image
python main.py race.jpg --confidence 0.3

# Run tests
python -m pytest tests/ -v
```

## Build for macOS

```bash
# Install py2app
pip install py2app

# Build .app bundle
python setup_macos.py py2app

# Output: dist/Bib Tagger.app
```

## Key Technical Details

### RapidOCR API (v3.4.3)

```python
from rapidocr import RapidOCR
ocr = RapidOCR()
result = ocr(image)
# result.boxes - list of polygon coordinates
# result.txts - list of recognized text strings
# result.scores - list of confidence scores
```

### Height Filtering Logic

Race bibs often have secondary numbers (gear check, timing chips). The 50% height threshold in `read_bib_number()` keeps only text regions at least half as tall as the largest detected text, reliably filtering secondary numbers.

### Metadata Format

Bib numbers are written as IPTC Keywords: `BIB:1234`

Requires `exiftool` system dependency.
