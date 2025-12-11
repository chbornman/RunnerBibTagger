# Bib Tagger

Automatic bib number detection and EXIF metadata tagging for race photography.

Feed it a folder of race photos, and it will detect bib numbers and write them to each image's IPTC Keywords metadata as `BIB:1234`. This makes photos instantly searchable by bib number in tools like Lightroom, Photo Mechanic, or any DAM software.

## Features

- **GUI and CLI** - Desktop app or command-line interface
- **YOLO-based bib detection** - Fast, accurate detection of race bibs
- **RapidOCR digit recognition** - PP-OCRv4 models for reading bib numbers
- **Smart filtering** - 50% height threshold rejects secondary numbers (gear check, timing chips)
- **EXIF metadata tagging** - Writes bib numbers to IPTC Keywords via exiftool
- **Batch processing** - Process entire folders of images
- **Debug visualization** - Optional annotated output images showing detections
- **Lightweight** - ONNX-only inference (~250MB vs 2GB+ with PyTorch)

## Installation

### Prerequisites

- Python 3.10+
- [exiftool](https://exiftool.org/) - Required for writing metadata

```bash
# Install exiftool
# macOS:
brew install exiftool

# Ubuntu/Debian:
sudo apt install libimage-exiftool-perl

# Arch Linux:
sudo pacman -S perl-image-exiftool
```

### Setup

```bash
# Clone the repository
git clone https://github.com/chbornman/bib-tagger.git
cd bib-tagger

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### GUI Mode

```bash
python main.py
```

This launches the desktop application where you can:
1. Select an image folder
2. Choose options:
   - Generate CSV report
   - Show detection preview (live view as processing runs)
   - Save debug images to disk
   - Write bib numbers to image metadata (IPTC)
   - Include subfolders (recursive processing)
3. Click "Start Processing"
4. View results in the log and preview panel
5. Navigate through processed images with Previous/Next buttons or arrow keys

### CLI Mode

```bash
# Process a single image
python main.py photo.jpg

# Process an entire folder
python main.py ./race_photos/

# With options
python main.py ./race_photos/ --csv --debug --confidence 0.3
```

### Command Line Options

| Option | Default | Description |
|--------|---------|-------------|
| `--csv` | off | Generate `bib_results.csv` with all detections |
| `--debug` | off | Save annotated debug images to `debug_images/` |
| `--no-metadata` | off | Skip writing IPTC metadata to images |
| `--confidence` | 0.25 | Detection confidence threshold (0.0-1.0) |

### Examples

```bash
# Generate CSV report
python main.py ./photos/ --csv

# Save debug images to see what was detected
python main.py ./photos/ --debug

# Higher confidence threshold (fewer false positives)
python main.py --confidence 0.5 ./photos/

# Process without modifying original images
python main.py ./photos/ --no-metadata --csv
```

## Output

### Console Output
```
Processing 3 image(s)...

[1/3] DSC_0001.jpg -> 1234
[2/3] DSC_0002.jpg -> 567, 890
[3/3] DSC_0003.jpg -> (no bibs)

Processed 3 images, found 3 bib(s)
```

### CSV Output (`bib_results.csv`)
```csv
Filename,Bib Numbers,Detections,Time (ms)
DSC_0001.jpg,1234,1,245
DSC_0002.jpg,"567, 890",2,312
DSC_0003.jpg,,0,89
```

### Debug Images
When using `--debug`, annotated images are saved showing:
- Green bounding boxes around detected bibs
- Detection confidence percentage
- Recognized bib number and OCR confidence

## How It Works

### Pipeline

```
Image → YOLO Detection → RapidOCR → Height Filtering → Metadata Write
```

1. **Detection**: YOLO ONNX model locates bib regions in the image
2. **OCR**: RapidOCR (PP-OCRv4) extracts text from each detected region
3. **Digit Filtering**: Non-digit characters are removed
4. **Height Filtering**: Text regions smaller than 50% of the tallest are rejected
5. **Selection**: The largest remaining text region is selected as the bib number
6. **Metadata**: Bib numbers are written to IPTC Keywords using exiftool

### Why 50% Height Filtering?

Race bibs often contain multiple numbers:
- **Main bib number** (large, prominent)
- Gear check numbers (small)
- Timing chip codes (small)
- Wave/corral indicators (small)

The 50% height threshold keeps only text that is at least half as tall as the largest detected text, reliably filtering out secondary numbers.

## Metadata Format

Bib numbers are written to IPTC Keywords:
```
BIB:1234
```

This format:
- Is searchable in Lightroom, Photo Mechanic, and most DAM software
- Doesn't conflict with other keywords
- Allows searches like `BIB:1234` or `BIB:*`

To verify metadata:
```bash
exiftool -Keywords photo.jpg
```

## Supported Formats

- JPEG (`.jpg`, `.jpeg`)
- PNG (`.png`)
- TIFF (`.tif`, `.tiff`)

Files with `_debug` in the filename are automatically skipped.

## Dependencies

- [onnxruntime](https://onnxruntime.ai/) - ONNX model inference
- [rapidocr](https://github.com/RapidAI/RapidOCR) - OCR with PP-OCRv4 models
- [opencv-python](https://opencv.org/) - Image processing
- [exiftool](https://exiftool.org/) - Metadata writing (system dependency)

## Troubleshooting

### "No bibs detected"
- Try lowering confidence: `--confidence 0.15`
- Check if bibs are clearly visible in frame
- Use `--debug` to see what's being detected

### Wrong numbers detected
- Try increasing confidence: `--confidence 0.5`
- The 50% height filter should reject most secondary numbers

### "Failed to write metadata"
- Ensure exiftool is installed: `exiftool -ver`
- Check file permissions
- Verify image format supports IPTC metadata

## Testing

Run the test suite:

```bash
# Install test dependencies
pip install pytest

# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_gui_options.py -v
```

Test images are included in `tests/fixtures/images/` (resized to ~1600px for faster testing).

To regenerate test fixtures from full-resolution images:
```bash
./scripts/prepare_test_images.sh
```

## License

MIT License - See [LICENSE](LICENSE) for details.

## Acknowledgments

- OCR powered by [RapidOCR](https://github.com/RapidAI/RapidOCR) with [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR) models
- YOLO architecture by [Ultralytics](https://ultralytics.com/)

### Training Data Attribution

The included bib detection model was trained on combined public datasets from Roboflow Universe:

| Dataset | Author | License |
|---------|--------|---------|
| [Bib Number Labeling](https://universe.roboflow.com/marco-cheung/bib-number-labeling) | Marco Cheung | CC BY 4.0 |
| [Bib Number Detection](https://universe.roboflow.com/ai-vsumn/bib_number_detection) | AI-VSUMN | Roboflow |
| [Bib Number](https://universe.roboflow.com/bibnumberdetection/bib-number-x7gbv) | BibNumberDetection | Roboflow |

Marco Cheung's dataset aggregates images from multiple contributors:
- thomas-lamalle/bib-detection
- rbnr/bib-detector
- sputtipa/bip
- bibnumber/bibnumber
- python-vertiefung/python-vertiefung
- hcmus-3p8wh/bib-detection-big-data
- h1-qtgu0/bib-number

Thank you to all the photographers and annotators who made their data publicly available.
