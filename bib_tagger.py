#!/usr/bin/env python3
"""
Bib Tagger - ONNX-based inference engine.

Uses ONNX Runtime for YOLO detection and RapidOCR for digit recognition.
No PyTorch or PaddlePaddle dependencies required.
"""

import os
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import onnxruntime as ort
from rapidocr import RapidOCR


class BibTagger:
    """Detects bib numbers in images using ONNX models."""

    def __init__(
        self,
        model_path: str,
        confidence: float = 0.25,
        box_padding: float = 0.0,
        debug: bool = False,
        progress_callback: Optional[Callable[[str], None]] = None,
    ):
        self.confidence = confidence
        self.box_padding = box_padding
        self.debug = debug
        self.progress_callback = progress_callback

        # Model input size (standard YOLO)
        self.input_size = 640

        self._log(f"Loading bib detector: {model_path}")

        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']
        )
        self.input_name = self.session.get_inputs()[0].name

        self._log("Initializing RapidOCR...")
        self.ocr = RapidOCR()

        self._log("Ready!")

    def _log(self, message: str):
        """Log message to callback if available."""
        if self.progress_callback:
            self.progress_callback(message)
        else:
            print(message)

    def _preprocess(self, image: np.ndarray) -> tuple[np.ndarray, float, tuple[int, int]]:
        """Preprocess image for YOLO inference."""
        h, w = image.shape[:2]

        # Calculate scale to fit in input_size while maintaining aspect ratio
        scale = min(self.input_size / w, self.input_size / h)
        new_w, new_h = int(w * scale), int(h * scale)

        # Resize image
        resized = cv2.resize(image, (new_w, new_h))

        # Create padded image (letterbox)
        padded = np.full((self.input_size, self.input_size, 3), 114, dtype=np.uint8)
        pad_x, pad_y = (self.input_size - new_w) // 2, (self.input_size - new_h) // 2
        padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized

        # Convert to float and normalize
        blob = padded.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)  # HWC to CHW
        blob = np.expand_dims(blob, 0)  # Add batch dimension

        return blob, scale, (pad_x, pad_y)

    def _postprocess(
        self,
        outputs: np.ndarray,
        scale: float,
        pad: tuple[int, int],
        orig_shape: tuple[int, int]
    ) -> list[dict]:
        """Postprocess YOLO outputs to get detections."""
        # YOLO output shape: (1, 5, 8400) where 5 = x, y, w, h, conf
        predictions = outputs[0].T  # Shape: (8400, 5)

        detections = []
        pad_x, pad_y = pad
        orig_h, orig_w = orig_shape

        for pred in predictions:
            confidence = pred[4]

            if confidence < self.confidence:
                continue

            # Get box coordinates (center x, center y, width, height)
            cx, cy, w, h = pred[:4]

            # Convert to corner coordinates
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2

            # Remove padding and scale back to original size
            x1 = (x1 - pad_x) / scale
            y1 = (y1 - pad_y) / scale
            x2 = (x2 - pad_x) / scale
            y2 = (y2 - pad_y) / scale

            # Clip to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            # Apply padding if configured
            if self.box_padding > 0:
                bw, bh = x2 - x1, y2 - y1
                pad_bx = int(bw * self.box_padding)
                pad_by = int(bh * self.box_padding)
                x1 = max(0, x1 - pad_bx)
                y1 = max(0, y1 - pad_by)
                x2 = min(orig_w, x2 + pad_bx)
                y2 = min(orig_h, y2 + pad_by)

            detections.append({
                'box': (int(x1), int(y1), int(x2), int(y2)),
                'confidence': float(confidence),
            })

        # Apply NMS (Non-Maximum Suppression)
        detections = self._nms(detections, iou_threshold=0.5)

        return detections

    def _nms(self, detections: list[dict], iou_threshold: float = 0.5) -> list[dict]:
        """Apply Non-Maximum Suppression."""
        if not detections:
            return []

        # Sort by confidence
        detections = sorted(detections, key=lambda x: x['confidence'], reverse=True)

        keep = []
        while detections:
            best = detections.pop(0)
            keep.append(best)

            detections = [
                d for d in detections
                if self._iou(best['box'], d['box']) < iou_threshold
            ]

        return keep

    def _iou(self, box1: tuple, box2: tuple) -> float:
        """Calculate Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])

        inter_area = max(0, x2 - x1) * max(0, y2 - y1)

        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union_area = box1_area + box2_area - inter_area

        return inter_area / union_area if union_area > 0 else 0

    def detect_bibs(self, image: np.ndarray) -> list[dict]:
        """Detect bib regions in an image."""
        orig_shape = image.shape[:2]

        # Preprocess
        blob, scale, pad = self._preprocess(image)

        # Run inference
        outputs = self.session.run(None, {self.input_name: blob})

        # Postprocess
        detections = self._postprocess(outputs[0], scale, pad, orig_shape)

        return detections

    def read_bib_number(self, image: np.ndarray, box: tuple) -> Optional[tuple[str, float]]:
        """Read the bib number from a detected region using OCR."""
        x1, y1, x2, y2 = box
        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        # Run RapidOCR (3.4.3 API - returns RapidOCROutput object)
        result = self.ocr(crop)

        if result is None or result.boxes is None or len(result.boxes) == 0:
            return None

        # Extract text regions with bounding boxes
        # RapidOCR 3.4.3 returns: result.boxes, result.txts, result.scores
        texts = []
        confidences = []
        boxes = []

        for i, (box_points, text, conf) in enumerate(zip(result.boxes, result.txts, result.scores)):
            # Filter to only digits
            digits_only = ''.join(c for c in str(text) if c.isdigit())

            if digits_only:
                texts.append(digits_only)
                confidences.append(float(conf))

                # Convert polygon to bounding box
                xs = [p[0] for p in box_points]
                ys = [p[1] for p in box_points]
                boxes.append([min(xs), min(ys), max(xs), max(ys)])

        if not texts:
            return None

        # Apply 50% height threshold filtering to reject secondary numbers
        if boxes:
            heights = [(b[3] - b[1]) for b in boxes]
            max_height = max(heights) if heights else 0

            if max_height > 0:
                # Keep only text regions with height >= 50% of max
                filtered = [
                    (texts[i], confidences[i], boxes[i])
                    for i in range(len(texts))
                    if (boxes[i][3] - boxes[i][1]) >= max_height * 0.5
                ]

                if filtered:
                    texts = [f[0] for f in filtered]
                    confidences = [f[1] for f in filtered]
                    boxes = [f[2] for f in filtered]

        # Return tallest text region (likely main bib number)
        # Height is more reliable than area since bib numbers are tall
        if boxes:
            heights = [(b[3] - b[1]) for b in boxes]
            max_idx = heights.index(max(heights))
            return texts[max_idx], confidences[max_idx]

        return texts[0], confidences[0] if confidences else 0.0

    def write_iptc_metadata(self, image_path: str, bib_numbers: list[str]) -> bool:
        """Write bib numbers to image IPTC metadata using exiftool."""
        if not bib_numbers:
            return True

        # Build IPTC keywords with BIB: prefix
        keywords = [f"BIB:{bib}" for bib in bib_numbers]

        # Build exiftool command
        cmd = ['exiftool', '-overwrite_original']
        for kw in keywords:
            cmd.extend(['-Keywords+=' + kw])
        cmd.append(image_path)

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            return result.returncode == 0
        except Exception:
            return False

    def process_image(self, image_path: str, write_metadata: bool = True) -> dict:
        """Process a single image: detect bibs, read numbers, optionally write metadata."""
        result = {
            'path': image_path,
            'filename': Path(image_path).name,
            'success': False,
            'bibs': [],
            'detections': 0,
            'time_ms': 0,
            'error': None,
        }

        start_time = time.time()

        # Load image
        image = cv2.imread(image_path)
        if image is None:
            result['error'] = 'Failed to load image'
            return result

        # Detect bibs
        detections = self.detect_bibs(image)
        result['detections'] = len(detections)

        # Read bib numbers from each detection
        for det in detections:
            ocr_result = self.read_bib_number(image, det['box'])
            if ocr_result:
                bib_number, confidence = ocr_result
                result['bibs'].append({
                    'number': bib_number,
                    'confidence': confidence,
                    'box': det['box'],
                    'detection_confidence': det['confidence'],
                })

        # Write metadata if bibs found and enabled
        if result['bibs'] and write_metadata:
            bib_numbers = [b['number'] for b in result['bibs']]
            if self.write_iptc_metadata(image_path, bib_numbers):
                result['success'] = True
            else:
                result['error'] = 'Failed to write metadata'
        else:
            result['success'] = True  # No bibs or metadata disabled is not an error

        result['time_ms'] = (time.time() - start_time) * 1000

        # Store image and detections for debug image generation
        result['_image'] = image
        result['_detections'] = detections

        return result

    def save_debug_image(self, result: dict, output_path: str):
        """Save annotated debug image."""
        image = result.get('_image')
        detections = result.get('_detections', [])
        bibs = result.get('bibs', [])

        if image is None:
            return

        debug_img = image.copy()

        if not detections:
            # No detections - draw "NO BIB FOUND" in center
            h, w = debug_img.shape[:2]
            text = "NO BIB FOUND"
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Scale font to fit ~50% of image width
            # Start with a test scale and adjust
            target_width = w * 0.5
            test_scale = 1.0
            (test_w, _), _ = cv2.getTextSize(text, font, test_scale, 1)
            font_scale = (target_width / test_w) * test_scale
            font_scale = max(0.5, min(font_scale, 8.0))  # Clamp between 0.5 and 8.0

            thickness = max(2, int(font_scale * 2))

            # Get text size to center it
            (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
            x = (w - text_w) // 2
            y = (h + text_h) // 2

            # Draw text with outline for visibility
            cv2.putText(debug_img, text, (x, y), font, font_scale, (0, 0, 0), thickness + 4)
            cv2.putText(debug_img, text, (x, y), font, font_scale, (0, 0, 255), thickness)
        else:
            # Create a lookup from detection box to bib result
            bib_by_box = {tuple(b['box']): b for b in bibs}

            for det in detections:
                x1, y1, x2, y2 = det['box']

                # Draw bounding box
                cv2.rectangle(debug_img, (x1, y1), (x2, y2), (0, 255, 0), 3)

                # Build label
                label = f"{det['confidence']*100:.1f}%"
                bib = bib_by_box.get((x1, y1, x2, y2))
                if bib:
                    label += f" -> {bib['number']} ({bib['confidence']*100:.1f}%)"

                # Get text size for background
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 2.0
                thickness = 4
                (text_w, text_h), baseline = cv2.getTextSize(label, font, font_scale, thickness)

                # Draw background rectangle
                text_x, text_y = x1, y1 - 20
                padding = 5
                cv2.rectangle(debug_img,
                             (text_x - padding, text_y - text_h - padding),
                             (text_x + text_w + padding, text_y + baseline + padding),
                             (0, 0, 0), -1)  # Filled black rectangle

                # Draw label
                cv2.putText(debug_img, label, (text_x, text_y), font, font_scale, (0, 255, 0), thickness)

        cv2.imwrite(output_path, debug_img)


def get_image_files(folder_path: str, recursive: bool = False) -> list[Path]:
    """Get all image files from a folder, optionally including subfolders."""
    image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
    folder = Path(folder_path)

    if recursive:
        # Use rglob to find all images in subfolders
        files = []
        for ext in image_extensions:
            files.extend(folder.rglob(f'*{ext}'))
            files.extend(folder.rglob(f'*{ext.upper()}'))
        # Filter out debug images and sort
        return sorted([
            f for f in files
            if f.is_file()
            and '_debug' not in f.stem
            and 'debug_images' not in f.parts
        ])
    else:
        return sorted([
            f for f in folder.iterdir()
            if f.is_file()
            and f.suffix.lower() in image_extensions
            and '_debug' not in f.stem
        ])
