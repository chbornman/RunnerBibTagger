#!/usr/bin/env python3
"""
Evaluate bib detector on validation dataset.

Compares YOLO detections against ground truth labels and computes
precision, recall, and mAP metrics.
"""

import sys
import time
from pathlib import Path

import cv2
import numpy as np

from bib_tagger import BibTagger


# Paths
VAL_IMAGES = Path('/home/caleb/projects/SonyTagger/CUSTOM_TRAINED_MODEL/datasets/images/val')
VAL_LABELS = Path('/home/caleb/projects/SonyTagger/CUSTOM_TRAINED_MODEL/datasets/labels/val')


def parse_yolo_label(label_path: Path, img_w: int, img_h: int) -> list[tuple]:
    """Parse YOLO label file and convert to absolute coordinates.

    Returns list of (x1, y1, x2, y2) boxes.
    """
    boxes = []
    if not label_path.exists():
        return boxes

    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue

            # YOLO format: class x_center y_center width height
            # (normalized 0-1)
            _, x_center, y_center, w, h = map(float, parts[:5])

            # Convert to absolute pixel coordinates
            x1 = int((x_center - w/2) * img_w)
            y1 = int((y_center - h/2) * img_h)
            x2 = int((x_center + w/2) * img_w)
            y2 = int((y_center + h/2) * img_h)

            boxes.append((x1, y1, x2, y2))

    return boxes


def iou(box1: tuple, box2: tuple) -> float:
    """Calculate Intersection over Union between two boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)

    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union_area = box1_area + box2_area - inter_area

    return inter_area / union_area if union_area > 0 else 0


def match_detections(pred_boxes: list, gt_boxes: list, iou_threshold: float = 0.5):
    """Match predicted boxes to ground truth boxes.

    Returns (true_positives, false_positives, false_negatives)
    """
    if not gt_boxes:
        return 0, len(pred_boxes), 0

    if not pred_boxes:
        return 0, 0, len(gt_boxes)

    # Track which GT boxes have been matched
    gt_matched = [False] * len(gt_boxes)
    tp = 0
    fp = 0

    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1

        for i, gt in enumerate(gt_boxes):
            if gt_matched[i]:
                continue

            current_iou = iou(pred, gt)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_idx = i

        if best_iou >= iou_threshold:
            tp += 1
            gt_matched[best_gt_idx] = True
        else:
            fp += 1

    fn = sum(1 for m in gt_matched if not m)

    return tp, fp, fn


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--confidence', '-c', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.5)
    args = parser.parse_args()

    print("=" * 60)
    print("Bib Detector Evaluation")
    print("=" * 60)

    # Get all validation images
    images = sorted(VAL_IMAGES.glob('*.jpg'))
    print(f"\nFound {len(images)} validation images")
    print(f"Labels directory: {VAL_LABELS}")

    # Initialize detector
    model_path = Path(__file__).parent / 'models' / 'bib_detector.onnx'
    print(f"\nLoading model: {model_path}")
    print(f"Confidence threshold: {args.confidence}")
    print(f"IoU threshold: {args.iou}")

    tagger = BibTagger(
        model_path=str(model_path),
        confidence=args.confidence,
        box_padding=0.0,
    )

    # Metrics
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0
    total_time = 0

    iou_thresholds = [0.5]  # Can add more: [0.5, 0.75]

    print(f"\nProcessing images (IoU threshold: {iou_thresholds[0]})...\n")

    for i, img_path in enumerate(images):
        # Load image to get dimensions
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Warning: Could not load {img_path.name}")
            continue

        img_h, img_w = img.shape[:2]

        # Get ground truth
        label_path = VAL_LABELS / f"{img_path.stem}.txt"
        gt_boxes = parse_yolo_label(label_path, img_w, img_h)

        # Run detection
        start = time.time()
        detections = tagger.detect_bibs(img)
        elapsed = time.time() - start
        total_time += elapsed

        pred_boxes = [d['box'] for d in detections]

        # Match detections
        tp, fp, fn = match_detections(pred_boxes, gt_boxes, iou_threshold=args.iou)

        total_tp += tp
        total_fp += fp
        total_fn += fn
        total_gt += len(gt_boxes)
        total_pred += len(pred_boxes)

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(images)}] processed...")

    # Calculate metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    avg_time_ms = (total_time / len(images)) * 1000

    # Results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nDataset:")
    print(f"  Images:              {len(images)}")
    print(f"  Ground truth bibs:   {total_gt}")
    print(f"  Predicted bibs:      {total_pred}")

    print(f"\nDetection Metrics (IoU >= {args.iou}):")
    print(f"  True Positives:      {total_tp}")
    print(f"  False Positives:     {total_fp}")
    print(f"  False Negatives:     {total_fn}")

    print(f"\nPerformance:")
    print(f"  Precision:           {precision:.3f} ({precision*100:.1f}%)")
    print(f"  Recall:              {recall:.3f} ({recall*100:.1f}%)")
    print(f"  F1 Score:            {f1:.3f}")

    print(f"\nSpeed:")
    print(f"  Total time:          {total_time:.1f}s")
    print(f"  Avg per image:       {avg_time_ms:.1f}ms")
    print(f"  Images/sec:          {len(images)/total_time:.1f}")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
