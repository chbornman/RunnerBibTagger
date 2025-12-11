#!/usr/bin/env python3
"""
Bib Tagger - Detect and tag race bib numbers in photos.

Usage:
    python main.py                          # Launch GUI
    python main.py <folder> [options]       # Process folder via CLI
    python main.py <image.jpg> [options]    # Process single image

Options:
    --csv           Generate bib_results.csv
    --debug         Save debug images to debug_images/
    --no-metadata   Skip writing IPTC metadata to images
    --confidence    Detection confidence threshold (default: 0.25)
"""

import argparse
import csv
import sys
from pathlib import Path

from bib_tagger import BibTagger, get_image_files


def get_model_path() -> str:
    """Get path to the bib detector model."""
    # Check common locations
    candidates = [
        Path(__file__).parent / 'models' / 'bib_detector.onnx',
        Path('models/bib_detector.onnx'),
    ]

    for path in candidates:
        if path.exists():
            return str(path)

    print("Error: bib_detector.onnx not found in models/", file=sys.stderr)
    sys.exit(1)


def run_cli(args):
    """Run in CLI mode."""
    input_path = Path(args.input)

    if not input_path.exists():
        print(f"Error: {args.input} not found", file=sys.stderr)
        sys.exit(1)

    # Get images
    if input_path.is_dir():
        images = get_image_files(str(input_path))
        output_dir = input_path
    else:
        images = [input_path]
        output_dir = input_path.parent

    if not images:
        print("No images found")
        sys.exit(1)

    # Initialize tagger
    model_path = get_model_path()
    tagger = BibTagger(
        model_path=model_path,
        confidence=args.confidence,
        box_padding=0.15,
    )

    # Create debug folder if needed
    debug_folder = None
    if args.debug:
        debug_folder = output_dir / 'debug_images'
        debug_folder.mkdir(exist_ok=True)
        print(f"Debug images: {debug_folder}/")

    print(f"\nProcessing {len(images)} image(s)...\n")

    # Process images
    results = []
    total_bibs = 0

    for i, img_path in enumerate(images, 1):
        print(f"[{i}/{len(images)}] {img_path.name}", end="")

        # Process image, respecting --no-metadata flag
        write_meta = not args.no_metadata
        result = tagger.process_image(str(img_path), write_metadata=write_meta)

        results.append(result)

        # Status
        if result['bibs']:
            bibs_str = ', '.join(b['number'] for b in result['bibs'])
            print(f" -> {bibs_str}")
            total_bibs += len(result['bibs'])
        else:
            print(f" -> (no bibs)")

        # Save debug image (always save when debug enabled)
        if debug_folder:
            debug_path = debug_folder / f"{img_path.stem}_debug{img_path.suffix}"
            tagger.save_debug_image(result, str(debug_path))

    # Generate CSV
    if args.csv:
        csv_path = output_dir / 'bib_results.csv'
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Filename', 'Bib Numbers', 'Detections', 'Time (ms)'])
            for r in results:
                bibs = ', '.join(b['number'] for b in r['bibs']) if r['bibs'] else ''
                writer.writerow([r['filename'], bibs, r['detections'], f"{r['time_ms']:.0f}"])
        print(f"\nCSV: {csv_path}")

    # Summary
    print(f"\nProcessed {len(images)} images, found {total_bibs} bib(s)")


def run_gui():
    """Launch the GUI."""
    from gui import BibTaggerApp
    app = BibTaggerApp()
    app.run()


def main():
    parser = argparse.ArgumentParser(
        description='Bib Tagger - Detect race bib numbers in photos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              Launch GUI
  python main.py photos/ --csv                Process folder, generate CSV
  python main.py race.jpg --debug             Process image with debug output
"""
    )

    parser.add_argument('input', nargs='?', help='Image or folder to process (omit for GUI)')
    parser.add_argument('--csv', action='store_true', help='Generate CSV results file')
    parser.add_argument('--debug', action='store_true', help='Save debug images')
    parser.add_argument('--no-metadata', action='store_true', help='Skip writing IPTC metadata')
    parser.add_argument('--confidence', type=float, default=0.25, help='Detection threshold (0-1)')

    args = parser.parse_args()

    if args.input:
        run_cli(args)
    else:
        run_gui()


if __name__ == '__main__':
    main()
