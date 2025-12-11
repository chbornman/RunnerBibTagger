#!/bin/bash
# Prepare test images for the repository
# Resizes images to 1600px longest edge, strips metadata, organizes into test folders

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SOURCE_DIR="$PROJECT_DIR/test_images"
TARGET_DIR="$PROJECT_DIR/tests/fixtures/images"

# Check for ImageMagick
if ! command -v magick &> /dev/null; then
    echo "Error: ImageMagick (magick) not found"
    exit 1
fi

# Check for exiftool
if ! command -v exiftool &> /dev/null; then
    echo "Error: exiftool not found"
    exit 1
fi

echo "Preparing test images..."
echo "Source: $SOURCE_DIR"
echo "Target: $TARGET_DIR"

# Create directory structure
rm -rf "$TARGET_DIR"
mkdir -p "$TARGET_DIR/single_bib"
mkdir -p "$TARGET_DIR/multiple_bibs"
mkdir -p "$TARGET_DIR/no_bib"
mkdir -p "$TARGET_DIR/challenging"

# Function to resize and copy image
resize_image() {
    local src="$1"
    local dst="$2"
    echo "  Processing: $(basename "$src") -> $(basename "$dst")"
    magick "$src" -resize 1600x1600\> -strip -quality 85 "$dst"
}

# Single bib images (clear detections with OCR)
echo ""
echo "=== Single Bib Images ==="
resize_image "$SOURCE_DIR/2025CCM3477.jpg" "$TARGET_DIR/single_bib/runner_397.jpg"
resize_image "$SOURCE_DIR/2025CCM3481.jpg" "$TARGET_DIR/single_bib/runner_3380.jpg"
resize_image "$SOURCE_DIR/2025CCM3788.jpg" "$TARGET_DIR/single_bib/runner_529.jpg"

# Multiple bibs images
echo ""
echo "=== Multiple Bibs Images ==="
resize_image "$SOURCE_DIR/2025CCM3479.jpg" "$TARGET_DIR/multiple_bibs/group_060_238.jpg"
resize_image "$SOURCE_DIR/2025CCM3784.jpg" "$TARGET_DIR/multiple_bibs/group_multi.jpg"

# No bib images (0 detections)
echo ""
echo "=== No Bib Images ==="
resize_image "$SOURCE_DIR/2025CCM3475.jpg" "$TARGET_DIR/no_bib/scene_001.jpg"
resize_image "$SOURCE_DIR/2025CCM3776.jpg" "$TARGET_DIR/no_bib/scene_002.jpg"

# Challenging images (detection but no OCR, or partial)
echo ""
echo "=== Challenging Images ==="
resize_image "$SOURCE_DIR/2025CCM2291.jpg" "$TARGET_DIR/challenging/partial_001.jpg"
resize_image "$SOURCE_DIR/2025CCM3474.jpg" "$TARGET_DIR/challenging/partial_002.jpg"

# Final strip of all metadata
echo ""
echo "=== Stripping all metadata ==="
exiftool -all= -overwrite_original "$TARGET_DIR"/**/*.jpg

# Report sizes
echo ""
echo "=== Done! ==="
echo "Total size:"
du -sh "$TARGET_DIR"
echo ""
echo "Per folder:"
du -sh "$TARGET_DIR"/*
echo ""
echo "Image count: $(find "$TARGET_DIR" -name "*.jpg" | wc -l)"
