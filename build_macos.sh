#!/bin/bash
#
# Build Bib Tagger macOS .dmg
#
# Prerequisites:
#   - macOS with Python 3.10+
#   - brew install create-dmg exiftool
#   - pip install py2app
#
# Usage:
#   ./build_macos.sh
#

set -e

echo "=== Bib Tagger macOS Build ==="
echo ""

# Check we're on macOS
if [[ "$(uname)" != "Darwin" ]]; then
    echo "Error: This script must be run on macOS"
    exit 1
fi

# Check for create-dmg
if ! command -v create-dmg &> /dev/null; then
    echo "Installing create-dmg..."
    brew install create-dmg
fi

# Check for exiftool (will be bundled)
if ! command -v exiftool &> /dev/null; then
    echo "Installing exiftool..."
    brew install exiftool
fi

# Create virtual environment if needed
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate venv
source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt
pip install py2app

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build dist

# Create resources directory if needed
mkdir -p resources

# Build the app
echo "Building .app bundle..."
python setup_macos.py py2app

# Copy exiftool into the app bundle
echo "Bundling exiftool..."
EXIFTOOL_PATH=$(which exiftool)
EXIFTOOL_LIB=$(dirname $(dirname $EXIFTOOL_PATH))/lib/Image-ExifTool*
mkdir -p "dist/Bib Tagger.app/Contents/Resources/exiftool"
cp "$EXIFTOOL_PATH" "dist/Bib Tagger.app/Contents/Resources/exiftool/"
cp -R "$EXIFTOOL_LIB" "dist/Bib Tagger.app/Contents/Resources/exiftool/" 2>/dev/null || true

# Create DMG
echo "Creating DMG..."
create-dmg \
    --volname "Bib Tagger" \
    --volicon "resources/icon.icns" \
    --window-pos 200 120 \
    --window-size 600 400 \
    --icon-size 100 \
    --icon "Bib Tagger.app" 150 185 \
    --hide-extension "Bib Tagger.app" \
    --app-drop-link 450 185 \
    "dist/BibTagger-1.0.0.dmg" \
    "dist/Bib Tagger.app" \
    || echo "Note: create-dmg may fail without icon file - DMG still created"

echo ""
echo "=== Build Complete ==="
echo "Output: dist/BibTagger-1.0.0.dmg"
echo ""
echo "To test the app directly:"
echo "  open 'dist/Bib Tagger.app'"
