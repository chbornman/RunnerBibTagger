"""
Pytest configuration and fixtures for Bib Tagger tests.

Provides:
- Test image paths
- Cleanup fixtures to reset state before tests
- Temporary directory fixtures
"""

import shutil
import subprocess
from pathlib import Path

import pytest

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
FIXTURES_DIR = Path(__file__).parent / "fixtures" / "images"
MODEL_PATH = PROJECT_ROOT / "models" / "bib_detector.onnx"


@pytest.fixture(scope="session")
def project_root():
    """Return the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def model_path():
    """Return path to the ONNX model."""
    if not MODEL_PATH.exists():
        pytest.skip(f"Model not found: {MODEL_PATH}")
    return str(MODEL_PATH)


@pytest.fixture(scope="session")
def fixtures_dir():
    """Return the test fixtures directory."""
    if not FIXTURES_DIR.exists():
        pytest.skip(f"Test fixtures not found: {FIXTURES_DIR}")
    return FIXTURES_DIR


# === Image path fixtures ===

@pytest.fixture
def single_bib_images(fixtures_dir):
    """Return list of paths to single-bib test images."""
    folder = fixtures_dir / "single_bib"
    return sorted(folder.glob("*.jpg"))


@pytest.fixture
def multiple_bib_images(fixtures_dir):
    """Return list of paths to multiple-bib test images."""
    folder = fixtures_dir / "multiple_bibs"
    return sorted(folder.glob("*.jpg"))


@pytest.fixture
def no_bib_images(fixtures_dir):
    """Return list of paths to no-bib test images."""
    folder = fixtures_dir / "no_bib"
    return sorted(folder.glob("*.jpg"))


@pytest.fixture
def challenging_images(fixtures_dir):
    """Return list of paths to challenging test images."""
    folder = fixtures_dir / "challenging"
    return sorted(folder.glob("*.jpg"))


@pytest.fixture
def all_test_images(fixtures_dir):
    """Return list of all test image paths."""
    return sorted(fixtures_dir.rglob("*.jpg"))


# === Cleanup fixtures ===

@pytest.fixture
def clean_test_output(fixtures_dir, tmp_path):
    """
    Provide a clean temporary directory for test output.
    Copies test images to temp dir and cleans up after test.
    """
    # Copy fixture images to temp directory
    test_dir = tmp_path / "test_images"
    shutil.copytree(fixtures_dir, test_dir)

    yield test_dir

    # Cleanup happens automatically with tmp_path


@pytest.fixture
def clean_fixtures(fixtures_dir):
    """
    Clean up any artifacts in the fixtures directory before test.
    Removes debug_images folder and CSV files.
    """
    # Remove debug_images if exists
    debug_dir = fixtures_dir / "debug_images"
    if debug_dir.exists():
        shutil.rmtree(debug_dir)

    # Remove any CSV files
    for csv_file in fixtures_dir.glob("*.csv"):
        csv_file.unlink()

    # Remove any CSV files in parent (tests/fixtures)
    for csv_file in fixtures_dir.parent.glob("*.csv"):
        csv_file.unlink()

    yield fixtures_dir

    # Post-test cleanup
    if debug_dir.exists():
        shutil.rmtree(debug_dir)
    for csv_file in fixtures_dir.glob("*.csv"):
        csv_file.unlink()
    for csv_file in fixtures_dir.parent.glob("*.csv"):
        csv_file.unlink()


@pytest.fixture
def strip_metadata(fixtures_dir):
    """
    Strip all IPTC metadata from test images before test.
    Restores clean state for metadata writing tests.
    """
    # Strip all metadata using exiftool
    try:
        result = subprocess.run(
            ["exiftool", "-all=", "-overwrite_original", "-r", str(fixtures_dir)],
            capture_output=True,
            text=True,
            timeout=30
        )
    except FileNotFoundError:
        pytest.skip("exiftool not installed")
    except subprocess.TimeoutExpired:
        pytest.fail("exiftool timed out")

    yield fixtures_dir


# === BibTagger fixtures ===

@pytest.fixture
def tagger(model_path):
    """Create a BibTagger instance for testing."""
    from bib_tagger import BibTagger
    return BibTagger(model_path=model_path, confidence=0.25)


@pytest.fixture
def tagger_high_confidence(model_path):
    """Create a BibTagger with high confidence threshold."""
    from bib_tagger import BibTagger
    return BibTagger(model_path=model_path, confidence=0.5)


# === Expected results for validation ===

# These are the expected bib numbers based on manual verification
# Format: filename -> list of expected bib numbers
EXPECTED_BIBS = {
    "runner_397.jpg": ["397"],
    "runner_3380.jpg": ["3380"],
    "runner_529.jpg": ["529"],
    "group_060_238.jpg": ["60", "238"],
    "group_multi.jpg": ["503", "430", "414", "341"],  # May vary with confidence
    "scene_001.jpg": [],
    "scene_002.jpg": [],
    "partial_001.jpg": [],  # Detection but no OCR
    "partial_002.jpg": [],  # Detection but no OCR
}


@pytest.fixture
def expected_bibs():
    """Return expected bib numbers for test images."""
    return EXPECTED_BIBS
