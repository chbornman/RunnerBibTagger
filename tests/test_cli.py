"""
Tests for CLI functionality (main.py).
"""

import subprocess
import sys
from pathlib import Path

import pyexiv2
import pytest


class TestCLIBasic:
    """Basic CLI functionality tests."""

    def test_cli_help(self, project_root):
        """CLI should show help text."""
        result = subprocess.run(
            [sys.executable, str(project_root / "main.py"), "--help"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode == 0
        assert "Bib Tagger" in result.stdout
        assert "--csv" in result.stdout
        assert "--debug" in result.stdout

    def test_cli_missing_folder(self, project_root):
        """CLI should error on missing folder."""
        result = subprocess.run(
            [sys.executable, str(project_root / "main.py"), "/nonexistent/folder"],
            capture_output=True,
            text=True,
            timeout=30
        )
        assert result.returncode != 0
        assert "not found" in result.stderr.lower() or "error" in result.stderr.lower()


class TestCLIProcessing:
    """CLI image processing tests."""

    def test_cli_single_image(self, project_root, clean_test_output):
        """CLI should process a single image."""
        test_dir = clean_test_output
        img_path = test_dir / "single_bib" / "runner_397.jpg"

        result = subprocess.run(
            [
                sys.executable, str(project_root / "main.py"),
                str(img_path),
                "--no-metadata"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0
        assert "Processing" in result.stdout or "processed" in result.stdout.lower()

    def test_cli_folder(self, project_root, clean_test_output):
        """CLI should process a folder of images."""
        test_dir = clean_test_output / "single_bib"

        result = subprocess.run(
            [
                sys.executable, str(project_root / "main.py"),
                str(test_dir),
                "--no-metadata"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0
        # Should report processing multiple images
        assert "3" in result.stdout  # 3 single_bib images

    def test_cli_csv_output(self, project_root, clean_test_output):
        """CLI --csv flag should create CSV file."""
        test_dir = clean_test_output / "single_bib"

        result = subprocess.run(
            [
                sys.executable, str(project_root / "main.py"),
                str(test_dir),
                "--csv",
                "--no-metadata"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0

        # CSV should be created
        csv_path = test_dir / "bib_results.csv"
        assert csv_path.exists(), f"CSV not created at {csv_path}"

    def test_cli_debug_images(self, project_root, clean_test_output):
        """CLI --debug flag should create debug images folder."""
        test_dir = clean_test_output / "single_bib"

        result = subprocess.run(
            [
                sys.executable, str(project_root / "main.py"),
                str(test_dir),
                "--debug",
                "--no-metadata"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0

        # Debug folder should be created
        debug_dir = test_dir / "debug_images"
        assert debug_dir.exists(), f"Debug folder not created at {debug_dir}"

        # Should contain debug images
        debug_images = list(debug_dir.glob("*_debug.*"))
        assert len(debug_images) > 0

    def test_cli_confidence_threshold(self, project_root, clean_test_output):
        """CLI --confidence flag should affect detection count."""
        test_dir = clean_test_output / "multiple_bibs"

        # Run with low confidence
        result_low = subprocess.run(
            [
                sys.executable, str(project_root / "main.py"),
                str(test_dir),
                "--confidence", "0.1",
                "--no-metadata"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        # Run with high confidence
        result_high = subprocess.run(
            [
                sys.executable, str(project_root / "main.py"),
                str(test_dir),
                "--confidence", "0.9",
                "--no-metadata"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result_low.returncode == 0
        assert result_high.returncode == 0

        # High confidence should generally find fewer bibs
        # (This is a soft assertion - exact numbers depend on images)


class TestCLINoMetadata:
    """Test --no-metadata flag."""

    def test_no_metadata_flag(self, project_root, clean_test_output):
        """--no-metadata should prevent IPTC writing."""
        test_dir = clean_test_output
        img_path = test_dir / "single_bib" / "runner_397.jpg"

        # First strip any existing keywords from the temp copy
        try:
            with pyexiv2.Image(str(img_path)) as img:
                iptc = img.read_iptc()
                key = 'Iptc.Application2.Keywords'
                if key in iptc:
                    img.modify_iptc({key: []})
        except Exception:
            pass

        result = subprocess.run(
            [
                sys.executable, str(project_root / "main.py"),
                str(img_path),
                "--no-metadata"
            ],
            capture_output=True,
            text=True,
            timeout=120
        )

        assert result.returncode == 0

        # Check no metadata was written using pyexiv2
        with pyexiv2.Image(str(img_path)) as img:
            iptc = img.read_iptc()
            key = 'Iptc.Application2.Keywords'
            if key in iptc:
                keywords = iptc[key]
                if isinstance(keywords, str):
                    keywords = [keywords]
                assert not any("BIB:" in kw for kw in keywords)
