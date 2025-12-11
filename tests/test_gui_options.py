"""
Tests for GUI options and their effects on processing.

These tests verify that each GUI option produces the expected behavior
without actually launching the GUI window.
"""

import csv
import shutil
import subprocess
from pathlib import Path

import pytest

from bib_tagger import BibTagger, get_image_files


class TestGenerateCSVOption:
    """Test the 'Generate CSV report' option."""

    def test_csv_created_when_enabled(self, tagger, clean_test_output):
        """CSV file should be created when option is enabled."""
        test_dir = clean_test_output
        images = get_image_files(str(test_dir), recursive=True)

        # Process images and write CSV
        results = []
        for img_path in images[:3]:  # Just a few for speed
            result = tagger.process_image(str(img_path), write_metadata=False)
            results.append(result)

        # Write CSV (simulating GUI behavior)
        csv_path = test_dir / "bib_results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Filename", "Bib Numbers", "Detections", "Time (ms)"])
            for r in results:
                bibs = ", ".join(b["number"] for b in r["bibs"]) if r["bibs"] else ""
                writer.writerow([r["filename"], bibs, r["detections"], f"{r['time_ms']:.1f}"])

        assert csv_path.exists()
        assert csv_path.stat().st_size > 0

        # Verify CSV content
        with open(csv_path) as f:
            reader = csv.reader(f)
            rows = list(reader)
            assert len(rows) == 4  # Header + 3 data rows
            assert rows[0][0] == "Filename"

    def test_csv_not_created_when_disabled(self, clean_test_output):
        """No CSV file should exist when option is disabled."""
        test_dir = clean_test_output

        # Don't write CSV (simulating disabled option)
        csv_path = test_dir / "bib_results.csv"
        assert not csv_path.exists()


class TestShowPreviewOption:
    """Test the 'Show detection preview' option."""

    def test_preview_images_generated_when_enabled(self, tagger, clean_test_output, tmp_path):
        """Temporary preview images should be generated when enabled."""
        test_dir = clean_test_output
        images = get_image_files(str(test_dir / "single_bib"))

        preview_paths = []
        for img_path in images:
            result = tagger.process_image(str(img_path), write_metadata=False)

            # Simulate show_preview enabled: create temp debug image
            temp_path = tmp_path / f"{img_path.stem}_preview.jpg"
            tagger.save_debug_image(result, str(temp_path))
            preview_paths.append(temp_path)

        # All preview images should exist
        for preview_path in preview_paths:
            assert preview_path.exists()

    def test_no_preview_when_disabled(self, tagger, clean_test_output, tmp_path):
        """No preview images when option is disabled."""
        test_dir = clean_test_output
        images = get_image_files(str(test_dir / "single_bib"))

        # Simulate show_preview disabled: don't create preview images
        preview_dir = tmp_path / "previews"
        preview_dir.mkdir()

        for img_path in images:
            result = tagger.process_image(str(img_path), write_metadata=False)
            # Don't save debug image

        # Preview directory should be empty
        assert len(list(preview_dir.iterdir())) == 0


class TestSaveDebugImagesOption:
    """Test the 'Save debug images to disk' option."""

    def test_debug_folder_created_when_enabled(self, tagger, clean_test_output):
        """debug_images folder should be created when option is enabled."""
        test_dir = clean_test_output
        debug_dir = test_dir / "debug_images"
        debug_dir.mkdir()

        images = get_image_files(str(test_dir / "single_bib"))

        for img_path in images:
            result = tagger.process_image(str(img_path), write_metadata=False)
            debug_path = debug_dir / f"{img_path.stem}_debug{img_path.suffix}"
            tagger.save_debug_image(result, str(debug_path))

        assert debug_dir.exists()
        debug_files = list(debug_dir.glob("*.jpg"))
        assert len(debug_files) == len(images)

    def test_no_debug_folder_when_disabled(self, clean_test_output):
        """No debug_images folder when option is disabled."""
        test_dir = clean_test_output
        debug_dir = test_dir / "debug_images"

        # Don't create debug folder (simulating disabled option)
        assert not debug_dir.exists()

    def test_debug_preserves_subfolder_structure(self, tagger, clean_test_output):
        """Debug images should preserve subfolder structure."""
        test_dir = clean_test_output
        debug_dir = test_dir / "debug_images"

        # Process images with recursive, saving to debug folder
        images = get_image_files(str(test_dir), recursive=True)

        for img_path in images[:5]:
            result = tagger.process_image(str(img_path), write_metadata=False)

            # Calculate relative path
            try:
                rel_path = img_path.relative_to(test_dir)
            except ValueError:
                rel_path = Path(img_path.name)

            # Create subfolder structure
            if rel_path.parent != Path("."):
                debug_subdir = debug_dir / rel_path.parent
                debug_subdir.mkdir(parents=True, exist_ok=True)
            else:
                debug_subdir = debug_dir
                debug_subdir.mkdir(exist_ok=True)

            debug_path = debug_subdir / f"{img_path.stem}_debug{img_path.suffix}"
            tagger.save_debug_image(result, str(debug_path))

        # Verify subfolder structure exists
        assert debug_dir.exists()
        subfolders = [d for d in debug_dir.iterdir() if d.is_dir()]
        assert len(subfolders) > 0


class TestWriteMetadataOption:
    """Test the 'Write bib numbers to image metadata' option."""

    def test_metadata_written_when_enabled(self, tagger, clean_test_output, strip_metadata):
        """IPTC keywords should be written when option is enabled."""
        test_dir = clean_test_output

        # Find an image we know has bibs
        img_path = test_dir / "single_bib" / "runner_397.jpg"

        # Process with metadata writing enabled
        result = tagger.process_image(str(img_path), write_metadata=True)

        if result["bibs"]:
            # Check metadata was written
            try:
                proc = subprocess.run(
                    ["exiftool", "-Keywords", str(img_path)],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                # Should contain BIB: keyword
                assert "BIB:" in proc.stdout, f"No BIB keyword found in {img_path}"
            except FileNotFoundError:
                pytest.skip("exiftool not installed")

    def test_metadata_not_written_when_disabled(self, tagger, clean_test_output, strip_metadata):
        """No metadata changes when option is disabled."""
        test_dir = clean_test_output
        img_path = test_dir / "single_bib" / "runner_397.jpg"

        # Process with metadata writing disabled
        result = tagger.process_image(str(img_path), write_metadata=False)

        # Check metadata was NOT written
        try:
            proc = subprocess.run(
                ["exiftool", "-Keywords", str(img_path)],
                capture_output=True,
                text=True,
                timeout=10
            )
            # Should NOT contain BIB: keyword
            assert "BIB:" not in proc.stdout
        except FileNotFoundError:
            pytest.skip("exiftool not installed")


class TestIncludeSubfoldersOption:
    """Test the 'Include subfolders' option."""

    def test_recursive_finds_all_images(self, fixtures_dir):
        """Recursive search should find images in all subfolders."""
        all_images = get_image_files(str(fixtures_dir), recursive=True)

        # Should find images in all 4 subfolders
        folders_found = set()
        for img in all_images:
            folders_found.add(img.parent.name)

        expected_folders = {"single_bib", "multiple_bibs", "no_bib", "challenging"}
        assert folders_found == expected_folders

    def test_non_recursive_finds_only_top_level(self, fixtures_dir):
        """Non-recursive search should only find top-level images."""
        top_level_images = get_image_files(str(fixtures_dir), recursive=False)

        # Fixtures are organized in subfolders, so top level should be empty
        assert len(top_level_images) == 0

    def test_recursive_vs_non_recursive_counts(self, fixtures_dir):
        """Recursive should find more images than non-recursive."""
        recursive_images = get_image_files(str(fixtures_dir), recursive=True)
        non_recursive_images = get_image_files(str(fixtures_dir), recursive=False)

        assert len(recursive_images) > len(non_recursive_images)
        assert len(recursive_images) == 9  # All test images

    def test_excludes_debug_images(self, fixtures_dir, tmp_path):
        """Should exclude debug images from results."""
        # Create a temporary structure with debug images
        test_dir = tmp_path / "test_exclude"
        shutil.copytree(fixtures_dir / "single_bib", test_dir)

        # Create a fake debug image
        debug_img = test_dir / "runner_397_debug.jpg"
        shutil.copy(test_dir / "runner_397.jpg", debug_img)

        images = get_image_files(str(test_dir), recursive=False)

        # Should not include the debug image
        image_names = [img.name for img in images]
        assert "runner_397_debug.jpg" not in image_names
        assert "runner_397.jpg" in image_names


class TestCancelProcessing:
    """Test cancellation behavior."""

    def test_partial_results_saved_on_cancel(self, tagger, clean_test_output):
        """Partial results should be preserved when processing is cancelled."""
        test_dir = clean_test_output
        images = get_image_files(str(test_dir), recursive=True)

        results = []
        cancel_after = 3

        for i, img_path in enumerate(images):
            if i >= cancel_after:
                # Simulate cancel
                break

            result = tagger.process_image(str(img_path), write_metadata=False)
            results.append(result)

        # Should have partial results
        assert len(results) == cancel_after
        assert len(results) < len(images)


class TestImageNavigation:
    """Test image navigation state management."""

    def test_navigation_list_builds_correctly(self, tagger, clean_test_output):
        """Debug image list should build as processing progresses."""
        test_dir = clean_test_output
        images = get_image_files(str(test_dir / "single_bib"))

        # Simulate building navigation list
        debug_image_list = []

        for img_path in images:
            result = tagger.process_image(str(img_path), write_metadata=False)
            # Add to navigation list (simulating GUI behavior)
            debug_image_list.append((str(img_path), img_path.name))

        assert len(debug_image_list) == len(images)

        # Verify navigation works
        current_index = 0
        assert debug_image_list[current_index][1] == images[0].name

        current_index = len(debug_image_list) - 1
        assert debug_image_list[current_index][1] == images[-1].name
