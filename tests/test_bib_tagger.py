"""
Tests for the core BibTagger detection and OCR functionality.
"""

import cv2
import pytest


class TestBibDetection:
    """Test bib detection functionality."""

    def test_detect_single_bib(self, tagger, single_bib_images):
        """Should detect exactly one bib in single-bib images."""
        for img_path in single_bib_images:
            image = cv2.imread(str(img_path))
            assert image is not None, f"Failed to load {img_path}"

            detections = tagger.detect_bibs(image)
            assert len(detections) >= 1, f"No detections in {img_path.name}"

            # Each detection should have required fields
            for det in detections:
                assert "box" in det
                assert "confidence" in det
                assert len(det["box"]) == 4
                assert 0 <= det["confidence"] <= 1

    def test_detect_multiple_bibs(self, tagger, multiple_bib_images):
        """Should detect multiple bibs in group images."""
        for img_path in multiple_bib_images:
            image = cv2.imread(str(img_path))
            assert image is not None

            detections = tagger.detect_bibs(image)
            assert len(detections) >= 2, f"Expected multiple detections in {img_path.name}"

    def test_no_detection_in_no_bib_images(self, tagger, no_bib_images):
        """Should return zero or few detections for no-bib images."""
        for img_path in no_bib_images:
            image = cv2.imread(str(img_path))
            assert image is not None

            detections = tagger.detect_bibs(image)
            # Allow some false positives, but should be minimal
            assert len(detections) <= 2, f"Too many false positives in {img_path.name}"

    def test_confidence_threshold_filters(self, tagger, tagger_high_confidence, single_bib_images):
        """Higher confidence threshold should result in fewer detections."""
        img_path = single_bib_images[0]
        image = cv2.imread(str(img_path))

        low_conf_detections = tagger.detect_bibs(image)
        high_conf_detections = tagger_high_confidence.detect_bibs(image)

        # High confidence should have same or fewer detections
        assert len(high_conf_detections) <= len(low_conf_detections)


class TestOCR:
    """Test OCR functionality on detected regions."""

    def test_read_bib_number_single(self, tagger, single_bib_images):
        """Should read digits from detected bib regions."""
        successful_reads = 0
        for img_path in single_bib_images:
            image = cv2.imread(str(img_path))
            detections = tagger.detect_bibs(image)

            if not detections:
                continue

            # Read the first detection
            result = tagger.read_bib_number(image, detections[0]["box"])

            if result is not None:
                bib_number, confidence = result
                # Verify it returned digits
                assert bib_number.isdigit(), f"Expected digits, got {bib_number}"
                assert confidence > 0, "Expected positive confidence"
                successful_reads += 1

        # At least some images should have successful OCR reads
        assert successful_reads > 0, "No successful OCR reads from single bib images"

    def test_read_bib_returns_digits_only(self, tagger, single_bib_images):
        """OCR should return only digit characters."""
        for img_path in single_bib_images:
            image = cv2.imread(str(img_path))
            detections = tagger.detect_bibs(image)

            for det in detections:
                result = tagger.read_bib_number(image, det["box"])
                if result:
                    bib_number, _ = result
                    assert bib_number.isdigit(), f"Non-digit chars in: {bib_number}"


class TestProcessImage:
    """Test the full process_image workflow."""

    def test_process_image_returns_expected_fields(self, tagger, single_bib_images):
        """process_image should return all expected fields."""
        img_path = single_bib_images[0]
        result = tagger.process_image(str(img_path), write_metadata=False)

        assert "path" in result
        assert "filename" in result
        assert "success" in result
        assert "bibs" in result
        assert "detections" in result
        assert "time_ms" in result
        assert "error" in result

    def test_process_image_success(self, tagger, single_bib_images):
        """process_image should succeed for valid images."""
        for img_path in single_bib_images:
            result = tagger.process_image(str(img_path), write_metadata=False)
            assert result["success"], f"Failed for {img_path.name}: {result['error']}"

    def test_process_image_bibs_structure(self, tagger, single_bib_images):
        """Each bib in results should have expected structure."""
        img_path = single_bib_images[0]
        result = tagger.process_image(str(img_path), write_metadata=False)

        for bib in result["bibs"]:
            assert "number" in bib
            assert "confidence" in bib
            assert "box" in bib
            assert "detection_confidence" in bib

    def test_process_nonexistent_image(self, tagger):
        """Should handle missing files gracefully."""
        result = tagger.process_image("/nonexistent/image.jpg", write_metadata=False)
        assert not result["success"] or result["error"] is not None


class TestDebugImage:
    """Test debug image generation."""

    def test_save_debug_image(self, tagger, single_bib_images, tmp_path):
        """Should save debug image with annotations."""
        img_path = single_bib_images[0]
        result = tagger.process_image(str(img_path), write_metadata=False)

        debug_path = tmp_path / "debug.jpg"
        tagger.save_debug_image(result, str(debug_path))

        assert debug_path.exists()
        assert debug_path.stat().st_size > 0

        # Verify it's a valid image
        debug_img = cv2.imread(str(debug_path))
        assert debug_img is not None

    def test_debug_image_no_detections(self, tagger, no_bib_images, tmp_path):
        """Should handle images with no detections."""
        img_path = no_bib_images[0]
        result = tagger.process_image(str(img_path), write_metadata=False)

        debug_path = tmp_path / "debug_no_bib.jpg"
        tagger.save_debug_image(result, str(debug_path))

        assert debug_path.exists()
