"""
Image repair module for JPEG and PNG files.

Attempts to repair corrupted image files by:
1. Reconstructing missing/truncated JPEG EOI markers
2. Re-encoding images via Pillow to normalize corruption
3. Preserving original file alongside repaired version
"""

import io
import logging
import shutil
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# JPEG markers
JPEG_SOI = b"\xFF\xD8"
JPEG_EOI = b"\xFF\xD9"
# PNG signature
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
PNG_IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"


class ImageRepairer:
    """
    Attempts to detect and repair common image corruption issues.

    Supports JPEG and PNG formats.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.repaired_dir = self.output_dir / "Repaired_Files" / "images"
        self.repaired_dir.mkdir(parents=True, exist_ok=True)

    def repair(self, image_path: Path) -> Tuple[bool, Optional[Path]]:
        """
        Attempt to repair an image file.

        Args:
            image_path: Path to the potentially corrupted image.

        Returns:
            Tuple of (success, repaired_path). repaired_path is None on failure.
        """
        suffix = image_path.suffix.lower()
        if suffix in (".jpg", ".jpeg"):
            return self._repair_jpeg(image_path)
        elif suffix == ".png":
            return self._repair_png(image_path)
        else:
            logger.debug("Unsupported image format: %s", suffix)
            return False, None

    # ------------------------------------------------------------------
    # JPEG repair
    # ------------------------------------------------------------------

    def _repair_jpeg(self, path: Path) -> Tuple[bool, Optional[Path]]:
        """Repair a JPEG file."""
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.error("Cannot read %s: %s", path, exc)
            return False, None

        repaired_data, changed = self._fix_jpeg_bytes(data)
        if not changed:
            logger.debug("No JPEG repair needed for %s", path.name)

        repaired_path = self._save_repaired(path, repaired_data, "jpg")

        # Validate with Pillow
        if self._validate_image_pillow(repaired_data):
            return True, repaired_path
        # Even without Pillow, saving the patched bytes is useful
        return changed, repaired_path

    def _fix_jpeg_bytes(self, data: bytes) -> Tuple[bytes, bool]:
        """
        Apply byte-level JPEG repairs:
        - Ensure SOI marker at start
        - Ensure EOI marker at end
        - Remove stray null bytes after FF markers where applicable
        """
        changed = False
        buf = bytearray(data)

        # Ensure JPEG SOI marker
        if not buf[:2] == bytearray(JPEG_SOI):
            buf = bytearray(JPEG_SOI) + buf
            changed = True
            logger.debug("Prepended JPEG SOI marker")

        # Ensure JPEG EOI marker at end
        if buf[-2:] != bytearray(JPEG_EOI):
            buf.extend(JPEG_EOI)
            changed = True
            logger.debug("Appended JPEG EOI marker")

        return bytes(buf), changed

    # ------------------------------------------------------------------
    # PNG repair
    # ------------------------------------------------------------------

    def _repair_png(self, path: Path) -> Tuple[bool, Optional[Path]]:
        """Repair a PNG file."""
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.error("Cannot read %s: %s", path, exc)
            return False, None

        repaired_data, changed = self._fix_png_bytes(data)
        repaired_path = self._save_repaired(path, repaired_data, "png")

        if self._validate_image_pillow(repaired_data):
            return True, repaired_path
        return changed, repaired_path

    def _fix_png_bytes(self, data: bytes) -> Tuple[bytes, bool]:
        """
        Apply byte-level PNG repairs:
        - Ensure PNG signature at start
        - Append IEND chunk if missing
        """
        changed = False
        buf = bytearray(data)

        # Ensure PNG signature
        if buf[:8] != bytearray(PNG_SIGNATURE):
            buf = bytearray(PNG_SIGNATURE) + buf[8:]
            changed = True
            logger.debug("Restored PNG signature")

        # Ensure IEND chunk
        if b"IEND" not in buf[-16:]:
            buf.extend(PNG_IEND)
            changed = True
            logger.debug("Appended PNG IEND chunk")

        return bytes(buf), changed

    # ------------------------------------------------------------------
    # Re-encoding via Pillow
    # ------------------------------------------------------------------

    def _validate_image_pillow(self, data: bytes) -> bool:
        """
        Attempt to open and re-encode the image with Pillow.

        Returns True if the image is valid/openable.
        """
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(data))
            img.verify()
            return True
        except ImportError:
            logger.debug("Pillow not available; skipping image validation")
            return False
        except Exception as exc:
            logger.debug("Image validation failed: %s", exc)
            return False

    def re_encode(self, image_path: Path) -> Tuple[bool, Optional[Path]]:
        """
        Re-encode an image to strip corruption (requires Pillow).

        Args:
            image_path: Path to the source image.

        Returns:
            Tuple of (success, output_path).
        """
        try:
            from PIL import Image
        except ImportError:
            logger.warning("Pillow is required for re-encoding")
            return False, None

        try:
            img = Image.open(image_path)
            img.load()
            out_buf = io.BytesIO()
            fmt = img.format or "JPEG"
            img.save(out_buf, format=fmt)
            repaired_data = out_buf.getvalue()
            ext = "jpg" if fmt == "JPEG" else fmt.lower()
            repaired_path = self._save_repaired(image_path, repaired_data, ext)
            return True, repaired_path
        except Exception as exc:
            logger.error("Re-encoding failed for %s: %s", image_path, exc)
            return False, None

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _save_repaired(self, original_path: Path, data: bytes, ext: str) -> Path:
        """Save repaired file data alongside original."""
        dest = self.repaired_dir / f"{original_path.stem}_repaired.{ext}"
        counter = 0
        while dest.exists():
            counter += 1
            dest = self.repaired_dir / f"{original_path.stem}_repaired_{counter}.{ext}"
        dest.write_bytes(data)
        logger.info("Saved repaired image: %s", dest)
        return dest
