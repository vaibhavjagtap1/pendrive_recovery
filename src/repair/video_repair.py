"""
Video repair module for MP4 and AVI files.

Attempts to repair corrupted video files by:
1. Reconstructing MP4 moov/ftyp atoms
2. Fixing AVI RIFF headers
3. Re-muxing using available system tools (ffmpeg) if present
"""

import logging
import shutil
import subprocess
import struct
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# MP4 atom / AVI chunk signatures
MP4_FTYP = b"ftyp"
MP4_MOOV = b"moov"
MP4_MDAT = b"mdat"
AVI_RIFF = b"RIFF"
AVI_AVI  = b"AVI "


class VideoRepairer:
    """
    Attempts to detect and repair common video file corruption issues.

    Supports MP4 and AVI formats.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.repaired_dir = self.output_dir / "Repaired_Files" / "videos"
        self.repaired_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def repair(self, video_path: Path) -> Tuple[bool, Optional[Path]]:
        """
        Attempt to repair a video file.

        Args:
            video_path: Path to the potentially corrupted video.

        Returns:
            Tuple of (success, repaired_path). repaired_path is None on failure.
        """
        suffix = video_path.suffix.lower()
        if suffix == ".mp4":
            return self._repair_mp4(video_path)
        elif suffix == ".avi":
            return self._repair_avi(video_path)
        else:
            logger.debug("Unsupported video format: %s", suffix)
            return False, None

    # ------------------------------------------------------------------
    # MP4 repair
    # ------------------------------------------------------------------

    def _repair_mp4(self, path: Path) -> Tuple[bool, Optional[Path]]:
        """Attempt to repair an MP4 file."""
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.error("Cannot read %s: %s", path, exc)
            return False, None

        repaired, changed = self._fix_mp4_bytes(data)

        if ffmpeg_available():
            return self._remux_with_ffmpeg(path, "mp4")

        repaired_path = self._save_repaired(path, repaired, "mp4")
        return changed, repaired_path

    def _fix_mp4_bytes(self, data: bytes) -> Tuple[bytes, bool]:
        """
        Basic MP4 repair:
        - Scan for the first ftyp atom; if the file doesn't start with it,
          try to relocate it to the beginning.
        - Check that an mdat atom exists.
        """
        changed = False
        buf = bytearray(data)

        # Find ftyp atom
        ftyp_idx = data.find(MP4_FTYP)
        if ftyp_idx > 4:
            # ftyp not at expected location; try to move it to start
            ftyp_start = ftyp_idx - 4  # 4-byte size prefix
            ftyp_size = struct.unpack_from(">I", data, ftyp_start)[0]
            if ftyp_start + ftyp_size <= len(data):
                ftyp_atom = data[ftyp_start : ftyp_start + ftyp_size]
                rest = data[:ftyp_start] + data[ftyp_start + ftyp_size :]
                buf = bytearray(ftyp_atom) + bytearray(rest)
                changed = True
                logger.debug("Relocated ftyp atom to start")

        # Verify mdat presence
        if MP4_MDAT not in data:
            logger.warning("No mdat atom found in MP4 – file may be severely corrupted")

        return bytes(buf), changed

    # ------------------------------------------------------------------
    # AVI repair
    # ------------------------------------------------------------------

    def _repair_avi(self, path: Path) -> Tuple[bool, Optional[Path]]:
        """Attempt to repair an AVI file."""
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.error("Cannot read %s: %s", path, exc)
            return False, None

        repaired, changed = self._fix_avi_bytes(data)

        if ffmpeg_available():
            return self._remux_with_ffmpeg(path, "avi")

        repaired_path = self._save_repaired(path, repaired, "avi")
        return changed, repaired_path

    def _fix_avi_bytes(self, data: bytes) -> Tuple[bytes, bool]:
        """
        Basic AVI repair:
        - Ensure RIFF header is present and size field is corrected.
        """
        changed = False
        buf = bytearray(data)

        if not data[:4] == AVI_RIFF:
            logger.warning("AVI RIFF header missing")
            return bytes(buf), changed

        # Fix RIFF chunk size (should be file_size - 8)
        expected_size = len(data) - 8
        actual_size = struct.unpack_from("<I", data, 4)[0]
        if actual_size != expected_size:
            struct.pack_into("<I", buf, 4, expected_size)
            changed = True
            logger.debug("Fixed AVI RIFF chunk size: %d -> %d", actual_size, expected_size)

        return bytes(buf), changed

    # ------------------------------------------------------------------
    # FFmpeg re-mux
    # ------------------------------------------------------------------

    def _remux_with_ffmpeg(self, path: Path, ext: str) -> Tuple[bool, Optional[Path]]:
        """Re-mux a video file using ffmpeg to fix container issues."""
        repaired_path = self._make_repaired_path(path, ext)
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-y",               # Overwrite output
                    "-err_detect", "ignore_err",
                    "-i", str(path),
                    "-c", "copy",       # Stream copy – no re-encoding
                    str(repaired_path),
                ],
                capture_output=True,
                timeout=300,
            )
            if result.returncode == 0:
                logger.info("FFmpeg re-mux succeeded: %s", repaired_path)
                return True, repaired_path
            logger.warning(
                "FFmpeg re-mux failed (rc=%d): %s",
                result.returncode,
                result.stderr[-500:] if result.stderr else "",
            )
            repaired_path.unlink(missing_ok=True)
            return False, None
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.error("FFmpeg error: %s", exc)
            return False, None

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _save_repaired(self, original_path: Path, data: bytes, ext: str) -> Path:
        dest = self._make_repaired_path(original_path, ext)
        dest.write_bytes(data)
        logger.info("Saved repaired video: %s", dest)
        return dest

    def _make_repaired_path(self, original_path: Path, ext: str) -> Path:
        dest = self.repaired_dir / f"{original_path.stem}_repaired.{ext}"
        counter = 0
        while dest.exists():
            counter += 1
            dest = self.repaired_dir / f"{original_path.stem}_repaired_{counter}.{ext}"
        return dest


def ffmpeg_available() -> bool:
    """Return True if ffmpeg is installed and executable."""
    return shutil.which("ffmpeg") is not None
