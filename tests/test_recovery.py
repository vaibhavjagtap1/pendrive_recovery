"""Tests for the recovery engine and file signatures."""

import os
import struct
import tempfile
from pathlib import Path

import pytest

from src.recovery.signatures import FILE_SIGNATURES, HEADER_LOOKUP, MAX_HEADER_LEN
from src.recovery.engine import RecoveryEngine, RecoveredFile
from src.recovery.filesystem import detect_filesystem, FilesystemType


# -----------------------------------------------------------------------
# Signatures
# -----------------------------------------------------------------------

class TestFileSignatures:
    def test_all_extensions_have_header(self):
        for ext, sig in FILE_SIGNATURES.items():
            assert "header" in sig, f"{ext} missing header"
            assert isinstance(sig["header"], bytes)
            assert len(sig["header"]) >= 2

    def test_max_size_positive(self):
        for ext, sig in FILE_SIGNATURES.items():
            assert sig["max_size"] > 0, f"{ext} max_size must be positive"

    def test_header_lookup_populated(self):
        assert len(HEADER_LOOKUP) > 0

    def test_max_header_len(self):
        expected = max(len(s["header"]) for s in FILE_SIGNATURES.values())
        assert MAX_HEADER_LEN == expected

    def test_jpeg_signature(self):
        assert FILE_SIGNATURES["jpg"]["header"] == b"\xFF\xD8\xFF"
        assert FILE_SIGNATURES["jpg"]["footer"] == b"\xFF\xD9"

    def test_png_signature(self):
        assert FILE_SIGNATURES["png"]["header"][:4] == b"\x89PNG"

    def test_pdf_signature(self):
        assert FILE_SIGNATURES["pdf"]["header"] == b"\x25\x50\x44\x46"


# -----------------------------------------------------------------------
# Filesystem detection
# -----------------------------------------------------------------------

class TestFilesystemDetection:
    def _boot_sector_with_oem(self, oem_id: bytes) -> bytes:
        sector = bytearray(512)
        sector[3:3 + len(oem_id)] = oem_id
        return bytes(sector)

    def test_detect_ntfs(self):
        sector = self._boot_sector_with_oem(b"NTFS    ")
        assert detect_filesystem(sector) == FilesystemType.NTFS

    def test_detect_exfat(self):
        sector = self._boot_sector_with_oem(b"EXFAT   ")
        assert detect_filesystem(sector) == FilesystemType.EXFAT

    def test_detect_fat32_by_type_string(self):
        sector = bytearray(512)
        sector[82:90] = b"FAT32   "
        assert detect_filesystem(bytes(sector)) == FilesystemType.FAT32

    def test_detect_unknown(self):
        sector = bytes(512)
        assert detect_filesystem(sector) == FilesystemType.UNKNOWN

    def test_too_short_returns_unknown(self):
        assert detect_filesystem(b"\x00" * 10) == FilesystemType.UNKNOWN


# -----------------------------------------------------------------------
# RecoveredFile
# -----------------------------------------------------------------------

class TestRecoveredFile:
    def test_save_creates_file(self, tmp_dir):
        data = b"\xFF\xD8\xFF" + b"\x00" * 100 + b"\xFF\xD9"
        rf = RecoveredFile(data=data, extension="jpg", offset=0, method="test")
        saved = rf.save(tmp_dir)
        assert saved.exists()
        assert saved.read_bytes() == data

    def test_save_avoids_overwrite(self, tmp_dir):
        data = b"\xFF\xD8\xFF" + b"\x00" * 100 + b"\xFF\xD9"
        rf = RecoveredFile(data=data, extension="jpg", offset=0, method="test")
        path1 = rf.save(tmp_dir)
        # Save again; should create a different filename
        rf2 = RecoveredFile(data=data, extension="jpg", offset=0, method="test")
        path2 = rf2.save(tmp_dir)
        assert path1 != path2 or not path1.exists()  # at least distinct writes attempted

    def test_sha256_computed(self):
        data = b"test data"
        rf = RecoveredFile(data=data, extension="txt", offset=0, method="test")
        import hashlib
        assert rf.sha256 == hashlib.sha256(data).hexdigest()

    def test_size_property(self):
        data = b"hello world"
        rf = RecoveredFile(data=data, extension="txt", offset=0, method="test")
        assert rf.size == len(data)


# -----------------------------------------------------------------------
# RecoveryEngine (integration-style with a synthesized disk image)
# -----------------------------------------------------------------------

def make_disk_image(path: Path) -> Path:
    """Create a small synthetic disk image containing embedded JPEG bytes."""
    jpeg_header = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00"
    jpeg_footer = b"\xFF\xD9"
    filler = b"\x00" * 512

    image_data = (
        filler * 4                 # padding before file
        + jpeg_header
        + b"\xAB" * 200            # fake JPEG body
        + jpeg_footer
        + filler * 2               # trailing padding
    )
    path.write_bytes(image_data)
    return path


class TestRecoveryEngine:
    def test_carve_jpeg_from_image(self, tmp_dir):
        img_path = tmp_dir / "disk.img"
        make_disk_image(img_path)
        engine = RecoveryEngine(str(img_path), str(tmp_dir / "output"))
        recovered = engine.run()
        jpeg_files = [r for r in recovered if r.extension == "jpg"]
        assert len(jpeg_files) >= 1, "Expected at least one JPEG to be carved"

    def test_recovery_deduplicates_by_sha256(self, tmp_dir):
        img_path = tmp_dir / "disk.img"
        make_disk_image(img_path)
        engine = RecoveryEngine(str(img_path), str(tmp_dir / "output"))
        recovered = engine.run()
        sha_set = {r.sha256 for r in recovered}
        assert len(sha_set) == len(recovered), "Duplicate SHA-256 found after dedup"

    def test_nonexistent_device_raises(self, tmp_dir):
        engine = RecoveryEngine("/nonexistent/device.img", str(tmp_dir / "output"))
        with pytest.raises(OSError):
            engine.run()
