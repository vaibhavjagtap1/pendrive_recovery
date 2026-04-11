"""Tests for the file repair modules."""

from pathlib import Path

import pytest

from src.repair.image_repair import ImageRepairer
from src.repair.video_repair import VideoRepairer
from src.repair.document_repair import DocumentRepairer


# -----------------------------------------------------------------------
# Image repair
# -----------------------------------------------------------------------

class TestImageRepairer:
    def test_repair_jpeg_appends_eoi(self, tmp_dir, minimal_jpeg_bytes):
        repairer = ImageRepairer(str(tmp_dir))
        # Truncate the EOI marker
        truncated = minimal_jpeg_bytes[:-2]
        img_file = tmp_dir / "broken.jpg"
        img_file.write_bytes(truncated)

        success, repaired_path = repairer.repair(img_file)
        assert repaired_path is not None
        assert repaired_path.exists()
        repaired_data = repaired_path.read_bytes()
        assert repaired_data[-2:] == b"\xFF\xD9"

    def test_repair_jpeg_prepends_soi(self, tmp_dir, minimal_jpeg_bytes):
        repairer = ImageRepairer(str(tmp_dir))
        # Remove the SOI marker
        broken = minimal_jpeg_bytes[2:]
        img_file = tmp_dir / "nosoi.jpg"
        img_file.write_bytes(broken)

        success, repaired_path = repairer.repair(img_file)
        assert repaired_path is not None
        repaired_data = repaired_path.read_bytes()
        assert repaired_data[:2] == b"\xFF\xD8"

    def test_repair_png_appends_iend(self, tmp_dir, minimal_png_bytes):
        repairer = ImageRepairer(str(tmp_dir))
        truncated = minimal_png_bytes[:-12]  # Remove IEND chunk
        img_file = tmp_dir / "broken.png"
        img_file.write_bytes(truncated)

        success, repaired_path = repairer.repair(img_file)
        assert repaired_path is not None
        repaired_data = repaired_path.read_bytes()
        assert b"IEND" in repaired_data

    def test_repair_unsupported_format(self, tmp_dir):
        repairer = ImageRepairer(str(tmp_dir))
        bmp_file = tmp_dir / "test.bmp"
        bmp_file.write_bytes(b"\x42\x4D" + b"\x00" * 50)
        success, path = repairer.repair(bmp_file)
        assert not success
        assert path is None

    def test_repair_unreadable_file(self, tmp_dir):
        repairer = ImageRepairer(str(tmp_dir))
        fake_path = tmp_dir / "nonexistent.jpg"
        success, path = repairer.repair(fake_path)
        assert not success


# -----------------------------------------------------------------------
# Video repair
# -----------------------------------------------------------------------

class TestVideoRepairer:
    def test_repair_avi_fixes_riff_size(self, tmp_dir):
        repairer = VideoRepairer(str(tmp_dir))
        import struct
        # Create AVI with wrong RIFF size
        body = b"AVI " + b"\x00" * 100
        wrong_size = 50  # Intentionally wrong
        data = b"RIFF" + struct.pack("<I", wrong_size) + body
        avi_file = tmp_dir / "broken.avi"
        avi_file.write_bytes(data)

        success, repaired_path = repairer.repair(avi_file)
        assert repaired_path is not None
        repaired_data = repaired_path.read_bytes()
        fixed_size = struct.unpack_from("<I", repaired_data, 4)[0]
        assert fixed_size == len(data) - 8

    def test_repair_unsupported_format(self, tmp_dir):
        repairer = VideoRepairer(str(tmp_dir))
        mkv_file = tmp_dir / "video.mkv"
        mkv_file.write_bytes(b"\x1A\x45\xDF\xA3" + b"\x00" * 20)
        success, path = repairer.repair(mkv_file)
        assert not success
        assert path is None

    def test_repair_unreadable_file(self, tmp_dir):
        repairer = VideoRepairer(str(tmp_dir))
        success, path = repairer.repair(tmp_dir / "missing.avi")
        assert not success


# -----------------------------------------------------------------------
# Document repair
# -----------------------------------------------------------------------

class TestDocumentRepairer:
    def test_repair_pdf_appends_eof(self, tmp_dir, minimal_pdf_bytes):
        repairer = DocumentRepairer(str(tmp_dir))
        broken = minimal_pdf_bytes.replace(b"%%EOF\n", b"")
        pdf_file = tmp_dir / "broken.pdf"
        pdf_file.write_bytes(broken)

        success, repaired_path = repairer.repair(pdf_file)
        assert repaired_path is not None
        repaired_data = repaired_path.read_bytes()
        assert b"%%EOF" in repaired_data

    def test_repair_pdf_prepends_header(self, tmp_dir):
        repairer = DocumentRepairer(str(tmp_dir))
        broken = b"garbage data before PDF\n%PDF-1.4\nsome content\n%%EOF\n"
        pdf_file = tmp_dir / "offset.pdf"
        pdf_file.write_bytes(broken)

        success, repaired_path = repairer.repair(pdf_file)
        assert repaired_path is not None
        repaired_data = repaired_path.read_bytes()
        assert repaired_data[:5] == b"%PDF-"

    def test_repair_docx_valid_zip(self, tmp_dir):
        """A valid DOCX (ZIP) should be accepted as-is."""
        import io, zipfile
        repairer = DocumentRepairer(str(tmp_dir))
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("[Content_Types].xml", "<Types/>")
            zf.writestr("word/document.xml", "<w:document/>")
        docx_file = tmp_dir / "valid.docx"
        docx_file.write_bytes(buf.getvalue())

        success, repaired_path = repairer.repair(docx_file)
        assert success
        assert repaired_path is not None

    def test_repair_unsupported_format(self, tmp_dir):
        repairer = DocumentRepairer(str(tmp_dir))
        txt_file = tmp_dir / "readme.txt"
        txt_file.write_bytes(b"Hello world")
        success, path = repairer.repair(txt_file)
        assert not success
        assert path is None
