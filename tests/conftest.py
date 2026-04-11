"""Shared pytest fixtures and configuration."""

import os
import struct
import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def tmp_dir():
    """Return a temporary directory that is cleaned up after the test."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def minimal_jpeg_bytes():
    """Return minimal valid JPEG bytes (SOI + EOI)."""
    return b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00\xFF\xD9"


@pytest.fixture
def minimal_png_bytes():
    """Return the minimal valid PNG bytes (just signature + IEND)."""
    png_sig = b"\x89PNG\r\n\x1a\n"
    # IHDR chunk: 13 bytes, width=1, height=1, bit_depth=8, color_type=2
    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    import zlib
    ihdr_crc = zlib.crc32(b"IHDR" + ihdr_data) & 0xFFFFFFFF
    ihdr_chunk = struct.pack(">I", 13) + b"IHDR" + ihdr_data + struct.pack(">I", ihdr_crc)
    # IDAT: empty deflated row
    raw_row = b"\x00\x00\x00\x00"  # filter byte + 1 RGB pixel
    idat_data = zlib.compress(raw_row)
    idat_crc = zlib.crc32(b"IDAT" + idat_data) & 0xFFFFFFFF
    idat_chunk = struct.pack(">I", len(idat_data)) + b"IDAT" + idat_data + struct.pack(">I", idat_crc)
    # IEND
    iend_crc = zlib.crc32(b"IEND") & 0xFFFFFFFF
    iend_chunk = struct.pack(">I", 0) + b"IEND" + struct.pack(">I", iend_crc)
    return png_sig + ihdr_chunk + idat_chunk + iend_chunk


@pytest.fixture
def minimal_pdf_bytes():
    """Return minimal valid PDF bytes."""
    return (
        b"%PDF-1.4\n"
        b"1 0 obj\n<< /Type /Catalog >>\nendobj\n"
        b"xref\n0 2\n0000000000 65535 f \n0000000009 00000 n \n"
        b"trailer\n<< /Size 2 /Root 1 0 R >>\n"
        b"startxref\n9\n%%EOF\n"
    )
