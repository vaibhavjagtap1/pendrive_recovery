"""
Document repair module for PDF and DOCX files.

Attempts to repair corrupted documents by:
1. Reconstructing PDF cross-reference tables (xref)
2. Patching PDF headers
3. Re-saving DOCX as a valid ZIP archive (since DOCX = ZIP)
"""

import io
import logging
import struct
import zipfile
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# PDF magic bytes
PDF_HEADER = b"%PDF-"
PDF_EOF = b"%%EOF"

# Office Open XML (DOCX/XLSX/PPTX) is a ZIP file
OOXML_SIGNATURE = b"\x50\x4B\x03\x04"


class DocumentRepairer:
    """
    Attempts to detect and repair common document file corruption.

    Supports PDF and DOCX formats.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.repaired_dir = self.output_dir / "Repaired_Files" / "documents"
        self.repaired_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def repair(self, doc_path: Path) -> Tuple[bool, Optional[Path]]:
        """
        Attempt to repair a document file.

        Args:
            doc_path: Path to the potentially corrupted document.

        Returns:
            Tuple of (success, repaired_path). repaired_path is None on failure.
        """
        suffix = doc_path.suffix.lower()
        if suffix == ".pdf":
            return self._repair_pdf(doc_path)
        elif suffix in (".docx", ".xlsx", ".pptx"):
            return self._repair_ooxml(doc_path)
        else:
            logger.debug("Unsupported document format: %s", suffix)
            return False, None

    # ------------------------------------------------------------------
    # PDF repair
    # ------------------------------------------------------------------

    def _repair_pdf(self, path: Path) -> Tuple[bool, Optional[Path]]:
        """Attempt byte-level PDF repair."""
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.error("Cannot read %s: %s", path, exc)
            return False, None

        repaired, changed = self._fix_pdf_bytes(data)
        repaired_path = self._save_repaired(path, repaired, "pdf")
        return changed, repaired_path

    def _fix_pdf_bytes(self, data: bytes) -> Tuple[bytes, bool]:
        """
        Apply byte-level PDF repairs:
        - Ensure %PDF- header at start
        - Ensure %%EOF at end
        - Attempt to patch broken xref offset
        """
        changed = False
        buf = bytearray(data)

        # Ensure PDF header
        if not buf[:5] == bytearray(PDF_HEADER):
            # Find where the header is
            idx = data.find(PDF_HEADER)
            if idx > 0:
                buf = bytearray(buf[idx:])
                changed = True
                logger.debug("Stripped %d leading bytes before PDF header", idx)
            else:
                # Prepend minimal header
                buf = bytearray(b"%PDF-1.4\n") + buf
                changed = True
                logger.debug("Prepended minimal PDF header")

        # Ensure %%EOF
        if not buf[-6:].rstrip(b"\n\r ").endswith(bytearray(PDF_EOF)):
            buf.extend(b"\n%%EOF\n")
            changed = True
            logger.debug("Appended %%EOF marker")

        # Attempt to fix startxref offset
        startxref_idx = bytes(buf).rfind(b"startxref")
        if startxref_idx != -1:
            # Find xref table position
            xref_idx = bytes(buf).find(b"\nxref")
            if xref_idx == -1:
                xref_idx = bytes(buf).find(b"\r\nxref")
            if xref_idx != -1:
                # Update startxref value
                sr_end = startxref_idx + len(b"startxref")
                # Find the end of the line
                line_end = bytes(buf).find(b"\n", sr_end)
                if line_end == -1:
                    line_end = len(buf)
                new_val = str(xref_idx + 1).encode()
                buf[sr_end:line_end] = b"\n" + new_val
                changed = True
                logger.debug("Updated startxref offset to %d", xref_idx + 1)

        return bytes(buf), changed

    # ------------------------------------------------------------------
    # OOXML (DOCX / XLSX / PPTX) repair
    # ------------------------------------------------------------------

    def _repair_ooxml(self, path: Path) -> Tuple[bool, Optional[Path]]:
        """
        Attempt to repair an OOXML document (DOCX/XLSX/PPTX).

        Strategy:
        1. Try to open as a ZIP; if it fails, try to salvage entries.
        2. Re-pack salvaged entries into a new ZIP.
        """
        try:
            data = path.read_bytes()
        except OSError as exc:
            logger.error("Cannot read %s: %s", path, exc)
            return False, None

        ext = path.suffix.lower().lstrip(".")

        # Try to read as ZIP
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                # File is valid – just copy it
                repaired_path = self._save_repaired(path, data, ext)
                logger.info("OOXML file is valid ZIP: %s", path.name)
                return True, repaired_path
        except zipfile.BadZipFile:
            pass

        # Attempt salvage: find local file headers and reconstruct
        logger.info("Attempting OOXML ZIP salvage for %s", path.name)
        salvaged = self._salvage_zip(data)
        if salvaged:
            repaired_path = self._save_repaired(path, salvaged, ext)
            return True, repaired_path

        return False, None

    def _salvage_zip(self, data: bytes) -> Optional[bytes]:
        """
        Attempt to reconstruct a ZIP archive from raw bytes by scanning
        for local file headers (PK\x03\x04).
        """
        LOCAL_HEADER = b"\x50\x4B\x03\x04"
        out_buf = io.BytesIO()

        try:
            entries = []
            pos = 0
            while pos < len(data):
                idx = data.find(LOCAL_HEADER, pos)
                if idx == -1:
                    break

                try:
                    # Parse local file header
                    if idx + 30 > len(data):
                        break
                    compression = struct.unpack_from("<H", data, idx + 8)[0]
                    crc32 = struct.unpack_from("<I", data, idx + 14)[0]
                    compressed_size = struct.unpack_from("<I", data, idx + 18)[0]
                    uncompressed_size = struct.unpack_from("<I", data, idx + 22)[0]
                    fname_len = struct.unpack_from("<H", data, idx + 26)[0]
                    extra_len = struct.unpack_from("<H", data, idx + 28)[0]

                    header_size = 30 + fname_len + extra_len
                    fname = data[idx + 30 : idx + 30 + fname_len].decode("utf-8", errors="replace")

                    entry_end = idx + header_size + compressed_size
                    if entry_end > len(data):
                        entry_end = len(data)

                    entry_data = data[idx : entry_end]
                    entries.append((fname, entry_data))
                    pos = entry_end
                except struct.error:
                    pos = idx + 1
                    continue

            if not entries:
                return None

            with zipfile.ZipFile(out_buf, "w", zipfile.ZIP_DEFLATED) as zf_out:
                for fname, raw_entry in entries:
                    # Extract compressed content
                    try:
                        tmp = zipfile.ZipFile(io.BytesIO(raw_entry))
                        content = tmp.read(tmp.namelist()[0])
                        zf_out.writestr(fname, content)
                    except Exception:
                        zf_out.writestr(fname, raw_entry)

            return out_buf.getvalue()
        except Exception as exc:
            logger.error("ZIP salvage failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _save_repaired(self, original_path: Path, data: bytes, ext: str) -> Path:
        dest = self.repaired_dir / f"{original_path.stem}_repaired.{ext}"
        counter = 0
        while dest.exists():
            counter += 1
            dest = self.repaired_dir / f"{original_path.stem}_repaired_{counter}.{ext}"
        dest.write_bytes(data)
        logger.info("Saved repaired document: %s", dest)
        return dest
