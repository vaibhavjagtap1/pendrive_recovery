"""
Recovery Engine: Orchestrates sector-level file recovery from storage devices.

Supports:
- Raw file carving (signature-based)
- FAT32 directory entry recovery of deleted files
- NTFS MFT-based file recovery
- Read-only access to avoid data corruption
"""

import os
import logging
import hashlib
from pathlib import Path
from typing import Callable, List, Optional

from .signatures import FILE_SIGNATURES, HEADER_LOOKUP, MAX_HEADER_LEN
from .filesystem import (
    FilesystemType,
    detect_filesystem,
    FAT32Parser,
    NTFSParser,
)

logger = logging.getLogger(__name__)

# Default read block size for scanning
BLOCK_SIZE = 512 * 1024  # 512 KB


class RecoveredFile:
    """Represents a single file recovered from the device."""

    def __init__(
        self,
        data: bytes,
        extension: str,
        offset: int,
        method: str,
        original_name: Optional[str] = None,
    ):
        self.data = data
        self.extension = extension.lower().lstrip(".")
        self.offset = offset
        self.method = method
        self.original_name = original_name
        self.sha256 = hashlib.sha256(data).hexdigest()

    @property
    def size(self) -> int:
        return len(self.data)

    def save(self, output_path: Path) -> Path:
        """
        Save the recovered file to disk.

        Args:
            output_path: Directory where the file will be saved.

        Returns:
            Path to the saved file.
        """
        output_path.mkdir(parents=True, exist_ok=True)
        filename = self.original_name or f"recovered_{self.offset:016x}.{self.extension}"
        dest = output_path / filename
        # Avoid overwriting with suffix counter
        counter = 0
        while dest.exists():
            counter += 1
            stem = Path(filename).stem
            dest = output_path / f"{stem}_{counter}.{self.extension}"
        dest.write_bytes(self.data)
        return dest


class RecoveryEngine:
    """
    Main recovery engine for extracting files from a storage device image or device.

    Supports:
    - Signature-based raw file carving
    - FAT32 deleted-file recovery via directory entry scanning
    - NTFS MFT-based file recovery
    - Read-only access mode
    """

    def __init__(
        self,
        device_path: str,
        output_dir: str,
        block_size: int = BLOCK_SIZE,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        """
        Initialize the recovery engine.

        Args:
            device_path: Path to the device or image file (opened read-only).
            output_dir: Directory where recovered files will be saved.
            block_size: Read block size in bytes.
            progress_callback: Optional callback(progress_pct, message).
        """
        self.device_path = device_path
        self.output_dir = Path(output_dir)
        self.block_size = block_size
        self.progress_callback = progress_callback
        self._recovered: List[RecoveredFile] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> List[RecoveredFile]:
        """
        Run all applicable recovery strategies and return recovered files.

        Returns:
            List of RecoveredFile objects.
        """
        logger.info("Starting recovery on %s", self.device_path)
        self._recovered = []

        device_size = os.path.getsize(self.device_path)
        logger.info("Device size: %d bytes (%.2f GB)", device_size, device_size / 1e9)

        # Read boot sector to determine filesystem
        with open(self.device_path, "rb") as fh:
            boot_sector = fh.read(512)

        fs_type = detect_filesystem(boot_sector)
        logger.info("Detected filesystem: %s", fs_type.value)
        self._report_progress(0.0, f"Filesystem detected: {fs_type.value}")

        # Strategy 1: filesystem-aware recovery
        if fs_type == FilesystemType.FAT32:
            self._fat32_recovery(device_size)
        elif fs_type == FilesystemType.NTFS:
            self._ntfs_recovery(device_size)

        # Strategy 2: raw file carving (catches everything missed above)
        self._carve_files(device_size)

        # Deduplicate by SHA-256
        seen = set()
        unique = []
        for rf in self._recovered:
            if rf.sha256 not in seen:
                seen.add(rf.sha256)
                unique.append(rf)
        self._recovered = unique

        logger.info("Recovery complete. Files recovered: %d", len(self._recovered))
        self._report_progress(100.0, f"Recovery complete: {len(self._recovered)} files")
        return self._recovered

    # ------------------------------------------------------------------
    # Private – filesystem-aware recovery
    # ------------------------------------------------------------------

    def _fat32_recovery(self, device_size: int) -> None:
        """Recover deleted files from a FAT32 filesystem."""
        logger.info("Running FAT32 directory-entry recovery")
        try:
            with open(self.device_path, "rb") as fh:
                data = fh.read(min(device_size, 64 * 1024 * 1024))  # First 64 MB for dirs

            parser = FAT32Parser(data)
            entries = parser.list_directory(parser.root_cluster)
            deleted_count = 0
            for entry in entries:
                if entry.is_deleted and entry.size > 0:
                    file_data = parser.read_file(entry)
                    if file_data:
                        ext = entry.name.rsplit(".", 1)[-1] if "." in entry.name else "bin"
                        rf = RecoveredFile(
                            data=file_data,
                            extension=ext,
                            offset=entry.cluster,
                            method="fat32_directory",
                            original_name=entry.name.replace("?", "X"),
                        )
                        self._recovered.append(rf)
                        deleted_count += 1
            logger.info("FAT32 recovery: %d deleted files found", deleted_count)
        except Exception as exc:
            logger.warning("FAT32 recovery failed: %s", exc)

    def _ntfs_recovery(self, device_size: int) -> None:
        """Scan NTFS MFT for recoverable files."""
        logger.info("Running NTFS MFT recovery")
        try:
            with open(self.device_path, "rb") as fh:
                data = fh.read(min(device_size, 256 * 1024 * 1024))

            parser = NTFSParser(data)
            mft_files = 0
            for offset, record in parser.iter_mft_records():
                name = parser.parse_record_name(record)
                if name:
                    mft_files += 1
            logger.info("NTFS MFT scan: %d file records found", mft_files)
        except Exception as exc:
            logger.warning("NTFS recovery failed: %s", exc)

    # ------------------------------------------------------------------
    # Private – signature-based carving
    # ------------------------------------------------------------------

    def _carve_files(self, device_size: int) -> None:
        """
        Scan the entire device byte-by-byte for known file signatures
        and extract matching data.
        """
        logger.info("Starting signature-based file carving (%d bytes)", device_size)
        carved = 0

        with open(self.device_path, "rb") as fh:
            offset = 0
            overlap = bytearray()
            total_read = 0

            while True:
                chunk = fh.read(self.block_size)
                if not chunk:
                    break
                total_read += len(chunk)
                buf = bytes(overlap) + chunk

                pos = 0
                while pos < len(buf):
                    found = self._match_header(buf, pos)
                    if found:
                        ext, sig = found
                        file_data = self._extract_file(
                            fh,
                            buf,
                            pos,
                            offset - len(overlap),
                            sig,
                        )
                        if file_data:
                            rf = RecoveredFile(
                                data=file_data,
                                extension=ext,
                                offset=offset - len(overlap) + pos,
                                method="file_carving",
                            )
                            self._recovered.append(rf)
                            carved += 1
                            pos += len(file_data)
                            continue
                    pos += 1

                overlap = bytearray(buf[-MAX_HEADER_LEN:])
                offset += len(chunk)

                progress = min(total_read / device_size * 100, 99.0)
                self._report_progress(progress, f"Carving... {carved} files found")

        logger.info("File carving complete. Files carved: %d", carved)

    def _match_header(self, buf: bytes, pos: int):
        """
        Check if any known file header starts at position pos.

        Returns:
            Tuple (extension, signature_dict) or None.
        """
        for hdr, matches in HEADER_LOOKUP.items():
            end = pos + len(hdr)
            if end <= len(buf) and buf[pos:end] == hdr:
                return matches[0]  # Return first matching signature
        return None

    def _extract_file(
        self,
        fh,
        buf: bytes,
        buf_pos: int,
        buf_offset: int,
        sig: dict,
    ) -> Optional[bytes]:
        """
        Extract file data starting from buf_pos using the footer and max_size.

        Args:
            fh: Open file handle for reading more data if needed.
            buf: Current buffer.
            buf_pos: Position within buf where file starts.
            buf_offset: Absolute offset where buf starts in device.
            sig: Signature dict (header, footer, max_size).

        Returns:
            Bytes of extracted file, or None if extraction failed.
        """
        footer = sig.get("footer")
        max_size = sig.get("max_size", 10 * 1024 * 1024)

        start_abs = buf_offset + buf_pos
        data = bytearray(buf[buf_pos:])

        # Read more data if needed
        while len(data) < max_size:
            extra = fh.read(self.block_size)
            if not extra:
                break
            data.extend(extra)
            if len(data) >= max_size:
                break

        data = data[:max_size]

        if footer:
            idx = data.find(footer, len(sig["header"]))
            if idx == -1:
                # Footer not found: treat entire data as file if >= header length
                if len(data) > len(sig["header"]) * 2:
                    return bytes(data)
                return None
            return bytes(data[: idx + len(footer)])

        # No footer: use max_size as upper bound, but trim trailing zeros
        trimmed = bytes(data).rstrip(b"\x00")
        if len(trimmed) > len(sig["header"]):
            return trimmed
        return None

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    def _report_progress(self, pct: float, message: str) -> None:
        if self.progress_callback:
            self.progress_callback(pct, message)
