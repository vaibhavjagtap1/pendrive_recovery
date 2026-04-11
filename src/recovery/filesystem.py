"""
Filesystem parser supporting FAT32, exFAT, and NTFS layouts.

Provides utilities to detect filesystem type and read directory entries
to aid recovery of logically deleted files.
"""

import struct
import logging
from enum import Enum
from typing import Optional, List, NamedTuple

logger = logging.getLogger(__name__)


class FilesystemType(Enum):
    FAT32 = "FAT32"
    EXFAT = "exFAT"
    NTFS = "NTFS"
    UNKNOWN = "UNKNOWN"


class DirectoryEntry(NamedTuple):
    name: str
    size: int
    cluster: int
    is_deleted: bool
    attributes: int


def detect_filesystem(data: bytes) -> FilesystemType:
    """
    Detect the filesystem type from the first 512 bytes (boot sector).

    Args:
        data: At least 512 bytes from the start of the volume.

    Returns:
        FilesystemType enum value.
    """
    if len(data) < 512:
        return FilesystemType.UNKNOWN

    # NTFS: OEM ID at offset 3
    if data[3:11] == b"NTFS    ":
        return FilesystemType.NTFS

    # exFAT: OEM ID at offset 3
    if data[3:11] == b"EXFAT   ":
        return FilesystemType.EXFAT

    # FAT32: check filesystem type string at offset 82
    if data[82:90] in (b"FAT32   ", b"FAT16   ", b"FAT     "):
        return FilesystemType.FAT32

    # FAT32 heuristic: valid bytes-per-sector and sectors-per-cluster
    bytes_per_sector = struct.unpack_from("<H", data, 11)[0]
    secs_per_cluster = data[13]
    if bytes_per_sector in (512, 1024, 2048, 4096) and secs_per_cluster in (
        1, 2, 4, 8, 16, 32, 64, 128,
    ):
        return FilesystemType.FAT32

    return FilesystemType.UNKNOWN


class FAT32Parser:
    """
    Parses a FAT32 filesystem image to enumerate (including deleted) files.
    """

    DIR_ENTRY_SIZE = 32

    def __init__(self, data: bytes):
        self.data = data
        self._parse_bpb()

    def _parse_bpb(self):
        """Parse the BIOS Parameter Block from the boot sector."""
        d = self.data
        self.bytes_per_sector = struct.unpack_from("<H", d, 11)[0]
        self.secs_per_cluster = d[13]
        self.reserved_sectors = struct.unpack_from("<H", d, 14)[0]
        self.num_fats = d[16]
        self.total_sectors_16 = struct.unpack_from("<H", d, 19)[0]
        self.sectors_per_fat = struct.unpack_from("<I", d, 36)[0]
        self.root_cluster = struct.unpack_from("<I", d, 44)[0]
        self.cluster_size = self.bytes_per_sector * self.secs_per_cluster
        self.fat_start = self.reserved_sectors * self.bytes_per_sector
        self.data_start = (
            self.fat_start + self.num_fats * self.sectors_per_fat * self.bytes_per_sector
        )

    def cluster_offset(self, cluster: int) -> int:
        """Return byte offset of a cluster."""
        return self.data_start + (cluster - 2) * self.cluster_size

    def _read_cluster_chain(self, start_cluster: int) -> bytes:
        """Follow FAT chain and read all cluster data."""
        result = bytearray()
        cluster = start_cluster
        visited = set()
        while cluster < 0x0FFFFFF8 and cluster not in visited:
            visited.add(cluster)
            offset = self.cluster_offset(cluster)
            result.extend(self.data[offset : offset + self.cluster_size])
            fat_offset = self.fat_start + cluster * 4
            if fat_offset + 4 > len(self.data):
                break
            cluster = struct.unpack_from("<I", self.data, fat_offset)[0] & 0x0FFFFFFF
        return bytes(result)

    def list_directory(self, cluster: int) -> List[DirectoryEntry]:
        """
        List all entries (including deleted) in a FAT32 directory cluster.

        Args:
            cluster: Starting cluster of the directory.

        Returns:
            List of DirectoryEntry objects.
        """
        entries = []
        dir_data = self._read_cluster_chain(cluster)
        offset = 0
        while offset + self.DIR_ENTRY_SIZE <= len(dir_data):
            raw = dir_data[offset : offset + self.DIR_ENTRY_SIZE]
            first_byte = raw[0]

            if first_byte == 0x00:
                # No more entries
                break

            is_deleted = first_byte == 0xE5
            if first_byte == 0x2E:
                # Dot entry, skip
                offset += self.DIR_ENTRY_SIZE
                continue

            attr = raw[11]
            if attr == 0x0F:
                # Long filename entry, skip
                offset += self.DIR_ENTRY_SIZE
                continue

            # Short filename: 8 chars name + 3 chars extension
            raw_name = raw[0:8].rstrip(b" ").decode("ascii", errors="replace")
            raw_ext = raw[8:11].rstrip(b" ").decode("ascii", errors="replace")
            if is_deleted:
                raw_name = "?" + raw_name[1:]
            name = f"{raw_name}.{raw_ext}" if raw_ext else raw_name

            cluster_hi = struct.unpack_from("<H", raw, 20)[0]
            cluster_lo = struct.unpack_from("<H", raw, 26)[0]
            start_cluster = (cluster_hi << 16) | cluster_lo
            file_size = struct.unpack_from("<I", raw, 28)[0]

            entries.append(
                DirectoryEntry(
                    name=name,
                    size=file_size,
                    cluster=start_cluster,
                    is_deleted=is_deleted,
                    attributes=attr,
                )
            )
            offset += self.DIR_ENTRY_SIZE
        return entries

    def read_file(self, entry: DirectoryEntry) -> Optional[bytes]:
        """Read file data for a directory entry."""
        if entry.cluster < 2:
            return None
        data = self._read_cluster_chain(entry.cluster)
        return data[: entry.size] if entry.size > 0 else data


class NTFSParser:
    """
    Basic NTFS parser to locate MFT entries and recover file data.
    """

    MFT_RECORD_SIZE = 1024
    SIGNATURE = b"FILE"

    def __init__(self, data: bytes):
        self.data = data
        self._parse_boot()

    def _parse_boot(self):
        d = self.data
        self.bytes_per_sector = struct.unpack_from("<H", d, 11)[0]
        self.secs_per_cluster = d[13]
        self.cluster_size = self.bytes_per_sector * self.secs_per_cluster
        mft_lcn = struct.unpack_from("<Q", d, 48)[0]
        self.mft_offset = mft_lcn * self.cluster_size

    def iter_mft_records(self):
        """Iterate over all MFT records in the image."""
        offset = self.mft_offset
        while offset + self.MFT_RECORD_SIZE <= len(self.data):
            record = self.data[offset : offset + self.MFT_RECORD_SIZE]
            if record[:4] == self.SIGNATURE:
                yield offset, record
            offset += self.MFT_RECORD_SIZE

    def parse_record_name(self, record: bytes) -> Optional[str]:
        """
        Extract file name from an MFT $FILE_NAME attribute (type 0x30).
        """
        try:
            attrs_offset = struct.unpack_from("<H", record, 20)[0]
            offset = attrs_offset
            while offset + 4 < len(record):
                attr_type = struct.unpack_from("<I", record, offset)[0]
                if attr_type == 0xFFFFFFFF:
                    break
                attr_len = struct.unpack_from("<I", record, offset + 4)[0]
                if attr_len == 0:
                    break
                if attr_type == 0x30:  # $FILE_NAME
                    resident = record[offset + 8]
                    if resident == 0:
                        content_offset = struct.unpack_from("<H", record, offset + 20)[0]
                        content = record[offset + content_offset :]
                        name_len = content[64]
                        name = content[66 : 66 + name_len * 2].decode("utf-16-le", errors="replace")
                        return name
                offset += attr_len
        except (struct.error, IndexError):
            pass
        return None
