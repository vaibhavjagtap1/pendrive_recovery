"""
USB / storage device handler.

Detects removable storage devices and provides read-only access helpers.
Works on Linux (via /proc/partitions and /sys/block) and Windows (via
win32 APIs through ctypes), with a graceful fallback for plain files.
"""

import logging
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import List, NamedTuple, Optional

logger = logging.getLogger(__name__)


class DeviceInfo(NamedTuple):
    name: str        # e.g. "sdb", "E:"
    path: str        # e.g. "/dev/sdb", "\\\\.\E:"
    size_bytes: int
    removable: bool
    label: Optional[str]


class USBHandler:
    """
    Detects USB/removable storage devices and mounts them read-only.
    """

    # ------------------------------------------------------------------
    # Device discovery
    # ------------------------------------------------------------------

    def list_devices(self) -> List[DeviceInfo]:
        """
        List all block devices (filtered to removable ones where possible).

        Returns:
            List of DeviceInfo objects.
        """
        system = platform.system()
        if system == "Linux":
            return self._list_linux()
        elif system == "Windows":
            return self._list_windows()
        elif system == "Darwin":
            return self._list_macos()
        logger.warning("Unsupported platform: %s", system)
        return []

    def _list_linux(self) -> List[DeviceInfo]:
        """List block devices on Linux using /sys/block."""
        devices = []
        sys_block = Path("/sys/block")
        if not sys_block.exists():
            return devices

        for block_dev in sys_block.iterdir():
            name = block_dev.name
            if name.startswith("loop") or name.startswith("ram"):
                continue

            # Check removable flag
            removable_path = block_dev / "removable"
            removable = False
            if removable_path.exists():
                try:
                    removable = removable_path.read_text().strip() == "1"
                except OSError:
                    pass

            # Get size
            size_file = block_dev / "size"
            size_bytes = 0
            if size_file.exists():
                try:
                    size_sectors = int(size_file.read_text().strip())
                    size_bytes = size_sectors * 512
                except (ValueError, OSError):
                    pass

            dev_path = f"/dev/{name}"
            devices.append(
                DeviceInfo(
                    name=name,
                    path=dev_path,
                    size_bytes=size_bytes,
                    removable=removable,
                    label=None,
                )
            )
        return devices

    def _list_windows(self) -> List[DeviceInfo]:
        """List drives on Windows using ctypes."""
        import ctypes
        import string

        devices = []
        bitmask = ctypes.windll.kernel32.GetLogicalDrives()
        for letter in string.ascii_uppercase:
            if bitmask & 1:
                drive = f"{letter}:\\"
                drive_type = ctypes.windll.kernel32.GetDriveTypeW(drive)
                # DRIVE_REMOVABLE = 2
                removable = drive_type == 2
                # Get free/total space
                free = ctypes.c_ulonglong(0)
                total = ctypes.c_ulonglong(0)
                avail = ctypes.c_ulonglong(0)
                ctypes.windll.kernel32.GetDiskFreeSpaceExW(
                    drive,
                    ctypes.byref(avail),
                    ctypes.byref(total),
                    ctypes.byref(free),
                )
                unc_path = f"\\\\.\\{letter}:"
                devices.append(
                    DeviceInfo(
                        name=f"{letter}:",
                        path=unc_path,
                        size_bytes=total.value,
                        removable=removable,
                        label=None,
                    )
                )
            bitmask >>= 1
        return devices

    def _list_macos(self) -> List[DeviceInfo]:
        """List block devices on macOS using diskutil."""
        devices = []
        try:
            result = subprocess.run(
                ["diskutil", "list", "-plist"],
                capture_output=True, timeout=10,
            )
            if result.returncode != 0:
                return devices
            import plistlib
            plist = plistlib.loads(result.stdout)
            for disk in plist.get("AllDisks", []):
                dev_path = f"/dev/{disk}"
                size = 0
                try:
                    info = subprocess.run(
                        ["diskutil", "info", "-plist", disk],
                        capture_output=True, timeout=10,
                    )
                    info_plist = plistlib.loads(info.stdout)
                    size = info_plist.get("TotalSize", 0)
                    removable = info_plist.get("Ejectable", False)
                except Exception:
                    removable = False
                devices.append(
                    DeviceInfo(
                        name=disk,
                        path=dev_path,
                        size_bytes=size,
                        removable=removable,
                        label=None,
                    )
                )
        except Exception as exc:
            logger.error("macOS device listing failed: %s", exc)
        return devices

    # ------------------------------------------------------------------
    # Read-only access
    # ------------------------------------------------------------------

    def open_readonly(self, device_path: str):
        """
        Open a device or image file for read-only access.

        Args:
            device_path: Path to device (e.g. /dev/sdb) or image file.

        Returns:
            Open file handle (binary read mode).

        Raises:
            OSError: If the device cannot be opened.
        """
        path = Path(device_path)
        if not path.exists():
            raise OSError(f"Device not found: {device_path}")
        return open(device_path, "rb")

    def mount_readonly(self, device_path: str, mount_point: str) -> bool:
        """
        Attempt to mount a device read-only (Linux only).

        Args:
            device_path: Path to the block device.
            mount_point: Directory to mount on.

        Returns:
            True if mounted successfully.
        """
        if platform.system() != "Linux":
            logger.warning("mount_readonly is only supported on Linux")
            return False

        Path(mount_point).mkdir(parents=True, exist_ok=True)
        result = subprocess.run(
            ["mount", "-o", "ro", device_path, mount_point],
            capture_output=True,
        )
        if result.returncode == 0:
            logger.info("Mounted %s at %s (read-only)", device_path, mount_point)
            return True
        logger.error(
            "Mount failed: %s",
            result.stderr.decode(errors="replace"),
        )
        return False

    def unmount(self, mount_point: str) -> bool:
        """Unmount a previously mounted device (Linux only)."""
        if platform.system() != "Linux":
            return False
        result = subprocess.run(["umount", mount_point], capture_output=True)
        return result.returncode == 0
