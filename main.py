#!/usr/bin/env python3
"""
Pendrive Recovery – CLI entry point.

Usage:
    python main.py --device /dev/sdb --output /tmp/recovered
    python main.py --device /dev/sdb --output /tmp/recovered --no-repair --no-ai
    python main.py --list-devices
"""

import argparse
import logging
import sys
from pathlib import Path

from src.utils.logger import setup_logger
from src.utils.usb_handler import USBHandler


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline AI-Based Pendrive Recovery & Face Organization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--device", "-d",
        metavar="PATH",
        help="Path to device or disk image (e.g. /dev/sdb or disk.img)",
    )
    parser.add_argument(
        "--output", "-o",
        metavar="DIR",
        default="./recovered_output",
        help="Output directory for recovered files (default: ./recovered_output)",
    )
    parser.add_argument(
        "--list-devices",
        action="store_true",
        help="List detected storage devices and exit",
    )
    parser.add_argument(
        "--no-repair",
        action="store_true",
        help="Skip the file repair step",
    )
    parser.add_argument(
        "--no-ai",
        action="store_true",
        help="Skip AI face detection and clustering",
    )
    parser.add_argument(
        "--dbscan-eps",
        type=float,
        default=0.5,
        metavar="EPS",
        help="DBSCAN epsilon for face clustering (default: 0.5)",
    )
    parser.add_argument(
        "--log-file",
        metavar="FILE",
        help="Write logs to this file in addition to stdout",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose (DEBUG) logging",
    )

    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    log_level = logging.DEBUG if args.verbose else logging.INFO
    setup_logger(level=log_level, log_file=args.log_file)
    logger = logging.getLogger("pendrive_recovery")

    if args.list_devices:
        handler = USBHandler()
        devices = handler.list_devices()
        if not devices:
            print("No block devices found.")
            return 0
        print(f"{'Name':<12} {'Path':<20} {'Size (GB)':>10}  {'Removable':<10}")
        print("-" * 60)
        for dev in devices:
            size_gb = dev.size_bytes / 1e9 if dev.size_bytes else 0
            print(
                f"{dev.name:<12} {dev.path:<20} {size_gb:>10.2f}  "
                f"{'Yes' if dev.removable else 'No':<10}"
            )
        return 0

    if not args.device:
        print("Error: --device is required. Use --list-devices to see available devices.")
        return 1

    device_path = args.device
    output_dir = args.output

    if not Path(device_path).exists():
        print(f"Error: Device or image not found: {device_path}")
        return 1

    logger.info("Starting recovery: device=%s output=%s", device_path, output_dir)

    from src.pipeline import RecoveryPipeline

    def progress(pct: float, msg: str) -> None:
        bar_len = 40
        filled = int(bar_len * pct / 100)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"\r[{bar}] {pct:5.1f}%  {msg:<50}", end="", flush=True)
        if pct >= 100:
            print()

    pipeline = RecoveryPipeline(
        device_path=device_path,
        output_dir=output_dir,
        log_file=args.log_file,
        enable_repair=not args.no_repair,
        enable_ai=not args.no_ai,
        dbscan_eps=args.dbscan_eps,
        progress_callback=progress,
    )

    try:
        report = pipeline.run()
    except PermissionError:
        print(
            f"\nPermission denied reading {device_path}. "
            "Try running with sudo or as Administrator."
        )
        return 1
    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1

    print("\n" + "=" * 60)
    print("  RECOVERY COMPLETE")
    print("=" * 60)
    print(f"  Files recovered : {report.get('recovery', {}).get('files_recovered', 0)}")
    print(f"  Files repaired  : {report.get('repair', {}).get('succeeded', 0)}")
    print(f"  Faces detected  : {report.get('faces', {}).get('faces_found', 0)}")
    print(f"  People clusters : {report.get('faces', {}).get('clusters', 0)}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
