"""
File organization engine.

Organizes recovered files into the structured layout:

    <output_dir>/
        Recovered_Files/
            images/
            videos/
            documents/
            audio/
            archives/
            other/
        Repaired_Files/
            images/
            videos/
            documents/
        Organized_Files/
            Person_1/
            Person_2/
            Unknown/

Maintains SQLite metadata for every file.
"""

import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional

from ..recovery.engine import RecoveredFile
from .metadata import MetadataDB

logger = logging.getLogger(__name__)

# Mapping from extension to category
EXTENSION_CATEGORIES: Dict[str, str] = {
    # Images
    "jpg": "images", "jpeg": "images", "png": "images",
    "gif": "images", "bmp": "images", "tiff": "images", "tif": "images",
    "webp": "images", "heic": "images", "raw": "images",
    # Videos
    "mp4": "videos", "avi": "videos", "mov": "videos", "mkv": "videos",
    "wmv": "videos", "flv": "videos", "webm": "videos", "m4v": "videos",
    # Documents
    "pdf": "documents", "docx": "documents", "doc": "documents",
    "xlsx": "documents", "xls": "documents", "pptx": "documents",
    "ppt": "documents", "txt": "documents", "rtf": "documents",
    "odt": "documents", "ods": "documents",
    # Audio
    "mp3": "audio", "wav": "audio", "flac": "audio", "aac": "audio",
    "ogg": "audio", "wma": "audio", "m4a": "audio",
    # Archives
    "zip": "archives", "rar": "archives", "7z": "archives",
    "tar": "archives", "gz": "archives", "bz2": "archives",
}


class FileOrganizer:
    """
    Organizes recovered files into the standard directory layout
    and records metadata in SQLite.
    """

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.recovered_dir = self.output_dir / "Recovered_Files"
        self.repaired_dir = self.output_dir / "Repaired_Files"
        self.organized_dir = self.output_dir / "Organized_Files"
        self._ensure_dirs()

        db_path = self.output_dir / "recovery_metadata.db"
        self.db = MetadataDB(str(db_path))

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _ensure_dirs(self) -> None:
        """Create all required output directories."""
        for category in set(EXTENSION_CATEGORIES.values()) | {"other"}:
            (self.recovered_dir / category).mkdir(parents=True, exist_ok=True)
            (self.repaired_dir / category).mkdir(parents=True, exist_ok=True)
        (self.organized_dir / "Unknown").mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_recovered_file(self, rf: RecoveredFile) -> Path:
        """
        Save a RecoveredFile to the appropriate Recovered_Files sub-directory.

        Args:
            rf: RecoveredFile object from the recovery engine.

        Returns:
            Path where the file was saved.
        """
        category = EXTENSION_CATEGORIES.get(rf.extension.lower(), "other")
        dest_dir = self.recovered_dir / category
        saved_path = rf.save(dest_dir)

        self.db.insert_file(
            str(saved_path),
            original_path=rf.original_name,
            file_size=rf.size,
            sha256=rf.sha256,
            extension=rf.extension,
            recovery_method=rf.method,
        )
        logger.debug("Saved %s -> %s", rf.original_name or "unknown", saved_path)
        return saved_path

    def save_all_recovered(self, recovered_files: List[RecoveredFile]) -> List[Path]:
        """
        Save all recovered files and return their paths.

        Args:
            recovered_files: List from RecoveryEngine.run().

        Returns:
            List of saved file paths.
        """
        saved = []
        for rf in recovered_files:
            try:
                path = self.save_recovered_file(rf)
                saved.append(path)
            except Exception as exc:
                logger.error("Failed to save recovered file: %s", exc)
        logger.info("Saved %d/%d recovered files", len(saved), len(recovered_files))
        return saved

    def organize_by_person(
        self,
        image_paths: List[Path],
        person_labels: List[str],
        confidence_scores: Optional[List[float]] = None,
    ) -> Dict[str, List[Path]]:
        """
        Organize images into per-person directories under Organized_Files/.

        Args:
            image_paths: List of image file paths.
            person_labels: Parallel list of person labels (e.g. "Person_1").
            confidence_scores: Optional parallel list of confidence scores.

        Returns:
            Dict mapping person_label -> list of organized file paths.
        """
        if confidence_scores is None:
            confidence_scores = [1.0] * len(image_paths)

        organized: Dict[str, List[Path]] = {}

        for img_path, label, conf in zip(image_paths, person_labels, confidence_scores):
            person_dir = self.organized_dir / label
            person_dir.mkdir(exist_ok=True)

            dest = person_dir / img_path.name
            counter = 0
            while dest.exists():
                counter += 1
                dest = person_dir / f"{img_path.stem}_{counter}{img_path.suffix}"

            shutil.copy2(str(img_path), str(dest))
            organized.setdefault(label, []).append(dest)

            # Update metadata
            rows = self.db.list_all()
            matched = [r for r in rows if r["saved_path"] == str(img_path)]
            for row in matched:
                self.db.update_person_label(row["id"], label, conf)

        total = sum(len(v) for v in organized.values())
        logger.info("Organized %d images across %d people", total, len(organized))
        return organized

    def mark_repaired(self, original_path: Path, repaired_path: Path) -> None:
        """Update metadata to mark a file as repaired."""
        rows = self.db.list_all()
        matched = [r for r in rows if r["saved_path"] == str(original_path)]
        for row in matched:
            self.db.update_repaired(row["id"], str(repaired_path))

    def report(self) -> Dict:
        """Return a summary report from the metadata database."""
        return self.db.stats()

    def close(self) -> None:
        """Clean up resources."""
        self.db.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
