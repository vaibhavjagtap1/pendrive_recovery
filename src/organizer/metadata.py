"""
Metadata database – SQLite-backed store for file recovery and organization records.
"""

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS recovered_files (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    original_path   TEXT,
    saved_path      TEXT NOT NULL,
    file_size       INTEGER,
    sha256          TEXT,
    extension       TEXT,
    recovery_method TEXT,
    recovered_at    TEXT NOT NULL,
    repaired        INTEGER DEFAULT 0,
    repaired_path   TEXT,
    person_label    TEXT,
    confidence      REAL
);

CREATE INDEX IF NOT EXISTS idx_sha256 ON recovered_files(sha256);
CREATE INDEX IF NOT EXISTS idx_extension ON recovered_files(extension);
CREATE INDEX IF NOT EXISTS idx_person ON recovered_files(person_label);
"""


class MetadataDB:
    """
    SQLite database for persisting recovery metadata.

    Each recovered file has:
    - original_path: where it was on the device (if known)
    - saved_path: where it was saved during this session
    - sha256: content hash for deduplication
    - recovery_method: 'file_carving' | 'fat32_directory' | 'ntfs_mft'
    - repaired: whether a repair was attempted
    - person_label: face-cluster assignment (if image with faces)
    - confidence: clustering confidence score
    """

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _init_schema(self) -> None:
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def insert_file(
        self,
        saved_path: str,
        *,
        original_path: Optional[str] = None,
        file_size: Optional[int] = None,
        sha256: Optional[str] = None,
        extension: Optional[str] = None,
        recovery_method: Optional[str] = None,
        repaired: bool = False,
        repaired_path: Optional[str] = None,
        person_label: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> int:
        """Insert a file record and return its row ID."""
        cur = self._conn.execute(
            """
            INSERT INTO recovered_files
                (original_path, saved_path, file_size, sha256, extension,
                 recovery_method, recovered_at, repaired, repaired_path,
                 person_label, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                original_path,
                saved_path,
                file_size,
                sha256,
                extension,
                recovery_method,
                datetime.now(timezone.utc).isoformat(),
                1 if repaired else 0,
                repaired_path,
                person_label,
                confidence,
            ),
        )
        self._conn.commit()
        return cur.lastrowid

    def update_person_label(self, row_id: int, person_label: str, confidence: float) -> None:
        """Update the person label and confidence score for a row."""
        self._conn.execute(
            "UPDATE recovered_files SET person_label = ?, confidence = ? WHERE id = ?",
            (person_label, confidence, row_id),
        )
        self._conn.commit()

    def update_repaired(self, row_id: int, repaired_path: str) -> None:
        """Mark a file as repaired."""
        self._conn.execute(
            "UPDATE recovered_files SET repaired = 1, repaired_path = ? WHERE id = ?",
            (repaired_path, row_id),
        )
        self._conn.commit()

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def get_by_id(self, row_id: int) -> Optional[Dict]:
        row = self._conn.execute(
            "SELECT * FROM recovered_files WHERE id = ?", (row_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_by_sha256(self, sha256: str) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM recovered_files WHERE sha256 = ?", (sha256,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_by_person(self, person_label: str) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM recovered_files WHERE person_label = ?", (person_label,)
        ).fetchall()
        return [dict(r) for r in rows]

    def list_all(self) -> List[Dict]:
        rows = self._conn.execute(
            "SELECT * FROM recovered_files ORDER BY id"
        ).fetchall()
        return [dict(r) for r in rows]

    def stats(self) -> Dict:
        """Return summary statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM recovered_files").fetchone()[0]
        repaired = self._conn.execute(
            "SELECT COUNT(*) FROM recovered_files WHERE repaired = 1"
        ).fetchone()[0]
        with_faces = self._conn.execute(
            "SELECT COUNT(*) FROM recovered_files WHERE person_label IS NOT NULL"
        ).fetchone()[0]
        by_ext = self._conn.execute(
            "SELECT extension, COUNT(*) as cnt FROM recovered_files GROUP BY extension ORDER BY cnt DESC"
        ).fetchall()
        by_person = self._conn.execute(
            "SELECT person_label, COUNT(*) as cnt FROM recovered_files "
            "WHERE person_label IS NOT NULL GROUP BY person_label ORDER BY cnt DESC"
        ).fetchall()
        return {
            "total_files": total,
            "repaired_files": repaired,
            "files_with_faces": with_faces,
            "by_extension": [dict(r) for r in by_ext],
            "by_person": [dict(r) for r in by_person],
        }

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
