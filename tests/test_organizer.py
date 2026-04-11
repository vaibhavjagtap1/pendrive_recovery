"""Tests for the file organizer and metadata database."""

import io
import zipfile
from pathlib import Path

import pytest

from src.organizer.metadata import MetadataDB
from src.organizer.file_organizer import FileOrganizer, EXTENSION_CATEGORIES
from src.recovery.engine import RecoveredFile


# -----------------------------------------------------------------------
# MetadataDB
# -----------------------------------------------------------------------

class TestMetadataDB:
    def test_insert_and_retrieve(self, tmp_dir):
        db = MetadataDB(str(tmp_dir / "meta.db"))
        row_id = db.insert_file(
            "/output/file.jpg",
            original_path="/dev/sdb@1234",
            file_size=5000,
            sha256="abc123",
            extension="jpg",
            recovery_method="file_carving",
        )
        assert row_id == 1

        row = db.get_by_id(1)
        assert row is not None
        assert row["saved_path"] == "/output/file.jpg"
        assert row["sha256"] == "abc123"
        assert row["extension"] == "jpg"
        db.close()

    def test_update_person_label(self, tmp_dir):
        db = MetadataDB(str(tmp_dir / "meta.db"))
        row_id = db.insert_file("/out/img.jpg", extension="jpg")
        db.update_person_label(row_id, "Person_1", 0.95)
        row = db.get_by_id(row_id)
        assert row["person_label"] == "Person_1"
        assert abs(row["confidence"] - 0.95) < 1e-6
        db.close()

    def test_update_repaired(self, tmp_dir):
        db = MetadataDB(str(tmp_dir / "meta.db"))
        row_id = db.insert_file("/out/img.jpg")
        db.update_repaired(row_id, "/out/repaired/img_repaired.jpg")
        row = db.get_by_id(row_id)
        assert row["repaired"] == 1
        assert "repaired" in row["repaired_path"]
        db.close()

    def test_get_by_sha256(self, tmp_dir):
        db = MetadataDB(str(tmp_dir / "meta.db"))
        db.insert_file("/out/a.jpg", sha256="deadbeef")
        db.insert_file("/out/b.jpg", sha256="deadbeef")
        db.insert_file("/out/c.jpg", sha256="cafebabe")
        rows = db.get_by_sha256("deadbeef")
        assert len(rows) == 2
        db.close()

    def test_list_by_person(self, tmp_dir):
        db = MetadataDB(str(tmp_dir / "meta.db"))
        for i in range(3):
            rid = db.insert_file(f"/out/p1_{i}.jpg", extension="jpg")
            db.update_person_label(rid, "Person_1", 0.9)
        rid = db.insert_file("/out/p2.jpg", extension="jpg")
        db.update_person_label(rid, "Person_2", 0.8)

        p1_rows = db.list_by_person("Person_1")
        assert len(p1_rows) == 3
        p2_rows = db.list_by_person("Person_2")
        assert len(p2_rows) == 1
        db.close()

    def test_stats(self, tmp_dir):
        db = MetadataDB(str(tmp_dir / "meta.db"))
        for i in range(5):
            rid = db.insert_file(f"/out/{i}.jpg", extension="jpg", sha256=str(i))
        db.update_repaired(1, "/out/repaired/0_repaired.jpg")
        db.update_person_label(2, "Person_1", 0.9)

        stats = db.stats()
        assert stats["total_files"] == 5
        assert stats["repaired_files"] == 1
        assert stats["files_with_faces"] == 1
        db.close()

    def test_context_manager(self, tmp_dir):
        with MetadataDB(str(tmp_dir / "meta.db")) as db:
            rid = db.insert_file("/out/x.jpg")
            assert rid == 1


# -----------------------------------------------------------------------
# FileOrganizer
# -----------------------------------------------------------------------

class TestFileOrganizer:
    def test_dirs_created(self, tmp_dir):
        organizer = FileOrganizer(str(tmp_dir))
        assert (tmp_dir / "Recovered_Files").is_dir()
        assert (tmp_dir / "Repaired_Files").is_dir()
        assert (tmp_dir / "Organized_Files" / "Unknown").is_dir()
        organizer.close()

    def test_save_recovered_file(self, tmp_dir):
        organizer = FileOrganizer(str(tmp_dir))
        data = b"\xFF\xD8\xFF" + b"\x00" * 50 + b"\xFF\xD9"
        rf = RecoveredFile(data=data, extension="jpg", offset=0, method="test")
        saved = organizer.save_recovered_file(rf)
        assert saved.exists()
        assert saved.read_bytes() == data
        organizer.close()

    def test_save_recovered_categorizes_correctly(self, tmp_dir):
        organizer = FileOrganizer(str(tmp_dir))
        # JPG should go to images/
        jpg_rf = RecoveredFile(data=b"\xFF\xD8\xFF\xFF\xD9", extension="jpg", offset=0, method="test")
        jpg_path = organizer.save_recovered_file(jpg_rf)
        assert "images" in str(jpg_path)

        # PDF should go to documents/
        pdf_rf = RecoveredFile(data=b"%PDF-1.4\n%%EOF\n", extension="pdf", offset=100, method="test")
        pdf_path = organizer.save_recovered_file(pdf_rf)
        assert "documents" in str(pdf_path)
        organizer.close()

    def test_save_all_recovered(self, tmp_dir):
        organizer = FileOrganizer(str(tmp_dir))
        files = [
            RecoveredFile(data=b"\xFF\xD8\xFF\xFF\xD9", extension="jpg", offset=i * 100, method="test")
            for i in range(5)
        ]
        saved = organizer.save_all_recovered(files)
        assert len(saved) == 5
        organizer.close()

    def test_organize_by_person(self, tmp_dir):
        organizer = FileOrganizer(str(tmp_dir))
        # Create some image files in Recovered_Files
        imgs_dir = tmp_dir / "Recovered_Files" / "images"
        imgs_dir.mkdir(parents=True, exist_ok=True)
        img_paths = []
        for i in range(4):
            p = imgs_dir / f"img_{i}.jpg"
            p.write_bytes(b"\xFF\xD8\xFF\xFF\xD9")
            img_paths.append(p)

        labels = ["Person_1", "Person_1", "Person_2", "Unknown"]
        result = organizer.organize_by_person(img_paths, labels)

        assert "Person_1" in result
        assert len(result["Person_1"]) == 2
        assert (tmp_dir / "Organized_Files" / "Person_1").is_dir()
        assert (tmp_dir / "Organized_Files" / "Person_2").is_dir()
        organizer.close()

    def test_mark_repaired(self, tmp_dir):
        organizer = FileOrganizer(str(tmp_dir))
        data = b"\xFF\xD8\xFF\xFF\xD9"
        rf = RecoveredFile(data=data, extension="jpg", offset=0, method="test")
        saved = organizer.save_recovered_file(rf)

        repaired_path = tmp_dir / "repaired.jpg"
        repaired_path.write_bytes(data)
        organizer.mark_repaired(saved, repaired_path)

        rows = organizer.db.list_all()
        assert any(r["repaired"] == 1 for r in rows)
        organizer.close()

    def test_report(self, tmp_dir):
        organizer = FileOrganizer(str(tmp_dir))
        rf = RecoveredFile(data=b"\xFF\xD8\xFF\xFF\xD9", extension="jpg", offset=0, method="test")
        organizer.save_recovered_file(rf)
        report = organizer.report()
        assert "total_files" in report
        assert report["total_files"] >= 1
        organizer.close()

    def test_extension_categories_complete(self):
        """Ensure common extensions are covered."""
        for ext in ("jpg", "jpeg", "png", "mp4", "avi", "pdf", "docx"):
            assert ext in EXTENSION_CATEGORIES, f"{ext} not in EXTENSION_CATEGORIES"
