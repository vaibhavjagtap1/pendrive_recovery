"""
Main pipeline: orchestrates recovery, repair, AI analysis and organization.
"""

import logging
from pathlib import Path
from typing import Callable, List, Optional

from .recovery.engine import RecoveryEngine, RecoveredFile
from .repair.image_repair import ImageRepairer
from .repair.video_repair import VideoRepairer
from .repair.document_repair import DocumentRepairer
from .ai.face_detector import FaceDetector
from .ai.face_embedder import FaceEmbedder
from .ai.face_clusterer import FaceClusterer
from .organizer.file_organizer import FileOrganizer
from .utils.logger import setup_logger

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {"jpg", "jpeg", "png", "bmp", "tiff", "tif", "gif"}
REPAIR_IMAGE_EXTS = {"jpg", "jpeg", "png"}
REPAIR_VIDEO_EXTS = {"mp4", "avi"}
REPAIR_DOC_EXTS = {"pdf", "docx", "xlsx", "pptx"}


class RecoveryPipeline:
    """
    Full end-to-end pipeline:
    1. Recover files from device
    2. Save to Recovered_Files/
    3. Repair corrupted files
    4. Detect & embed faces in recovered images
    5. Cluster faces and organize into Organized_Files/
    6. Write recovery report
    """

    def __init__(
        self,
        device_path: str,
        output_dir: str,
        log_file: Optional[str] = None,
        enable_repair: bool = True,
        enable_ai: bool = True,
        dbscan_eps: float = 0.5,
        progress_callback: Optional[Callable[[float, str], None]] = None,
    ):
        self.device_path = device_path
        self.output_dir = Path(output_dir)
        self.enable_repair = enable_repair
        self.enable_ai = enable_ai
        self.dbscan_eps = dbscan_eps
        self.progress_callback = progress_callback

        setup_logger(
            level=logging.INFO,
            log_file=log_file or str(self.output_dir / "recovery.log"),
        )

    # ------------------------------------------------------------------

    def run(self) -> dict:
        """
        Execute the full pipeline.

        Returns:
            Dict with summary statistics.
        """
        self._progress(0.0, "Starting pipeline")

        organizer = FileOrganizer(str(self.output_dir))

        # Step 1: Recovery
        self._progress(5.0, "Recovering files from device…")
        engine = RecoveryEngine(
            self.device_path,
            str(self.output_dir),
            progress_callback=lambda p, m: self._progress(5.0 + p * 0.5, m),
        )
        recovered_files: List[RecoveredFile] = engine.run()

        # Step 2: Save recovered files
        self._progress(55.0, f"Saving {len(recovered_files)} recovered files…")
        saved_paths = organizer.save_all_recovered(recovered_files)

        # Step 3: Repair
        repair_stats = {"attempted": 0, "succeeded": 0}
        if self.enable_repair:
            self._progress(60.0, "Repairing corrupted files…")
            repair_stats = self._repair_files(saved_paths, organizer)

        # Step 4 & 5: AI face analysis
        face_stats = {"images_analysed": 0, "faces_found": 0, "clusters": 0}
        if self.enable_ai:
            self._progress(75.0, "Running AI face analysis…")
            face_stats = self._run_face_pipeline(saved_paths, organizer)

        # Step 6: Report
        self._progress(95.0, "Generating report…")
        report = organizer.report()
        report.update(
            {
                "recovery": {"files_recovered": len(recovered_files)},
                "repair": repair_stats,
                "faces": face_stats,
            }
        )
        self._write_report(report)
        organizer.close()

        self._progress(100.0, "Pipeline complete")
        return report

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _repair_files(self, paths: List[Path], organizer: FileOrganizer) -> dict:
        img_repairer = ImageRepairer(str(self.output_dir))
        vid_repairer = VideoRepairer(str(self.output_dir))
        doc_repairer = DocumentRepairer(str(self.output_dir))

        attempted = 0
        succeeded = 0
        for path in paths:
            ext = path.suffix.lower().lstrip(".")
            success, repaired_path = False, None

            if ext in REPAIR_IMAGE_EXTS:
                success, repaired_path = img_repairer.repair(path)
            elif ext in REPAIR_VIDEO_EXTS:
                success, repaired_path = vid_repairer.repair(path)
            elif ext in REPAIR_DOC_EXTS:
                success, repaired_path = doc_repairer.repair(path)
            else:
                continue

            attempted += 1
            if success and repaired_path:
                succeeded += 1
                organizer.mark_repaired(path, repaired_path)

        logger.info("Repair: %d attempted, %d succeeded", attempted, succeeded)
        return {"attempted": attempted, "succeeded": succeeded}

    def _run_face_pipeline(self, paths: List[Path], organizer: FileOrganizer) -> dict:
        image_paths = [p for p in paths if p.suffix.lower().lstrip(".") in IMAGE_EXTENSIONS]
        if not image_paths:
            return {"images_analysed": 0, "faces_found": 0, "clusters": 0}

        try:
            import cv2
        except ImportError:
            logger.warning("opencv-python not installed; skipping AI analysis")
            return {"images_analysed": 0, "faces_found": 0, "clusters": 0}

        detector = FaceDetector()
        embedder = FaceEmbedder()
        clusterer = FaceClusterer(eps=self.dbscan_eps)

        all_embeddings = []
        face_image_map = []  # (image_path, face_index_in_all_embeddings)

        for img_path in image_paths:
            import cv2 as _cv2
            img = _cv2.imread(str(img_path))
            if img is None:
                continue
            faces = detector.detect_from_array(img)
            for fb in faces:
                emb = embedder.embed(img, fb)
                if emb is not None:
                    face_image_map.append(img_path)
                    all_embeddings.append(emb)

        faces_found = len(all_embeddings)
        if faces_found == 0:
            return {
                "images_analysed": len(image_paths),
                "faces_found": 0,
                "clusters": 0,
            }

        labels = clusterer.assign_labels(all_embeddings)
        # Deduplicate: pick best label per image (most common label wins)
        img_label_map: dict = {}
        for img_path, label in zip(face_image_map, labels):
            img_label_map.setdefault(str(img_path), []).append(label)

        final_img_paths = []
        final_labels = []
        for img_path_str, img_labels in img_label_map.items():
            # Pick the most common label for this image
            best = max(set(img_labels), key=img_labels.count)
            final_img_paths.append(Path(img_path_str))
            final_labels.append(best)

        organizer.organize_by_person(final_img_paths, final_labels)
        n_clusters = len({l for l in labels if l != "Unknown"})

        return {
            "images_analysed": len(image_paths),
            "faces_found": faces_found,
            "clusters": n_clusters,
        }

    def _write_report(self, report: dict) -> None:
        report_path = self.output_dir / "recovery_report.txt"
        lines = ["=" * 60, "  PENDRIVE RECOVERY REPORT", "=" * 60, ""]
        lines.append(f"Files recovered  : {report.get('recovery', {}).get('files_recovered', 0)}")
        lines.append(f"Files repaired   : {report.get('repair', {}).get('succeeded', 0)}")
        lines.append(f"Faces detected   : {report.get('faces', {}).get('faces_found', 0)}")
        lines.append(f"People clusters  : {report.get('faces', {}).get('clusters', 0)}")
        lines.append("")
        lines.append("--- By Extension ---")
        for item in report.get("by_extension", []):
            lines.append(f"  .{item['extension']:10s} {item['cnt']:>6d}")
        lines.append("")
        lines.append("--- By Person ---")
        for item in report.get("by_person", []):
            lines.append(f"  {item['person_label']:15s} {item['cnt']:>6d}")
        lines.append("")
        report_path.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Report written to %s", report_path)

    def _progress(self, pct: float, msg: str) -> None:
        logger.info("[%.0f%%] %s", pct, msg)
        if self.progress_callback:
            self.progress_callback(pct, msg)
