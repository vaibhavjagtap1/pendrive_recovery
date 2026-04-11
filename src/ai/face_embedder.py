"""
Face embedder module – generate 512-d face embeddings offline.

Primary backend: ArcFace via insightface package.
Fallback backend: OpenCV DNN with a local ONNX model (if present).
Last-resort: simple HOG-based descriptor for testing/demo purposes.
"""

import logging
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np

from .face_detector import FaceBox

logger = logging.getLogger(__name__)

_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
_ARCFACE_ONNX = _MODEL_DIR / "arcface_r100.onnx"

EMBEDDING_DIM = 512


class FaceEmbedder:
    """
    Generates normalised 512-dimensional face embeddings.

    Priority:
    1. ArcFace via insightface (most accurate)
    2. OpenCV DNN with local ONNX model
    3. HOG-based fallback (for development/testing)
    """

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir:
            global _MODEL_DIR, _ARCFACE_ONNX
            _MODEL_DIR = Path(model_dir)
            _ARCFACE_ONNX = _MODEL_DIR / "arcface_r100.onnx"

        self._backend = self._init_backend()
        logger.info("FaceEmbedder initialized with backend: %s", self._backend)

    # ------------------------------------------------------------------
    # Backend initialization
    # ------------------------------------------------------------------

    def _init_backend(self) -> str:
        # 1. InsightFace ArcFace
        try:
            import insightface  # type: ignore

            self._app = insightface.app.FaceAnalysis(
                allowed_modules=["recognition"],
                providers=["CPUExecutionProvider"],
            )
            self._app.prepare(ctx_id=-1, det_size=(640, 640))
            return "insightface"
        except ImportError:
            pass

        # 2. OpenCV DNN + ONNX
        if _ARCFACE_ONNX.exists():
            try:
                self._net = cv2.dnn.readNetFromONNX(str(_ARCFACE_ONNX))
                return "opencv_onnx"
            except cv2.error as exc:
                logger.warning("OpenCV ONNX load failed: %s", exc)

        # 3. HOG fallback
        self._hog = cv2.HOGDescriptor()
        logger.warning(
            "Using HOG fallback for face embeddings. "
            "Install insightface for production-quality results."
        )
        return "hog"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed(self, img: np.ndarray, face_box: FaceBox) -> Optional[np.ndarray]:
        """
        Generate a normalised embedding for a single face crop.

        Args:
            img: Full image (BGR NumPy array).
            face_box: Bounding box of the face to embed.

        Returns:
            1-D NumPy array of length EMBEDDING_DIM, or None on failure.
        """
        crop = self._crop_face(img, face_box)
        if crop is None or crop.size == 0:
            return None

        if self._backend == "insightface":
            return self._embed_insightface(img, face_box)
        elif self._backend == "opencv_onnx":
            return self._embed_onnx(crop)
        else:
            return self._embed_hog(crop)

    def embed_all(self, img: np.ndarray, faces: List[FaceBox]) -> List[Optional[np.ndarray]]:
        """Generate embeddings for all detected face boxes."""
        return [self.embed(img, fb) for fb in faces]

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _embed_insightface(self, img: np.ndarray, face_box: FaceBox) -> Optional[np.ndarray]:
        """Use insightface to extract embedding."""
        try:
            import insightface  # type: ignore

            faces = self._app.get(img)
            if not faces:
                return None
            # Match the face closest to our detected box
            best = min(
                faces,
                key=lambda f: abs(f.bbox[0] - face_box.x) + abs(f.bbox[1] - face_box.y),
            )
            emb = best.embedding
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception as exc:
            logger.error("InsightFace embedding error: %s", exc)
            return None

    def _embed_onnx(self, crop: np.ndarray) -> Optional[np.ndarray]:
        """Use OpenCV DNN to run an ONNX ArcFace model."""
        try:
            resized = cv2.resize(crop, (112, 112))
            blob = cv2.dnn.blobFromImage(
                resized, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True,
            )
            self._net.setInput(blob)
            out = self._net.forward()
            emb = out.flatten()
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception as exc:
            logger.error("ONNX embedding error: %s", exc)
            return None

    def _embed_hog(self, crop: np.ndarray) -> np.ndarray:
        """
        HOG-based fallback embedding.

        Not identity-accurate; only useful for structural testing.
        """
        try:
            resized = cv2.resize(crop, (64, 128))
            desc = self._hog.compute(resized)
            emb = desc.flatten()[:EMBEDDING_DIM]
            # Pad or truncate to EMBEDDING_DIM
            if len(emb) < EMBEDDING_DIM:
                emb = np.pad(emb, (0, EMBEDDING_DIM - len(emb)))
            else:
                emb = emb[:EMBEDDING_DIM]
            norm = np.linalg.norm(emb)
            return emb / norm if norm > 0 else emb
        except Exception as exc:
            logger.error("HOG embedding error: %s", exc)
            return np.zeros(EMBEDDING_DIM, dtype=np.float32)

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _crop_face(img: np.ndarray, fb: FaceBox) -> Optional[np.ndarray]:
        """Crop and return the face region from the image."""
        h, w = img.shape[:2]
        x1 = max(0, fb.x)
        y1 = max(0, fb.y)
        x2 = min(w, fb.x + fb.w)
        y2 = min(h, fb.y + fb.h)
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2]
