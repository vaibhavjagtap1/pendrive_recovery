"""
Face detector module – fully offline face detection.

Primary backend: OpenCV's DNN module with a bundled Caffe/TF model.
Fallback backend: OpenCV Haar cascade (always available with opencv-python).

When advanced models (RetinaFace / InsightFace) are installed they are
preferred automatically.
"""

import logging
from pathlib import Path
from typing import List, NamedTuple, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# OpenCV DNN SSD model for face detection
# These model files ship with opencv-python-contrib or can be placed in models/
_MODEL_DIR = Path(__file__).parent.parent.parent / "models"
_PROTO_PATH = _MODEL_DIR / "deploy.prototxt"
_MODEL_PATH = _MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

# Confidence threshold for DNN detection
DNN_CONFIDENCE_THRESHOLD = 0.5


class FaceBox(NamedTuple):
    """A detected face bounding box."""
    x: int
    y: int
    w: int
    h: int
    confidence: float


class FaceDetector:
    """
    Detects faces in images using the best available offline backend.

    Priority:
    1. RetinaFace (if insightface package installed)
    2. OpenCV DNN SSD (if model files present in models/)
    3. OpenCV Haar cascade (always available as fallback)
    """

    def __init__(self, model_dir: Optional[str] = None, confidence: float = DNN_CONFIDENCE_THRESHOLD):
        self.confidence_threshold = confidence
        if model_dir:
            global _MODEL_DIR, _PROTO_PATH, _MODEL_PATH
            _MODEL_DIR = Path(model_dir)
            _PROTO_PATH = _MODEL_DIR / "deploy.prototxt"
            _MODEL_PATH = _MODEL_DIR / "res10_300x300_ssd_iter_140000.caffemodel"

        self._backend = self._init_backend()
        logger.info("FaceDetector initialized with backend: %s", self._backend)

    # ------------------------------------------------------------------
    # Backend initialization
    # ------------------------------------------------------------------

    def _init_backend(self) -> str:
        # Try RetinaFace first
        try:
            from retinaface import RetinaFace  # type: ignore
            self._retinaface = RetinaFace
            return "retinaface"
        except ImportError:
            pass

        # Try OpenCV DNN SSD
        if _PROTO_PATH.exists() and _MODEL_PATH.exists():
            try:
                self._dnn_net = cv2.dnn.readNetFromCaffe(
                    str(_PROTO_PATH), str(_MODEL_PATH)
                )
                return "opencv_dnn"
            except cv2.error as exc:
                logger.warning("OpenCV DNN init failed: %s", exc)

        # Fallback: Haar cascade
        haar_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._haar = cv2.CascadeClassifier(haar_path)
        if self._haar.empty():
            logger.error("Haar cascade failed to load – face detection unavailable")
            return "none"
        return "haar"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, image_path: Path) -> List[FaceBox]:
        """
        Detect all faces in an image.

        Args:
            image_path: Path to a JPEG or PNG image.

        Returns:
            List of FaceBox objects (may be empty).
        """
        img = cv2.imread(str(image_path))
        if img is None:
            logger.debug("Could not load image: %s", image_path)
            return []
        return self.detect_from_array(img)

    def detect_from_array(self, img: np.ndarray) -> List[FaceBox]:
        """
        Detect faces from a NumPy image array (BGR, HWC).

        Args:
            img: Image as a NumPy array.

        Returns:
            List of FaceBox objects.
        """
        if self._backend == "retinaface":
            return self._detect_retinaface(img)
        elif self._backend == "opencv_dnn":
            return self._detect_dnn(img)
        elif self._backend == "haar":
            return self._detect_haar(img)
        return []

    # ------------------------------------------------------------------
    # Backend implementations
    # ------------------------------------------------------------------

    def _detect_retinaface(self, img: np.ndarray) -> List[FaceBox]:
        """Use RetinaFace (insightface) for detection."""
        try:
            faces = self._retinaface.detect_faces(img)
            results = []
            for key, face in faces.items():
                box = face["facial_area"]  # [x1, y1, x2, y2]
                score = face.get("score", 1.0)
                if score >= self.confidence_threshold:
                    x1, y1, x2, y2 = box
                    results.append(FaceBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1, confidence=score))
            return results
        except Exception as exc:
            logger.error("RetinaFace detection error: %s", exc)
            return []

    def _detect_dnn(self, img: np.ndarray) -> List[FaceBox]:
        """Use OpenCV DNN SSD model."""
        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(img, (300, 300)), 1.0, (300, 300),
            (104.0, 177.0, 123.0), swapRB=False,
        )
        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()

        results = []
        for i in range(detections.shape[2]):
            conf = float(detections[0, 0, i, 2])
            if conf >= self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                results.append(
                    FaceBox(x=x1, y=y1, w=x2 - x1, h=y2 - y1, confidence=conf)
                )
        return results

    def _detect_haar(self, img: np.ndarray) -> List[FaceBox]:
        """Use Haar cascade as a fallback."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self._haar.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        results = []
        if len(faces) > 0:
            for x, y, w, h in faces:
                results.append(FaceBox(x=int(x), y=int(y), w=int(w), h=int(h), confidence=1.0))
        return results
