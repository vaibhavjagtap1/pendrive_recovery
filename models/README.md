# AI Model Files

This directory is for locally-bundled AI model files used by the face
detection and embedding modules. All models run **fully offline**.

## Supported Models

### Face Detection (optional – falls back to OpenCV Haar cascade)

Place the following files here for OpenCV DNN-based detection:
- `deploy.prototxt` – network architecture
- `res10_300x300_ssd_iter_140000.caffemodel` – pre-trained weights

These files are available from the OpenCV GitHub repository (
`opencv/samples/dnn/face_detector/`).

### Face Recognition / Embeddings (optional – falls back to HOG)

Place the following file here for ArcFace-quality embeddings:
- `arcface_r100.onnx` – ArcFace ResNet-100 ONNX model

This model can be obtained from the InsightFace model zoo.

## Recommended: insightface package

For production-quality face recognition, install the `insightface`
Python package which automatically manages model downloads:

```bash
pip install insightface onnxruntime
```

When insightface is installed, it is used automatically instead of
the fallback backends.
