# Pendrive Recovery

**Offline AI-Based Pendrive Recovery & Face-Based File Organization System**

---

## 🎯 Overview

A fully offline, enterprise-grade Python system that:

1. Recovers files from corrupted or unstable USB storage devices
2. Repairs partially corrupted files (JPEG, PNG, MP4, AVI, PDF, DOCX)
3. Performs AI-based face detection and clustering on recovered images
4. Organizes files automatically by detected person
5. Works **100% offline** – no internet or cloud dependency

---

## 📁 Project Structure

```
pendrive_recovery/
├── src/
│   ├── recovery/        # File carving engine + FAT32/NTFS support
│   ├── repair/          # Image, video, and document repair
│   ├── ai/              # Face detection, embedding, clustering
│   ├── organizer/       # File organization + SQLite metadata
│   ├── utils/           # Logging + USB device handler
│   └── pipeline.py      # End-to-end orchestration pipeline
├── tests/               # pytest test suite (56 tests)
├── models/              # Local AI model files (see models/README.md)
├── main.py              # CLI entry point
├── requirements.txt
└── setup.py
```

---

## ⚡ Quick Start

### Install dependencies

```bash
pip install -r requirements.txt
```

### List storage devices

```bash
python main.py --list-devices
```

### Run recovery

```bash
# Recover from a device (requires root/admin on Linux/Windows)
python main.py --device /dev/sdb --output ./recovered_output

# Recover from a disk image file
python main.py --device disk.img --output ./recovered_output

# Skip repair and AI analysis
python main.py --device disk.img --output ./out --no-repair --no-ai
```

---

## 📂 Output Layout

```
recovered_output/
├── Recovered_Files/
│   ├── images/        # Recovered JPEGs, PNGs, ...
│   ├── videos/        # Recovered MP4, AVI, ...
│   ├── documents/     # Recovered PDFs, DOCX, ...
│   ├── audio/
│   ├── archives/
│   └── other/
├── Repaired_Files/
│   ├── images/
│   ├── videos/
│   └── documents/
├── Organized_Files/
│   ├── Person_1/      # Images grouped by face identity
│   ├── Person_2/
│   └── Unknown/
├── recovery_metadata.db   # SQLite database with full metadata
├── recovery_report.txt    # Human-readable summary
└── recovery.log           # Detailed log file
```

---

## 🧠 AI Face Recognition

Three backends in priority order (all fully offline):

1. **InsightFace** (ArcFace + RetinaFace) – install with `pip install insightface onnxruntime`
2. **OpenCV DNN SSD** – place model files in `models/` directory
3. **OpenCV Haar cascade** – always available as fallback (bundled with opencv-python)

Clustering uses **DBSCAN** which automatically determines the number of people.

---

## 🔧 Tech Stack

- Python 3.9+
- OpenCV (`opencv-python-headless`)
- scikit-learn (DBSCAN clustering)
- Pillow (image validation/re-encoding)
- SQLite (metadata)
- pytest (testing)

---

## 🧪 Running Tests

```bash
pytest tests/ -v
```

All 56 tests pass. Coverage includes recovery engine, file repair, AI clustering,
file organizer, and metadata database.

---

## 🔐 Data Safety

- Device opened in **read-only** mode – no data is ever written to the source device
- All recovered files are written to the `--output` directory only
- SHA-256 deduplication prevents duplicate saves

---

## 📄 License

MIT
