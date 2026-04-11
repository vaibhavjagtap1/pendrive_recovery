"""
File signatures (magic bytes) for signature-based file carving.

Each entry maps a file extension to its:
- header: bytes that identify the start of the file
- footer: bytes that identify the end of the file (None if unknown)
- max_size: maximum expected file size in bytes
"""

FILE_SIGNATURES = {
    # Images
    "jpg": {
        "header": b"\xFF\xD8\xFF",
        "footer": b"\xFF\xD9",
        "max_size": 30 * 1024 * 1024,  # 30 MB
        "mime": "image/jpeg",
    },
    "png": {
        "header": b"\x89\x50\x4E\x47\x0D\x0A\x1A\x0A",
        "footer": b"\x49\x45\x4E\x44\xAE\x42\x60\x82",
        "max_size": 30 * 1024 * 1024,
        "mime": "image/png",
    },
    "gif": {
        "header": b"\x47\x49\x46\x38",
        "footer": b"\x00\x3B",
        "max_size": 10 * 1024 * 1024,
        "mime": "image/gif",
    },
    "bmp": {
        "header": b"\x42\x4D",
        "footer": None,
        "max_size": 30 * 1024 * 1024,
        "mime": "image/bmp",
    },
    "tiff": {
        "header": b"\x49\x49\x2A\x00",
        "footer": None,
        "max_size": 100 * 1024 * 1024,
        "mime": "image/tiff",
    },
    # Videos
    "mp4": {
        "header": b"\x00\x00\x00\x18\x66\x74\x79\x70",
        "footer": None,
        "max_size": 4 * 1024 * 1024 * 1024,  # 4 GB
        "mime": "video/mp4",
    },
    "avi": {
        "header": b"\x52\x49\x46\x46",
        "footer": None,
        "max_size": 4 * 1024 * 1024 * 1024,
        "mime": "video/avi",
    },
    "mov": {
        "header": b"\x00\x00\x00\x14\x66\x74\x79\x70\x71\x74",
        "footer": None,
        "max_size": 4 * 1024 * 1024 * 1024,
        "mime": "video/quicktime",
    },
    "mkv": {
        "header": b"\x1A\x45\xDF\xA3",
        "footer": None,
        "max_size": 4 * 1024 * 1024 * 1024,
        "mime": "video/x-matroska",
    },
    # Documents
    "pdf": {
        "header": b"\x25\x50\x44\x46",
        "footer": b"\x25\x25\x45\x4F\x46",
        "max_size": 500 * 1024 * 1024,  # 500 MB
        "mime": "application/pdf",
    },
    "docx": {
        "header": b"\x50\x4B\x03\x04",
        "footer": None,
        "max_size": 100 * 1024 * 1024,
        "mime": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    },
    "xlsx": {
        "header": b"\x50\x4B\x03\x04",
        "footer": None,
        "max_size": 100 * 1024 * 1024,
        "mime": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    },
    "doc": {
        "header": b"\xD0\xCF\x11\xE0\xA1\xB1\x1A\xE1",
        "footer": None,
        "max_size": 100 * 1024 * 1024,
        "mime": "application/msword",
    },
    # Audio
    "mp3": {
        "header": b"\xFF\xFB",
        "footer": None,
        "max_size": 100 * 1024 * 1024,
        "mime": "audio/mpeg",
    },
    "wav": {
        "header": b"\x52\x49\x46\x46",
        "footer": None,
        "max_size": 500 * 1024 * 1024,
        "mime": "audio/wav",
    },
    # Archives
    "zip": {
        "header": b"\x50\x4B\x03\x04",
        "footer": b"\x50\x4B\x05\x06",
        "max_size": 4 * 1024 * 1024 * 1024,
        "mime": "application/zip",
    },
    "7z": {
        "header": b"\x37\x7A\xBC\xAF\x27\x1C",
        "footer": None,
        "max_size": 4 * 1024 * 1024 * 1024,
        "mime": "application/x-7z-compressed",
    },
    "rar": {
        "header": b"\x52\x61\x72\x21\x1A\x07",
        "footer": None,
        "max_size": 4 * 1024 * 1024 * 1024,
        "mime": "application/x-rar-compressed",
    },
    # Executables / other
    "exe": {
        "header": b"\x4D\x5A",
        "footer": None,
        "max_size": 500 * 1024 * 1024,
        "mime": "application/x-msdownload",
    },
    "sqlite": {
        "header": b"\x53\x51\x4C\x69\x74\x65\x20\x66\x6F\x72\x6D\x61\x74\x20\x33\x00",
        "footer": None,
        "max_size": 4 * 1024 * 1024 * 1024,
        "mime": "application/x-sqlite3",
    },
}

# Build a reverse lookup: header bytes -> (extension, signature)
HEADER_LOOKUP = {}
for ext, sig in FILE_SIGNATURES.items():
    hdr = sig["header"]
    if hdr not in HEADER_LOOKUP:
        HEADER_LOOKUP[hdr] = []
    HEADER_LOOKUP[hdr].append((ext, sig))

# Maximum header length across all signatures
MAX_HEADER_LEN = max(len(sig["header"]) for sig in FILE_SIGNATURES.values())
