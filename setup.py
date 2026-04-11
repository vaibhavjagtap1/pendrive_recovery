"""Setup configuration for pendrive_recovery package."""

from pathlib import Path
from setuptools import setup, find_packages

long_description = (Path(__file__).parent / "README.md").read_text(encoding="utf-8")

setup(
    name="pendrive_recovery",
    version="1.0.0",
    description="Offline AI-Based Pendrive Recovery & Face-Based File Organization System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Pendrive Recovery Team",
    python_requires=">=3.9",
    packages=find_packages(where=".", include=["src", "src.*"]),
    package_dir={"": "."},
    install_requires=[
        "numpy>=1.21.0",
        "opencv-python-headless>=4.5.0",
        "scikit-learn>=1.0.0",
        "Pillow>=9.0.0",
    ],
    extras_require={
        "ai": [
            "insightface>=0.7.0",
            "onnxruntime>=1.12.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "pendrive-recovery=main:main",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: System :: Recovery Tools",
    ],
)
