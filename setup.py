"""Setup configuration for Model Unfetter."""

from setuptools import setup, find_packages
from pathlib import Path

readme = Path(__file__).parent / "README.md"
long_description = readme.read_text(encoding="utf-8") if readme.exists() else ""

setup(
    name="model-unfetter",
    version="0.1.0",
    author="HOLYKEYZ",
    description="Multi-tier model unalignment framework using directional ablation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/HOLYKEYZ/model-unfetter",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "unfetter": ["datasets/*.json"],
    },
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "safetensors>=0.4.0",
        "click>=8.0.0",
        "psutil>=5.9.0",
    ],
    extras_require={
        "gpu": [
            "bitsandbytes>=0.41.0",
            "accelerate>=0.24.0",
        ],
        "datasets": [
            "datasets>=2.16.0",
        ],
        "full": [
            "bitsandbytes>=0.41.0",
            "accelerate>=0.24.0",
            "datasets>=2.16.0",
            "optuna>=3.4.0",
            "rich>=13.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "ruff>=0.1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "unfetter=unfetter.cli.main:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
