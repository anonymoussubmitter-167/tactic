"""
Setup script for TACTIC-Kinetics.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read() if fh else ""

setup(
    name="tactic-kinetics",
    version="0.1.0",
    author="TACTIC Team",
    description="Thermodynamic-Native Inference for Enzyme Mechanism Discovery",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "torchdiffeq>=0.2.0",
        "tqdm>=4.62.0",
    ],
    extras_require={
        "equilibrator": [
            "equilibrator-api>=0.4.0",
        ],
        "data": [
            "pandas>=1.3.0",
            "openpyxl>=3.0.0",
            "xlrd>=2.0.0",
        ],
        "api": [
            "requests>=2.26.0",
            "zeep>=4.0.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=21.0.0",
            "isort>=5.0.0",
            "flake8>=3.9.0",
        ],
        "all": [
            "equilibrator-api>=0.4.0",
            "pandas>=1.3.0",
            "openpyxl>=3.0.0",
            "xlrd>=2.0.0",
            "requests>=2.26.0",
            "zeep>=4.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
