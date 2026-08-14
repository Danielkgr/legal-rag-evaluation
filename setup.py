"""Minimal setuptools scaffold for installing fairwork-rag as a CLI."""

from setuptools import setup, find_packages

setup(
    name="fairwork-rag",
    version="0.1.0",
    description="Interactive CLI for the Fair Work Act & Modern Awards RAG System",
    # Packages live directly under src/ — e.g. data_preprocessing, embedding, etc.
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.10",
    install_requires=[
        "openai",
        "chromadb",
        "rank-bm25",
        "transformers",
        "bitsandbytes",
        "pdfplumber",
        "pymupdf",
    ],
    entry_points={
        "console_scripts": [
            "fairwork-rag=fair_work_rag.cli:main",
        ],
    },
)
