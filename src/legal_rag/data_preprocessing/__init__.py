"""
Legal document preprocessing module for Fair Work Act and modern awards.
"""

from .chunking import LegalChunker
from .metadata_extractor import MetadataExtractor
from .pdf_parser import PDFParser

__all__ = ["PDFParser", "LegalChunker", "MetadataExtractor"]
