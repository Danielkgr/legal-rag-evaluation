"""
Legal document preprocessing module for Fair Work Act and modern awards.
"""

from .pdf_parser import PDFParser
from .chunking import LegalChunker
from .metadata_extractor import MetadataExtractor

__all__ = ["PDFParser", "LegalChunker", "MetadataExtractor"]
