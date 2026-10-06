"""
PDF parsing module for Fair Work Act and modern awards documents.
"""

import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import pdfplumber

logger = logging.getLogger(__name__)

# The first page of an Act, regulation, or award names it on a line of its
# own, for example "Spam Act 2003" or "Fair Work Regulations 2009".
TITLE_LINE = re.compile(
    r"^\s*(?P<title>[A-Z][A-Za-z ,'()-]{1,120}? (?P<kind>Act|Regulations|Rules|Award)"
    r" \d{4})\s*$"
)


@dataclass
class DocumentPage:
    """Represents a single page of parsed text."""

    page_number: int
    text: str
    width: float
    height: float


@dataclass
class LegalDocument:
    """Represents a parsed legal document."""

    title: str
    document_type: str  # 'Act', 'Award', 'Regulation', etc.
    pages: List[DocumentPage]
    metadata: Optional[Dict] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PDFParser:
    """
    Parser for Fair Work Act and modern awards PDFs.
    Extracts text while preserving document structure.
    """

    def parse_pdf(
        self, pdf_path: str, pages: Optional[List[int]] = None
    ) -> LegalDocument:
        """
        Parse a PDF document and extract text with structure.

        Args:
            pdf_path: Path to the PDF file
            pages: Optional list of page numbers to parse (1-indexed)

        Returns:
            LegalDocument object with parsed content
        """
        logger.info(f"Parsing PDF: {pdf_path}")

        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            logger.info(f"Total pages: {total_pages}")

            # Determine which pages to parse
            if pages:
                pages_to_parse = [p - 1 for p in pages if p <= total_pages]
            else:
                pages_to_parse = range(total_pages)

            doc_pages = []
            full_text = []

            for page_num in pages_to_parse:
                page = pdf.pages[page_num]
                text = page.extract_text(x_tolerance=3, y_tolerance=3)

                if text:
                    doc_pages.append(
                        DocumentPage(
                            page_number=page_num + 1,
                            text=text,
                            width=float(page.width),
                            height=float(page.height),
                        )
                    )
                    full_text.append(text)

            # Extract document metadata
            metadata = self._extract_metadata(pdf, full_text)

            return LegalDocument(
                title=metadata.get("title", Path(pdf_path).stem),
                document_type=metadata.get("document_type", "Unknown"),
                pages=doc_pages,
                metadata=metadata,
            )

    def _extract_metadata(self, pdf, full_text: List[str]) -> Dict:
        """Read the document's title and type from its first page."""
        metadata = {}
        if pdf.metadata:
            metadata.update(pdf.metadata)

        first_page = full_text[0] if full_text else ""
        for line in first_page.splitlines()[:15]:
            match = TITLE_LINE.match(line)
            if match:
                metadata["title"] = match.group("title")
                metadata["document_type"] = match.group("kind").rstrip("s")
                break
        return metadata
