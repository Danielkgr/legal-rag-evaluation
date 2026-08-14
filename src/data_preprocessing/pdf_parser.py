"""
PDF parsing module for Fair Work Act and modern awards documents.
"""

import pdfplumber
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PDFParser:
    """
    Parser for Fair Work Act and modern awards PDFs.
    Extracts text while preserving document structure.
    """
    
    def __init__(self):
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize regex patterns for legal document structure."""
        self.patterns = {
            'section': re.compile(r'(?:Section|Sec|s)\s*([\d]+(?:\.\d+)*)', re.IGNORECASE),
            'schedule': re.compile(r'Schedule\s+[\dA-Z]', re.IGNORECASE),
            'part': re.compile(r'Part\s+[\dA-Z]', re.IGNORECASE),
            'definition': re.compile(r'defined\s+(?:in|as)\s+(?:the\s+)?["\']?([\w\s]+?)["\']?(?:s)?'),
            'cross_ref': re.compile(r'(?:see|see also|refer to|pursuant to)\s+(?:section|s)\.?[\s]*([\d\w\.\s,]+)', re.IGNORECASE),
        }
    
    def parse_pdf(self, pdf_path: str, pages: Optional[List[int]] = None) -> LegalDocument:
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
                    doc_pages.append(DocumentPage(
                        page_number=page_num + 1,
                        text=text,
                        width=float(page.width),
                        height=float(page.height)
                    ))
                    full_text.append(text)
            
            # Extract document metadata
            metadata = self._extract_metadata(pdf, full_text)
            
            return LegalDocument(
                title=metadata.get('title', Path(pdf_path).stem),
                document_type=metadata.get('document_type', 'Unknown'),
                pages=doc_pages,
                metadata=metadata
            )
    
    def _extract_metadata(self, pdf, full_text: List[str]) -> Dict:
        """Extract document metadata from the PDF."""
        metadata = {}
        
        # Try to get title from PDF metadata
        if pdf.metadata:
            metadata.update(pdf.metadata)
        
        # Extract document type from text
        full_text_str = '\n'.join(full_text[:5000])  # Check first 5000 chars
        
        if re.search(r'Fair\s+Work\s+Act', full_text_str, re.IGNORECASE):
            metadata['document_type'] = 'Act'
            metadata['title'] = 'Fair Work Act 2009'
        elif re.search(r'Modern\s+Award', full_text_str, re.IGNORECASE):
            metadata['document_type'] = 'Award'
            # Try to extract award name
            award_match = re.search(r'Modern\s+Award\s*[:–-]?\s*(.+?)(?:\n|$)', full_text_str, re.IGNORECASE)
            if award_match:
                metadata['award_name'] = award_match.group(1).strip()
        elif re.search(r'Regulation', full_text_str, re.IGNORECASE):
            metadata['document_type'] = 'Regulation'
        
        # Extract effective date
        date_match = re.search(r'([1-9]?\d{1,3}[/-]\d{1,2}[/-]\d{2,4})', full_text_str)
        if date_match:
            metadata['effective_date'] = date_match.group(1)
        
        return metadata
    
    def parse_multiple_pdfs(self, pdf_paths: List[str]) -> List[LegalDocument]:
        """Parse multiple PDF documents."""
        return [self.parse_pdf(path) for path in pdf_paths]


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pdf_parser.py <pdf_path> [page1 page2 ...]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    pages = [int(p) for p in sys.argv[2:]] if len(sys.argv) > 2 else None
    
    parser = PDFParser()
    doc = parser.parse_pdf(pdf_path, pages)
    
    print(f"Document: {doc.title}")
    print(f"Type: {doc.document_type}")
    print(f"Pages: {len(doc.pages)}")
    print(f"\nFirst 500 chars of first page:\n{doc.pages[0].text[:500]}...")


if __name__ == "__main__":
    main()
