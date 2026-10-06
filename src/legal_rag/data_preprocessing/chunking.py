"""
Legal-aware chunking module for Fair Work Act and modern awards.
Preserves legal structure while creating appropriately sized chunks.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import logging

from .pdf_parser import LegalDocument

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChunkType(Enum):
    """Types of chunks based on legal document structure."""
    DEFINITION = "definition"
    SECTION = "section"
    SUBSECTION = "subsection"
    LIST_ITEM = "list_item"
    PARAGRAPH = "paragraph"
    TABLE = "table"
    SCHEDULE = "schedule"
    PART = "part"
    DIVISION = "division"


@dataclass
class Chunk:
    """Represents a chunked piece of legal text."""
    text: str
    chunk_type: ChunkType
    section_number: Optional[str]
    page_numbers: List[int]
    start_char: int
    end_char: int
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class LegalChunker:
    """
    Chunks legal documents preserving structure and context.
    Focuses on creating semantically meaningful chunks for RAG.
    """
    
    def __init__(
        self,
        max_chunk_size: int = 512,
        min_chunk_size: int = 100,
        overlap_size: int = 50,
        preserve_section_structure: bool = True
    ):
        """
        Initialize the legal chunker.
        
        Args:
            max_chunk_size: Maximum tokens per chunk
            min_chunk_size: Minimum tokens per chunk
            overlap_size: Overlap between consecutive chunks
            preserve_section_structure: Whether to keep sections intact
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        self.overlap_size = overlap_size
        self.preserve_section_structure = preserve_section_structure
        
        # Patterns for legal structure
        self.section_pattern = re.compile(
            r'(?:Section|Sec|s)\.?\s*([\d]+(?:\.\d+)*)',
            re.IGNORECASE
        )
        self.subsection_pattern = re.compile(
            r'\(([\d]+)\)',
            re.MULTILINE
        )
        self.definition_pattern = re.compile(
            r'([“"\'\w\s]+?)\s+means\s+',
            re.IGNORECASE | re.MULTILINE
        )
        self.cross_ref_pattern = re.compile(
            r'(?:see|see also|refer to|pursuant to|defined in)\s+(?:section|s|sch)\.?\s*([\d\w\.\s,]+)',
            re.IGNORECASE
        )
    
    def chunk_document(self, document: LegalDocument) -> List[Chunk]:
        """
        Chunk a legal document into meaningful segments.
        
        Args:
            document: LegalDocument to chunk
            
        Returns:
            List of Chunk objects
        """
        chunks = []
        
        for page in document.pages:
            page_chunks = self._chunk_page(page, document)
            chunks.extend(page_chunks)
        
        # Merge chunks if they're too small (preserving structure)
        if self.preserve_section_structure:
            chunks = self._merge_small_chunks(chunks)
        
        return chunks
    
    def _chunk_page(self, page, document: LegalDocument) -> List[Chunk]:
        """Chunk a single page, preserving section boundaries."""
        chunks = []
        text = page.text
        
        if not text.strip():
            return chunks
        
        # Find section boundaries
        section_boundaries = self._find_section_boundaries(text)
        
        if section_boundaries:
            # Chunk by section
            chunks = self._chunk_by_sections(text, section_boundaries, page.page_number)
        else:
            # Fallback to size-based chunking
            chunks = self._chunk_by_size(text, page.page_number)
        
        return chunks
    
    def _find_section_boundaries(self, text: str) -> List[Tuple[int, str]]:
        """Find section boundaries in text."""
        boundaries = []
        
        for match in self.section_pattern.finditer(text):
            section_num = match.group(1)
            start_pos = match.start()
            
            # Try to find the full section heading
            line_start = text.rfind('\n', 0, start_pos) + 1
            line_end = text.find('\n', start_pos)
            if line_end == -1:
                line_end = len(text)
            
            heading = text[line_start:line_end].strip()
            boundaries.append((start_pos, section_num, heading))
        
        return sorted(boundaries, key=lambda x: x[0])
    
    def _chunk_by_sections(
        self,
        text: str,
        boundaries: List[Tuple],
        page_number: int
    ) -> List[Chunk]:
        """Chunk text by identified section boundaries."""
        chunks = []
        
        # Add boundaries for text start and end
        full_boundaries = [(0, None, "START")] + boundaries + [(len(text), None, "END")]
        
        for i in range(len(full_boundaries) - 1):
            start = full_boundaries[i][0]
            end = full_boundaries[i + 1][0]
            
            chunk_text = text[start:end].strip()
            
            # Skip if too short
            if len(chunk_text) < self.min_chunk_size:
                continue
            
            section_num = full_boundaries[i][1]
            heading = full_boundaries[i][2]
            
            # Determine chunk type
            chunk_type = self._determine_chunk_type(chunk_text, heading)
            
            chunks.append(Chunk(
                text=chunk_text,
                chunk_type=chunk_type,
                section_number=section_num,
                page_numbers=[page_number],
                start_char=start,
                end_char=end,
                metadata={
                    'heading': heading,
                    'document_type': 'section'
                }
            ))
        
        return chunks
    
    def _chunk_by_size(self, text: str, page_number: int) -> List[Chunk]:
        """Chunk text by size with overlap."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_start = 0
        
        for i, word in enumerate(words):
            current_chunk.append(word)
            current_text = ' '.join(current_chunk)
            
            # Check if chunk is too large
            if len(current_text) > self.max_chunk_size and len(current_chunk) > 10:
                chunk_text = ' '.join(current_chunk)
                
                # Determine chunk type
                chunk_type = self._determine_chunk_type(chunk_text, "")
                
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_type=chunk_type,
                    section_number=None,
                    page_numbers=[page_number],
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    metadata={'document_type': 'paragraph'}
                ))
                
                # Start new chunk with overlap
                overlap_words = int(self.overlap_size / 5)  # Approximate words
                current_chunk = current_chunk[-overlap_words:] if overlap_words > 0 else []
                current_start = current_start + len(chunk_text) - len(' '.join(current_chunk))
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunks.append(Chunk(
                text=chunk_text,
                chunk_type=self._determine_chunk_type(chunk_text, ""),
                section_number=None,
                page_numbers=[page_number],
                start_char=current_start,
                end_char=current_start + len(chunk_text),
                metadata={'document_type': 'paragraph'}
            ))
        
        return chunks
    
    def _determine_chunk_type(self, text: str, heading: str = "") -> ChunkType:
        """Determine the type of chunk based on content."""
        # Check for definitions
        if self.definition_pattern.search(text) or 'means' in text.lower()[:100]:
            return ChunkType.DEFINITION
        
        # Check for schedules
        if 'Schedule' in heading or text.startswith('Schedule'):
            return ChunkType.SCHEDULE
        
        # Check for lists
        if re.search(r'^[\s]*[\(\d]+[.\)]', text, re.MULTILINE):
            return ChunkType.LIST_ITEM
        
        # Check for tables (contains pipe characters or tabular data patterns)
        if '|' in text and '\n' in text:
            return ChunkType.TABLE
        
        # Check for sections
        if self.section_pattern.search(heading):
            return ChunkType.SECTION
        
        return ChunkType.PARAGRAPH
    
    def _merge_small_chunks(self, chunks: List[Chunk]) -> List[Chunk]:
        """Merge small chunks while preserving structure."""
        if not chunks:
            return chunks
        
        merged = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            # Check if merging would exceed max size
            combined_text = current_chunk.text + '\n\n' + next_chunk.text
            
            if len(combined_text) <= self.max_chunk_size:
                # Merge chunks
                current_chunk = Chunk(
                    text=combined_text,
                    chunk_type=self._determine_chunk_type(combined_text),
                    section_number=current_chunk.section_number or next_chunk.section_number,
                    page_numbers=current_chunk.page_numbers + next_chunk.page_numbers,
                    start_char=current_chunk.start_char,
                    end_char=next_chunk.end_char,
                    metadata={
                        'document_type': 'merged',
                        'merged_chunks': [current_chunk.metadata.get('chunk_id', 0), next_chunk.metadata.get('chunk_id', 0)]
                    }
                )
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk
        
        merged.append(current_chunk)
        return merged
    
    def get_cross_references(self, chunk: Chunk) -> List[str]:
        """Extract cross-references from a chunk."""
        references = []
        
        for match in self.cross_ref_pattern.finditer(chunk.text):
            ref_text = match.group(1).strip()
            # Parse multiple references
            refs = [r.strip() for r in ref_text.split(',')]
            references.extend(refs)
        
        return references


def main():
    """Example usage."""
    import sys
    from pathlib import Path
    
    if len(sys.argv) < 2:
        print("Usage: python chunker.py <pdf_path> [output_dir]")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    output_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("output")
    
    # Parse document
    from pdf_parser import PDFParser
    parser = PDFParser()
    doc = parser.parse_pdf(pdf_path)
    
    # Chunk document
    chunker = LegalChunker()
    chunks = chunker.chunk_document(doc)
    
    print(f"Document: {doc.title}")
    print(f"Total chunks: {len(chunks)}")
    print(f"\nSample chunks:")
    
    for i, chunk in enumerate(chunks[:3]):
        print(f"\n--- Chunk {i+1} ({chunk.chunk_type.value}, section: {chunk.section_number}) ---")
        print(f"Text: {chunk.text[:200]}...")


if __name__ == "__main__":
    main()
