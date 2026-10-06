"""
Metadata extraction module for legal documents.
Extracts cross-references, definitions, and document structure.
"""

import re
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Definition:
    """Represents a defined term in the legislation."""
    term: str
    definition_text: str
    section_number: Optional[str]
    page_number: Optional[int]


@dataclass
class CrossReference:
    """Represents a cross-reference to another provision."""
    source_section: str
    target_sections: List[str]
    reference_text: str
    context: str


@dataclass
class LegalStructure:
    """Represents the overall structure of a legal document."""
    parts: List[Dict]
    divisions: List[Dict]
    sections: List[Dict]
    schedules: List[Dict]


class MetadataExtractor:
    """
    Extracts structured metadata from legal documents.
    Focuses on definitions, cross-references, and document structure.
    """
    
    def __init__(self):
        self._init_patterns()
    
    def _init_patterns(self):
        """Initialize regex patterns for metadata extraction."""
        self.patterns = {
            # Definition patterns
            'definition_simple': re.compile(
                r'([“"\'\w\s]+?)\s+means\s+([^;.:]+?)(?:;|\.|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            'definition_includes': re.compile(
                r'([“"\'\w\s]+?)\+?includes\s+([^;.:]+?)(?:;|\.|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            'definition_inferred': re.compile(
                r'has\s+the\s+meaning\s+(?:of|given\s+to)\s+([“"\'\w\s]+?)(?:;|\.|$)',
                re.IGNORECASE | re.MULTILINE
            ),
            
            # Cross-reference patterns
            'cross_ref_see': re.compile(
                r'(?:see|see also|refer to|pursuant to)\s+(?:section|s|sch)\.?\s*([\d\w\.\s,]+)',
                re.IGNORECASE
            ),
            'cross_ref_defined_in': re.compile(
                r'defined\s+(?:in|at)\s+(?:section|s)\.?\s*([\d\w\.\s,]+)',
                re.IGNORECASE
            ),
            
            # Section patterns
            'section_heading': re.compile(
                r'(?:Section|Part|Schedule)\s+([\dA-Z]+(?:\.\d+)*)\s*(.*)',
                re.IGNORECASE
            ),
        }
    
    def extract_definitions(self, text: str, section_num: str = None, page_num: int = None) -> List[Definition]:
        """
        Extract all defined terms from text.
        
        Args:
            text: Text to extract definitions from
            section_num: Section number where definitions appear
            page_num: Page number where definitions appear
            
        Returns:
            List of Definition objects
        """
        definitions = []
        
        # Try different definition patterns
        patterns = [
            self.patterns['definition_simple'],
            self.patterns['definition_includes'],
            self.patterns['definition_inferred']
        ]
        
        for pattern in patterns:
            for match in pattern.finditer(text):
                term = match.group(1).strip().strip('""\'')
                definition_text = match.group(2).strip()
                
                definitions.append(Definition(
                    term=term,
                    definition_text=definition_text,
                    section_number=section_num,
                    page_number=page_num
                ))
        
        # Remove duplicates (keep first occurrence)
        seen_terms = set()
        unique_definitions = []
        for defn in definitions:
            term_lower = defn.term.lower()
            if term_lower not in seen_terms:
                seen_terms.add(term_lower)
                unique_definitions.append(defn)
        
        return unique_definitions
    
    def extract_cross_references(self, text: str, source_section: str = None) -> List[CrossReference]:
        """
        Extract cross-references from text.
        
        Args:
            text: Text to extract references from
            source_section: Source section number
            
        Returns:
            List of CrossReference objects
        """
        references = []
        
        for match in self.patterns['cross_ref_see'].finditer(text):
            ref_text = match.group(1).strip()
            
            # Parse multiple references (comma-separated)
            targets = []
            for ref in ref_text.split(','):
                ref = ref.strip()
                # Try to extract section numbers
                section_nums = re.findall(r'\d+(?:\.\d+)*', ref)
                targets.extend(section_nums)
            
            if targets:
                references.append(CrossReference(
                    source_section=source_section,
                    target_sections=targets,
                    reference_text=match.group(0),
                    context=text[max(0, match.start()-100):min(len(text), match.end()+100)]
                ))
        
        # Also check for "defined in" references
        for match in self.patterns['cross_ref_defined_in'].finditer(text):
            ref_text = match.group(1).strip()
            targets = re.findall(r'\d+(?:\.\d+)*', ref_text)
            
            if targets:
                references.append(CrossReference(
                    source_section=source_section,
                    target_sections=targets,
                    reference_text=match.group(0),
                    context=text[max(0, match.start()-100):min(len(text), match.end()+100)]
                ))
        
        return references
    
    def extract_section_structure(self, text: str) -> Dict[str, List[str]]:
        """
        Extract section hierarchy from text.
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict mapping section numbers to their content
        """
        sections = defaultdict(list)
        
        # Find all section headings
        lines = text.split('\n')
        current_section = None
        
        for line in lines:
            match = self.patterns['section_heading'].match(line.strip())
            if match:
                section_num = match.group(1)
                section_title = match.group(2).strip()
                current_section = section_num
                sections[section_num].append(f"HEADING: {section_num} - {section_title}")
            elif current_section:
                sections[current_section].append(line)
        
        return dict(sections)
    
    def extract_definitions_index(self, documents: List) -> Dict[str, List[Definition]]:
        """
        Build a cross-document index of definitions.
        
        Args:
            documents: List of LegalDocument objects
            
        Returns:
            Dict mapping terms to their definitions across documents
        """
        definitions_index = defaultdict(list)
        
        for doc in documents:
            for page in doc.pages:
                page_defs = self.extract_definitions(page.text, page_num=page.page_number)
                for definition in page_defs:
                    term_lower = definition.term.lower()
                    definitions_index[term_lower].append(definition)
        
        return dict(definitions_index)
    
    def extract_citations(self, text: str) -> List[str]:
        """
        Extract legislative citations from text.
        
        Args:
            text: Text to extract citations from
            
        Returns:
            List of citation strings
        """
        citations = []
        
        # Patterns for citations
        citation_patterns = [
            r'Fair\s+Work\s+Act\s+(?:\d{4})?',
            r'Fair\s+Work\s+Regulations\s+(?:\d{4})?',
            r'(?:Modern\s+)?Award\s+(?:\w+\s+)?(?:\d{4})?',
            r'section\s+\d+(?:\.\d+)*',
            r's\s+\d+(?:\.\d+)*',
        ]
        
        for pattern in citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            citations.extend(matches)
        
        return list(set(citations))


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python metadata_extractor.py <text_file>")
        sys.exit(1)
    
    text_file = sys.argv[1]
    
    with open(text_file, 'r') as f:
        text = f.read()
    
    extractor = MetadataExtractor()
    
    # Extract definitions
    definitions = extractor.extract_definitions(text)
    print(f"Found {len(definitions)} definitions:")
    for defn in definitions[:5]:
        print(f"  - {defn.term}: {defn.definition_text[:100]}...")
    
    # Extract cross-references
    refs = extractor.extract_cross_references(text)
    print(f"\nFound {len(refs)} cross-references:")
    for ref in refs[:5]:
        print(f"  - Section {ref.source_section} -> {ref.target_sections}")
    
    # Extract citations
    citations = extractor.extract_citations(text)
    print(f"\nFound {len(citations)} citations:")
    for cit in citations[:10]:
        print(f"  - {cit}")


if __name__ == "__main__":
    main()
