"""
Evaluation set generation module.
Creates test queries with gold-standard relevance annotations.
"""

import json
import random
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Query:
    """Represents a test query with metadata."""
    query_id: str
    query_text: str
    query_type: str  # 'fact', 'hypothetical', 'cross-reference'
    difficulty: str  # 'easy', 'medium', 'hard'
    expected_sections: List[str]
    expected_documents: List[str]
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class RelevanceAnnotation:
    """Represents a relevance judgment."""
    query_id: str
    chunk_id: str
    relevance_score: int  # 0=irrelevant, 1=relevant, 2=highly relevant
    reason: str = ""


class EvaluationSetGenerator:
    """
    Generates evaluation sets for RAG systems.
    Focuses on legal document retrieval with high precision requirements.
    """
    
    def __init__(self, output_dir: str = "evaluation_set"):
        """
        Initialize the evaluation set generator.
        
        Args:
            output_dir: Directory to save evaluation sets
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.queries: List[Query] = []
        self.annotations: List[RelevanceAnnotation] = []
        
        logger.info(f"Initialized EvaluationSetGenerator at {output_dir}")
    
    def generate_queries(
        self,
        chunks: List[Dict],
        num_queries: int = 100,
        query_type_distribution: Dict[str, float] = None
    ) -> List[Query]:
        """
        Generate test queries from available chunks.
        
        Args:
            chunks: List of chunk dicts with 'text', 'section_number', 'document_name'
            num_queries: Number of queries to generate
            query_type_distribution: Distribution of query types
            
        Returns:
            List of generated queries
        """
        if query_type_distribution is None:
            query_type_distribution = {
                'fact': 0.4,
                'hypothetical': 0.35,
                'cross-reference': 0.25
            }
        
        # Group chunks by document and section
        self.chunks_by_doc = defaultdict(list)
        self.chunks_by_section = defaultdict(list)
        
        for chunk in chunks:
            doc = chunk.get('document_name', 'unknown')
            section = chunk.get('section_number', 'unknown')
            
            self.chunks_by_doc[doc].append(chunk)
            self.chunks_by_section[section].append(chunk)
        
        # Generate queries by type
        queries = []
        
        # Fact-based queries (40%)
        fact_count = int(num_queries * query_type_distribution['fact'])
        queries.extend(self._generate_fact_queries(fact_count))
        
        # Hypothetical queries (35%)
        hypo_count = int(num_queries * query_type_distribution['hypothetical'])
        queries.extend(self._generate_hypothetical_queries(hypo_count))
        
        # Cross-reference queries (25%)
        cross_count = num_queries - fact_count - hypo_count
        queries.extend(self._generate_cross_reference_queries(cross_count))
        
        self.queries = queries
        logger.info(f"Generated {len(queries)} queries")
        
        return queries
    
    def _generate_fact_queries(self, count: int) -> List[Query]:
        """Generate fact-based definition and section questions."""
        queries = []
        
        # Sample chunks for fact queries
        sample_chunks = random.sample(self.chunks_by_section.values(), min(count, len(self.chunks_by_section)))
        
        for i, chunk_list in enumerate(sample_chunks[:count]):
            chunk = random.choice(chunk_list)
            section = chunk.get('section_number', 'unknown')
            doc = chunk.get('document_name', 'unknown')
            text = chunk.get('text', '')
            
            # Generate different fact query patterns
            patterns = [
                lambda t, s: f"What is defined in Section {s}?",
                lambda t, s: f"What does Section {s} say about",
                lambda t, s: f"Find the definition in Section {s}",
                lambda t, s: f"What is the text of Section {s}?",
                lambda t, s: f"Explain Section {s} of the {doc}",
            ]
            
            # Extract key terms from text for more specific queries
            key_terms = self._extract_key_terms(text, max_terms=3)
            
            for j, pattern in enumerate(random.sample(patterns, min(2, len(patterns)))):
                query_id = f"fact_{i}_{j}"
                
                if key_terms:
                    # Use key term for more specific query
                    query_text = f"What is the definition of '{key_terms[0]}' in the {doc}?"
                    expected = [section]
                else:
                    query_text = pattern(text, section)
                    expected = [section]
                
                queries.append(Query(
                    query_id=query_id,
                    query_text=query_text,
                    query_type='fact',
                    difficulty='easy',
                    expected_sections=expected,
                    expected_documents=[doc],
                    metadata={'pattern': str(pattern), 'key_terms': key_terms}
                ))
                
                if len(queries) >= count:
                    return queries
        
        return queries[:count]
    
    def _generate_hypothetical_queries(self, count: int) -> List[Query]:
        """Generate hypothetical scenario questions."""
        queries = []
        
        # Common hypothetical patterns for workplace law
        patterns = [
            ("overtime", "Can an employer require an employee to work overtime?",
             "overtime requirements"),
            ("annual_leave", "How much annual leave is an employee entitled to?",
             "annual leave entitlements"),
            ("personal_leave", "What are the rules for personal/carer's leave?",
             "personal leave"),
            ("redundancy", "When is an employee entitled to redundancy pay?",
             "redundancy pay"),
            ("unfair_dismissal", "What constitutes unfair dismissal?",
             "unfair dismissal"),
            ("flexible_work", "Can employees request flexible working arrangements?",
             "flexible work arrangements"),
            ("notice_period", "How much notice is required to terminate employment?",
             "notice period"),
            ("pay_period", "How often must employees be paid?",
             "pay period requirements"),
        ]
        
        for i, (key, pattern, _) in enumerate(patterns):
            query_id = f"hypo_{i}"
            queries.append(Query(
                query_id=query_id,
                query_text=pattern,
                query_type='hypothetical',
                difficulty='medium',
                expected_sections=[],
                expected_documents=['fair_work_act_2009'],
                metadata={'topic': key, 'category': 'entitlements'}
            ))
        
        # Fill remaining with variations
        while len(queries) < count:
            i = len(queries)
            topics = ['leave', 'hours', 'pay', 'termination', 'agreement']
            topic = random.choice(topics)
            
            patterns = [
                f"What are the rules regarding {topic} in the Fair Work Act?",
                f"Explain the provisions about {topic}",
                f"How does the Act handle {topic}?",
            ]
            
            queries.append(Query(
                query_id=f"hypo_{i}",
                query_text=random.choice(patterns),
                query_type='hypothetical',
                difficulty='medium',
                expected_sections=[],
                expected_documents=['fair_work_act_2009'],
                metadata={'topic': topic}
            ))
        
        return queries[:count]
    
    def _generate_cross_reference_queries(self, count: int) -> List[Query]:
        """Generate queries requiring cross-referencing multiple provisions."""
        queries = []
        
        # Cross-reference patterns
        patterns = [
            ("definition_cross", "According to the definition of 'employee', what conditions must be met?",
             ["definition", "employee"]),
            ("lookup_chain", "What is the definition of 'modern award' and where is it referenced?",
             ["definition", "modern award"]),
            ("cross_section", "How do sections 31 and 32 interact regarding national employment standards?",
             ["section_31", "section_32"]),
        ]
        
        for i, (_, pattern, expected) in enumerate(patterns):
            query_id = f"cross_{i}"
            queries.append(Query(
                query_id=query_id,
                query_text=pattern,
                query_type='cross-reference',
                difficulty='hard',
                expected_sections=expected,
                expected_documents=['fair_work_act_2009'],
                metadata={'type': 'cross-reference', 'complexity': 'multi-section'}
            ))
        
        # Fill remaining
        while len(queries) < count:
            i = len(queries)
            queries.append(Query(
                query_id=f"cross_{i}",
                query_text="How are the provisions about unfair dismissal connected to other relevant sections?",
                query_type='cross-reference',
                difficulty='hard',
                expected_sections=[],
                expected_documents=['fair_work_act_2009'],
                metadata={'type': 'connection_search', 'complexity': 'multi-section'}
            ))
        
        return queries[:count]
    
    def _extract_key_terms(self, text: str, max_terms: int = 5) -> List[str]:
        """Extract key terms from text."""
        # Simple keyword extraction - could use NLP for better results
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                     'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were', 'be',
                     'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
                     'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall'}
        
        words = text.lower().split()
        word_counts = defaultdict(int)
        
        for word in words:
            # Clean word
            word = word.strip('.,;:!?()"\'').strip()
            if word and word not in stop_words and len(word) > 2:
                word_counts[word] += 1
        
        # Get most common words
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:max_terms]]
    
    def generate_annotations(
        self,
        chunks: List[Dict],
        auto_annotate: bool = True
    ) -> List[RelevanceAnnotation]:
        """
        Generate relevance annotations for chunks.
        
        Args:
            chunks: List of chunk dicts
            auto_annotate: Whether to auto-annotate based on section matching
            
        Returns:
            List of relevance annotations
        """
        annotations = []
        
        if auto_annotate and self.queries:
            # Auto-annotate based on section matching
            chunks_by_id = {c.get('chunk_id', f"chunk_{i}"): c for i, c in enumerate(chunks)}
            
            for query in self.queries:
                for chunk_id in chunks_by_id:
                    chunk = chunks_by_id[chunk_id]
                    section = chunk.get('section_number')
                    doc = chunk.get('document_name')
                    
                    # Determine relevance
                    if section in query.expected_sections or doc in query.expected_documents:
                        relevance = 2  # Highly relevant
                    elif section and any(sec in section for sec in query.expected_sections):
                        relevance = 1  # Somewhat relevant
                    else:
                        relevance = 0  # Irrelevant
                    
                    annotations.append(RelevanceAnnotation(
                        query_id=query.query_id,
                        chunk_id=chunk_id,
                        relevance_score=relevance,
                        reason=f"Section match: {section} in {query.expected_sections}"
                    ))
        
        self.annotations = annotations
        logger.info(f"Generated {len(annotations)} annotations")
        
        return annotations
    
    def save_evaluation_set(self):
        """Save evaluation set to files."""
        # Save queries
        queries_path = self.output_dir / "queries.json"
        with open(queries_path, 'w') as f:
            json.dump([asdict(q) for q in self.queries], f, indent=2)
        logger.info(f"Saved queries to {queries_path}")
        
        # Save annotations
        annotations_path = self.output_dir / "annotations.json"
        with open(annotations_path, 'w') as f:
            json.dump([asdict(a) for a in self.annotations], f, indent=2)
        logger.info(f"Saved annotations to {annotations_path}")
        
        # Save summary
        summary = {
            'num_queries': len(self.queries),
            'query_type_distribution': {},
            'num_annotations': len(self.annotations)
        }
        
        for q in self.queries:
            qtype = q.query_type
            summary['query_type_distribution'][qtype] = summary['query_type_distribution'].get(qtype, 0) + 1
        
        summary_path = self.output_dir / "summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_path}")
    
    def load_evaluation_set(self) -> bool:
        """Load evaluation set from files."""
        queries_path = self.output_dir / "queries.json"
        annotations_path = self.output_dir / "annotations.json"
        
        if not queries_path.exists() or not annotations_path.exists():
            return False
        
        with open(queries_path, 'r') as f:
            queries_data = json.load(f)
        self.queries = [Query(**q) for q in queries_data]
        
        with open(annotations_path, 'r') as f:
            annotations_data = json.load(f)
        self.annotations = [RelevanceAnnotation(**a) for a in annotations_data]
        
        logger.info(f"Loaded evaluation set: {len(self.queries)} queries, {len(self.annotations)} annotations")
        return True


def main():
    """Example usage."""
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python eval_generator.py <chunks_json> [num_queries]")
        sys.exit(1)
    
    chunks_file = sys.argv[1]
    num_queries = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    # Load chunks
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    # Generate evaluation set
    generator = EvaluationSetGenerator()
    queries = generator.generate_queries(chunks, num_queries)
    annotations = generator.generate_annotations(chunks)
    generator.save_evaluation_set()
    
    print(f"Generated evaluation set with {len(queries)} queries")


if __name__ == "__main__":
    main()
