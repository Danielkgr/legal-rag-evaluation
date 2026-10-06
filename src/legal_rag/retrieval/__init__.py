"""
Hybrid retrieval system combining vector search and BM25 for legal documents.
"""

import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from rank_bm25 import BM25Okapi

from legal_rag.data_preprocessing.metadata_extractor import MetadataExtractor
from legal_rag.embedding import EmbeddingManager

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Represents a retrieval result."""

    chunk_id: str
    text: str
    section_number: Optional[str]
    document_name: str
    chunk_type: str
    vector_score: float
    bm25_score: float
    combined_score: float
    metadata: Optional[Dict] = None
    # Set on chunks added after retrieval: "cross_reference" or "definition".
    expansion: Optional[str] = None
    expansion_reason: Optional[str] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HybridRetriever:
    """
    Hybrid retrieval combining vector embeddings and BM25 through reciprocal
    rank fusion.
    """

    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Args:
            embedding_manager: EmbeddingManager with loaded embeddings
        """
        self.embedding_manager = embedding_manager

        # Build BM25 index
        self.bm25 = None
        self.tokenized_texts: List[List[str]] = []
        self._build_bm25_index()
        self._build_structure_index()

        logger.info("Initialized HybridRetriever")

    def _build_bm25_index(self):
        """Build BM25 index from embedded chunks."""
        chunks = self.embedding_manager.embeddings

        if not chunks:
            logger.warning("No chunks to build BM25 index")
            return

        # Tokenize texts
        self.tokenized_texts = []
        for chunk in chunks:
            # Simple tokenization - could be improved
            tokens = chunk.text.lower().split()
            self.tokenized_texts.append(tokens)

        # Build BM25
        self.bm25 = BM25Okapi(self.tokenized_texts)
        logger.info(f"Built BM25 index with {len(self.tokenized_texts)} documents")

    def retrieve(
        self,
        query: str,
        k: int = 10,
        filter_document: str = None,
        filter_chunk_type: str = None,
        expand: bool = False,
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using hybrid search.

        Args:
            query: Query string
            k: Number of results to return
            filter_document: Filter by document name
            filter_chunk_type: Filter by chunk type
            expand: Append the sections the results cite and the definitions
                of terms they use, after the k results and at lower scores

        Returns:
            List of RetrievalResult objects
        """
        # Vector search
        vector_results = self.embedding_manager.search_by_text(
            query,
            k=k * 2,  # Get more for fusion
            filter_document=filter_document,
            filter_chunk_type=filter_chunk_type,
        )

        # BM25 search
        bm25_results = self._bm25_search(
            query,
            k=k * 2,
            filter_document=filter_document,
            filter_chunk_type=filter_chunk_type,
        )

        combined = self._reciprocal_rank_fusion(vector_results, bm25_results, k=k)

        if expand:
            combined = combined + self.expand(combined, query)
        return combined

    def _build_structure_index(self):
        """Index each document's sections, schedules, and defined terms."""
        extractor = MetadataExtractor()
        self._first_chunk: Dict[Tuple[str, Optional[str], Optional[str]], int] = {}
        self._definitions: Dict[
            str, List[Tuple[re.Pattern, str, int, Optional[Tuple]]]
        ] = defaultdict(list)
        for index, record in enumerate(self.embedding_manager.embeddings):
            doc = record.document_name
            schedule = (record.metadata or {}).get("schedule")
            sections = (record.metadata or {}).get("sections") or [
                record.section_number
            ]
            self._first_chunk.setdefault((doc, schedule, None), index)
            for section in sections:
                if section:
                    self._first_chunk.setdefault(
                        (doc, schedule, section.upper()), index
                    )
            for definition in extractor.extract_definitions(record.text):
                target = None
                if definition.target_section or definition.target_schedule:
                    target = (definition.target_schedule, definition.target_section)
                self._definitions[doc].append(
                    (_term_pattern(definition.term), definition.term, index, target)
                )

    def expand(
        self,
        results: List[RetrievalResult],
        query: str,
        max_cross_references: int = 3,
        max_definitions: int = 3,
    ) -> List[RetrievalResult]:
        """
        Return chunks to add after the retrieved ones, without duplicates.

        Cross-references come first, in the rank order of the chunks that
        cite them, then definitions of terms that appear in the question or
        in a retrieved chunk, question terms first and longer terms first.
        Each added chunk scores half the lowest retrieved score, so it sorts
        after everything that was actually retrieved.
        """
        if not results:
            return []
        records = self.embedding_manager.embeddings
        present = {r.chunk_id for r in results}
        score = min(r.combined_score for r in results) / 2
        extractor = MetadataExtractor()
        added: List[RetrievalResult] = []

        def add(index: int, kind: str, reason: str) -> bool:
            record = records[index]
            if record.chunk_id in present:
                return False
            present.add(record.chunk_id)
            added.append(_as_result(record, score, kind, reason))
            return True

        cross_added = 0
        for result in results:
            if cross_added >= max_cross_references:
                break
            own = (result.metadata or {}).get("sections") or [result.section_number]
            for target in extractor.extract_cross_references(result.text, own):
                if target.startswith("Schedule "):
                    key = (result.document_name, target.split()[1], None)
                else:
                    key = (result.document_name, None, target)
                index = self._first_chunk.get(key)
                label = target if target.startswith("Schedule") else f"s {target}"
                if index is not None and add(
                    index, "cross_reference", f"cited as {label} by {result.chunk_id}"
                ):
                    cross_added += 1
                    if cross_added >= max_cross_references:
                        break

        candidates = []
        documents = list(dict.fromkeys(r.document_name for r in results))
        for doc in documents:
            texts = " ".join(r.text for r in results if r.document_name == doc)
            for pattern, term, index, target in self._definitions.get(doc, []):
                in_query = bool(pattern.search(query))
                if in_query or pattern.search(texts):
                    candidates.append(
                        (
                            not in_query,
                            -len(term.split()),
                            -len(term),
                            doc,
                            term,
                            index,
                            target,
                        )
                    )
        candidates.sort(key=lambda c: c[:3])
        definitions_added = 0
        for _, _, _, doc, term, index, target in candidates:
            if definitions_added >= max_definitions:
                break
            if target is not None:
                index = self._first_chunk.get((doc, target[0], target[1]), index)
            if add(index, "definition", f"defines '{term}'"):
                definitions_added += 1
        return added

    def _bm25_search(
        self,
        query: str,
        k: int = 10,
        filter_document: str = None,
        filter_chunk_type: str = None,
    ) -> List[Dict]:
        """Search using BM25."""
        # Tokenize query
        query_tokens = query.lower().split()

        # Get scores
        scores = self.bm25.get_scores(query_tokens)

        # Build results
        results = []
        chunks = self.embedding_manager.embeddings

        for i, score in enumerate(scores):
            chunk = chunks[i]

            # Apply filters
            if filter_document and chunk.document_name != filter_document:
                continue
            if filter_chunk_type and chunk.chunk_type != filter_chunk_type:
                continue

            results.append({"chunk": chunk, "score": float(score)})

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:k]

    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 10,
        rrf_k: int = 60,
    ) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).

        RRF score = 1/(k + rank_vector) + 1/(k + rank_bm25)
        """
        vector_ranks = {r["chunk"].chunk_id: i for i, r in enumerate(vector_results, 1)}
        bm25_ranks = {r["chunk"].chunk_id: i for i, r in enumerate(bm25_results, 1)}
        vector_scores = {r["chunk"].chunk_id: r["score"] for r in vector_results}
        bm25_scores = {r["chunk"].chunk_id: r["score"] for r in bm25_results}
        chunks_by_id = {r["chunk"].chunk_id: r["chunk"] for r in vector_results}
        for r in bm25_results:
            chunks_by_id.setdefault(r["chunk"].chunk_id, r["chunk"])

        def fused(chunk_id: str) -> float:
            score = 0.0
            for ranks in (vector_ranks, bm25_ranks):
                if chunk_id in ranks:
                    score += 1.0 / (rrf_k + ranks[chunk_id])
            return score

        # chunks_by_id keeps first-seen order, so ties break the same way on
        # every run.
        ranked = sorted(chunks_by_id, key=fused, reverse=True)[:k]

        results = []
        for chunk_id in ranked:
            chunk = chunks_by_id[chunk_id]
            score = fused(chunk_id)
            vector_score = vector_scores.get(chunk_id, 0.0)
            bm25_score = bm25_scores.get(chunk_id, 0.0)
            results.append(
                RetrievalResult(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    section_number=chunk.section_number,
                    document_name=chunk.document_name,
                    chunk_type=chunk.chunk_type,
                    vector_score=vector_score,
                    bm25_score=bm25_score,
                    combined_score=score,
                    metadata=chunk.metadata,
                )
            )

        return results


def _term_pattern(term: str) -> re.Pattern:
    """Match a defined term as a whole phrase, plural allowed."""
    flags = re.IGNORECASE if term.islower() else 0
    return re.compile(r"(?<![\w-])" + re.escape(term) + r"s?(?![\w-])", flags)


def _as_result(record, score: float, kind: str, reason: str) -> RetrievalResult:
    return RetrievalResult(
        chunk_id=record.chunk_id,
        text=record.text,
        section_number=record.section_number,
        document_name=record.document_name,
        chunk_type=record.chunk_type,
        vector_score=0.0,
        bm25_score=0.0,
        combined_score=score,
        metadata=record.metadata,
        expansion=kind,
        expansion_reason=reason,
    )
