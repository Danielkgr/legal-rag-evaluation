"""
Hybrid retrieval system combining vector search and BM25 for legal documents.
"""

import os
import json
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from collections import defaultdict
import logging
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from embedding import EmbeddingManager

logging.basicConfig(level=logging.INFO)
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
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class HybridRetriever:
    """
    Hybrid retrieval combining vector embeddings and BM25.
    Implements reciprocal rank fusion for optimal results.
    """
    
    def __init__(
        self,
        embedding_manager: EmbeddingManager,
        alpha: float = 0.5,
        beta: float = 0.5
    ):
        """
        Initialize the hybrid retriever.
        
        Args:
            embedding_manager: EmbeddingManager with loaded embeddings
            alpha: Weight for vector similarity (0-1)
            beta: Weight for BM25 score (0-1)
        """
        self.embedding_manager = embedding_manager
        self.alpha = alpha
        self.beta = beta
        
        # Build BM25 index
        self.bm25 = None
        self.tokenized_texts: List[List[str]] = []
        self._build_bm25_index()
        
        logger.info(f"Initialized HybridRetriever with alpha={alpha}, beta={beta}")
    
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
        use_rrf: bool = True
    ) -> List[RetrievalResult]:
        """
        Retrieve relevant chunks using hybrid search.
        
        Args:
            query: Query string
            k: Number of results to return
            filter_document: Filter by document name
            filter_chunk_type: Filter by chunk type
            use_rrf: Whether to use Reciprocal Rank Fusion
            
        Returns:
            List of RetrievalResult objects
        """
        # Vector search
        vector_results = self.embedding_manager.search_by_text(
            query,
            k=k * 2,  # Get more for fusion
            filter_document=filter_document,
            filter_chunk_type=filter_chunk_type
        )
        
        # BM25 search
        bm25_results = self._bm25_search(
            query,
            k=k * 2,
            filter_document=filter_document,
            filter_chunk_type=filter_chunk_type
        )
        
        # Combine results
        if use_rrf:
            combined = self._reciprocal_rank_fusion(vector_results, bm25_results, k=k)
        else:
            combined = self._weighted_sum(vector_results, bm25_results, k=k)
        
        return combined
    
    def _bm25_search(
        self,
        query: str,
        k: int = 10,
        filter_document: str = None,
        filter_chunk_type: str = None
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
            
            results.append({
                'chunk': chunk,
                'score': float(score)
            })
        
        # Sort by score
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:k]
    
    def _reciprocal_rank_fusion(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 10,
        rrf_k: int = 60
    ) -> List[RetrievalResult]:
        """
        Combine results using Reciprocal Rank Fusion (RRF).
        
        RRF score = 1/(k + rank_vector) + 1/(k + rank_bm25)
        """
        # Create rank maps
        vector_ranks = {r['chunk'].chunk_id: i + 1 for i, r in enumerate(vector_results)}
        bm25_ranks = {r['chunk'].chunk_id: i + 1 for i, r in enumerate(bm25_results)}
        
        # Get all unique chunk IDs
        all_chunk_ids = set(vector_ranks.keys()) | set(bm25_ranks.keys())
        
        # Calculate RRF scores
        rrf_scores = {}
        for chunk_id in all_chunk_ids:
            vector_rank = vector_ranks.get(chunk_id, float('inf'))
            bm25_rank = bm25_ranks.get(chunk_id, float('inf'))
            
            rrf_score = 1.0 / (rrf_k + vector_rank) + 1.0 / (rrf_k + bm25_rank)
            rrf_scores[chunk_id] = rrf_score
        
        # Get top k
        sorted_chunks = sorted(
            rrf_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        # Build results
        results = []
        chunks_by_id = {r['chunk'].chunk_id: r['chunk'] for r in vector_results}
        chunks_by_id.update({r['chunk'].chunk_id: r['chunk'] for r in bm25_results})
        
        for chunk_id, score in sorted_chunks:
            chunk = chunks_by_id[chunk_id]
            
            # Get individual scores
            vector_score = 0.0
            bm25_score = 0.0
            
            for r in vector_results:
                if r['chunk'].chunk_id == chunk_id:
                    vector_score = r['score']
                    break
            
            for r in bm25_results:
                if r['chunk'].chunk_id == chunk_id:
                    bm25_score = r['score']
                    break
            
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                section_number=chunk.section_number,
                document_name=chunk.document_name,
                chunk_type=chunk.chunk_type,
                vector_score=vector_score,
                bm25_score=bm25_score,
                combined_score=score,
                metadata=chunk.metadata
            ))
        
        return results
    
    def _weighted_sum(
        self,
        vector_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 10
    ) -> List[RetrievalResult]:
        """Combine results using weighted sum of scores."""
        # Normalize scores
        vector_scores = self._normalize_scores(vector_results)
        bm25_scores = self._normalize_scores(bm25_results)
        
        # Get all chunk IDs
        all_chunk_ids = set(vector_scores.keys()) | set(bm25_scores.keys())
        
        # Calculate combined scores
        combined_scores = {}
        for chunk_id in all_chunk_ids:
            v_score = vector_scores.get(chunk_id, 0.0)
            b_score = bm25_scores.get(chunk_id, 0.0)
            combined_scores[chunk_id] = self.alpha * v_score + self.beta * b_score
        
        # Get top k
        sorted_chunks = sorted(
            combined_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        # Build results
        results = []
        chunks_by_id = {r['chunk'].chunk_id: r['chunk'] for r in vector_results}
        chunks_by_id.update({r['chunk'].chunk_id: r['chunk'] for r in bm25_results})
        
        for chunk_id, score in sorted_chunks:
            chunk = chunks_by_id[chunk_id]
            
            results.append(RetrievalResult(
                chunk_id=chunk.chunk_id,
                text=chunk.text,
                section_number=chunk.section_number,
                document_name=chunk.document_name,
                chunk_type=chunk.chunk_type,
                vector_score=vector_scores.get(chunk_id, 0.0),
                bm25_score=bm25_scores.get(chunk_id, 0.0),
                combined_score=score,
                metadata=chunk.metadata
            ))
        
        return results
    
    def _normalize_scores(self, results: List[Dict]) -> Dict[str, float]:
        """Normalize scores to [0, 1] range."""
        if not results:
            return {}
        
        scores = [r['score'] for r in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score - min_score < 1e-10:
            # All scores are the same
            return {r['chunk'].chunk_id: 1.0 for r in results}
        
        return {
            r['chunk'].chunk_id: (r['score'] - min_score) / (max_score - min_score)
            for r in results
        }
    
    def rerank(
        self,
        results: List[RetrievalResult],
        query: str,
        k: int = 10
    ) -> List[RetrievalResult]:
        """
        Rerank results using a cross-encoder or re-ranking score.
        For now, uses simple query term overlap reranking.
        
        Args:
            results: Initial retrieval results
            query: Original query
            k: Number of results to return
            
        Returns:
            Reranked results
        """
        query_terms = set(query.lower().split())
        
        # Calculate rerank score based on query term overlap
        scored_results = []
        for result in results:
            text_terms = set(result.text.lower().split())
            overlap = len(query_terms & text_terms)
            overlap_score = overlap / max(len(query_terms), 1)
            
            # Boost score based on overlap
            new_score = result.combined_score * (1 + 0.5 * overlap_score)
            scored_results.append((result, new_score))
        
        # Sort and return top k
        scored_results.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in scored_results[:k]]


def main():
    """Example usage."""
    import sys
    
    # This would typically be used after building embeddings
    # For demo purposes, shows the interface
    
    print("HybridRetriever initialized with embedding manager")
    print("Usage:")
    print("  retriever = HybridRetriever(embedding_manager)")
    print("  results = retriever.retrieve('What are ordinary hours?', k=5)")
    print("  for r in results:")
    print("      print(f'{r.section_number}: {r.text[:100]}...')")


if __name__ == "__main__":
    main()
