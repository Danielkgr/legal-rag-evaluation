"""Vector search and rank fusion on fixed vectors, with no embedding server."""

import math

from legal_rag.embedding import EmbeddingManager
from legal_rag.retrieval import HybridRetriever

VECTORS = {
    "query": [1.0, 0.0, 0.0],
    "same direction": [2.0, 0.0, 0.0],
    "diagonal": [1.0, 1.0, 0.0],
    "orthogonal": [0.0, 1.0, 0.0],
    "zero": [0.0, 0.0, 0.0],
}


class FixedVectors:
    def embed_text(self, text):
        return VECTORS[text]

    def embed_texts(self, texts):
        return [VECTORS[t] for t in texts]


def _manager(tmp_path):
    manager = EmbeddingManager(FixedVectors(), storage_path=str(tmp_path))
    manager.add_chunks_batch(
        [
            {
                "chunk_id": name,
                "text": name,
                "document_name": doc,
                "chunk_type": "section",
            }
            for name, doc in [
                ("orthogonal", "a"),
                ("zero", "a"),
                ("diagonal", "b"),
                ("same direction", "a"),
            ]
        ]
    )
    return manager


def test_cosine_search_ranks_by_angle_and_scores_a_zero_vector_as_zero(tmp_path):
    hits = _manager(tmp_path).search_by_text("query", k=4)
    assert [h["chunk"].chunk_id for h in hits] == [
        "same direction",
        "diagonal",
        "orthogonal",
        "zero",
    ]
    assert math.isclose(hits[0]["score"], 1.0)
    assert math.isclose(hits[1]["score"], 1 / math.sqrt(2))
    assert hits[3]["score"] == 0.0


def test_cosine_search_applies_filters_before_taking_k(tmp_path):
    hits = _manager(tmp_path).search_by_text("query", k=1, filter_document="b")
    assert [h["chunk"].chunk_id for h in hits] == ["diagonal"]


def test_fusion_prefers_a_chunk_both_searches_rank_highly(tmp_path):
    retriever = HybridRetriever(_manager(tmp_path))
    chunks = {r.chunk_id: r for r in retriever.embedding_manager.embeddings}
    vector = [{"chunk": chunks[c], "score": 1.0} for c in ["diagonal", "zero"]]
    bm25 = [{"chunk": chunks[c], "score": 1.0} for c in ["orthogonal", "diagonal"]]
    fused = retriever._reciprocal_rank_fusion(vector, bm25, k=3)
    # orthogonal is first for BM25 (1/61), zero second for vectors (1/62).
    assert [r.chunk_id for r in fused] == ["diagonal", "orthogonal", "zero"]
    assert fused[0].combined_score == 1 / 61 + 1 / 62
