"""Metric semantics, including the convention that makes auto-annotation misleading.

The bundled evaluation generator marks a chunk relevant to a query using only
shallow lexical overlap, so in practice it labels *every* chunk relevant to
*every* query. Under this implementation that inflates recall and mean average
precision to 1.0 for any query with no judged-relevant chunk, which is why the
auto-generated run cannot be read as a retrieval score. These tests pin down
both the intended behaviour on a small hand-built judgment set and that
no-relevant-chunks convention, so the behaviour is documented rather than
surprising.
"""

from evaluation.eval_metrics import EvaluationMetrics


def _ann(query_id, chunk_id, score):
    return {"query_id": query_id, "chunk_id": chunk_id, "relevance_score": score}


def test_retrieval_of_the_relevant_chunk_scores_perfectly():
    annotations = [
        _ann("q1", "A", 1),
        _ann("q1", "B", 0),
    ]
    metrics = EvaluationMetrics(annotations)
    overall = metrics.compute_overall_metrics({"q1": ["A", "B"]}, k_values=[1, 2])
    assert overall["precision@1"] == 1.0
    assert overall["recall@1"] == 1.0
    assert overall["mrr"] == 1.0


def test_irrelevant_top_result_lowers_precision_but_not_mrr():
    annotations = [_ann("q1", "A", 1), _ann("q1", "B", 0)]
    metrics = EvaluationMetrics(annotations)
    overall = metrics.compute_overall_metrics({"q1": ["B", "A"]}, k_values=[2])
    # One of two retrieved is relevant, but the first hit is at rank 2.
    assert overall["precision@2"] == 0.5
    assert overall["mrr"] == 0.5


def test_query_with_no_relevant_chunks_is_recalled_as_perfect_by_convention():
    # Documents the convention behind the inflated auto-annotated scores: with
    # nothing judged relevant, recall and MAP default to 1.0 while precision is
    # 0.0, so an aggregate over such queries is uninterpretable.
    annotations = [_ann("q1", "A", 0)]
    metrics = EvaluationMetrics(annotations)
    overall = metrics.compute_overall_metrics({"q1": ["A", "B"]}, k_values=[1])
    assert overall["recall@1"] == 1.0
    assert overall["map"] == 1.0
    assert overall["precision@1"] == 0.0
