"""
Evaluation metrics module for RAG systems.
Computes precision, recall, and other standard metrics.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class RetrievalMetrics:
    """Container for retrieval evaluation metrics."""

    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    f1_at_k: Dict[int, float]
    mean_reciprocal_rank: float
    average_precision: float
    num_relevant: int
    num_retrieved: int


class EvaluationMetrics:
    """
    Computes evaluation metrics for RAG systems.
    Focuses on precision and recall as requested.
    """

    def __init__(self, annotations: List[Dict]):
        """
        Initialize with relevance annotations.

        Args:
            annotations: List of relevance annotation dicts
        """
        self.annotations = annotations

        # Build lookup: query_id -> {chunk_id: relevance_score}
        self.relevance_judgments = self._build_relevance_judgments()

        logger.info(
            f"Initialized EvaluationMetrics with {len(annotations)} annotations"
        )

    def _build_relevance_judgments(self) -> Dict[str, Dict[str, int]]:
        """Build query -> chunk -> relevance lookup."""
        judgments = defaultdict(dict)

        for ann in self.annotations:
            query_id = ann["query_id"]
            chunk_id = ann["chunk_id"]
            relevance = ann["relevance_score"]
            judgments[query_id][chunk_id] = relevance

        return judgments

    def compute_precision_at_k(
        self,
        query_id: str,
        retrieved_chunks: List[str],
        k: int = 10,
        relevant_threshold: int = 1,
    ) -> float:
        """
        Compute Precision@K for a query.

        Args:
            query_id: Query identifier
            retrieved_chunks: List of chunk IDs in retrieval order
            k: Number of results to consider
            relevant_threshold: Minimum relevance score to be considered relevant

        Returns:
            Precision@K score
        """
        retrieved_k = retrieved_chunks[:k]
        relevant_judgments = self.relevance_judgments.get(query_id, {})

        if not retrieved_k:
            return 0.0

        relevant_count = sum(
            1
            for chunk_id in retrieved_k
            if relevant_judgments.get(chunk_id, 0) >= relevant_threshold
        )

        return relevant_count / len(retrieved_k)

    def compute_recall_at_k(
        self,
        query_id: str,
        retrieved_chunks: List[str],
        k: int = 10,
        relevant_threshold: int = 1,
    ) -> float:
        """
        Compute Recall@K for a query.

        Args:
            query_id: Query identifier
            retrieved_chunks: List of chunk IDs in retrieval order
            k: Number of results to consider
            relevant_threshold: Minimum relevance score to be considered relevant

        Returns:
            Recall@K score
        """
        relevant_judgments = self.relevance_judgments.get(query_id, {})

        # Count total relevant documents
        total_relevant = sum(
            1 for score in relevant_judgments.values() if score >= relevant_threshold
        )

        if total_relevant == 0:
            return 1.0  # No relevant docs = perfect recall (nothing to retrieve)

        retrieved_k = retrieved_chunks[:k]
        relevant_retrieved = sum(
            1
            for chunk_id in retrieved_k
            if relevant_judgments.get(chunk_id, 0) >= relevant_threshold
        )

        return relevant_retrieved / total_relevant

    def compute_f1_at_k(
        self,
        query_id: str,
        retrieved_chunks: List[str],
        k: int = 10,
        relevant_threshold: int = 1,
    ) -> float:
        """
        Compute F1@K for a query.

        Args:
            query_id: Query identifier
            retrieved_chunks: List of chunk IDs in retrieval order
            k: Number of results to consider
            relevant_threshold: Minimum relevance score to be considered relevant

        Returns:
            F1@K score
        """
        precision = self.compute_precision_at_k(
            query_id, retrieved_chunks, k, relevant_threshold
        )
        recall = self.compute_recall_at_k(
            query_id, retrieved_chunks, k, relevant_threshold
        )

        if precision + recall < 1e-10:
            return 0.0

        return 2 * precision * recall / (precision + recall)

    def compute_reciprocal_rank(
        self, query_id: str, retrieved_chunks: List[str], relevant_threshold: int = 1
    ) -> float:
        """
        Compute Reciprocal Rank for a query.

        Args:
            query_id: Query identifier
            retrieved_chunks: List of chunk IDs in retrieval order
            relevant_threshold: Minimum relevance score to be considered relevant

        Returns:
            Reciprocal rank (1/position of first relevant result)
        """
        relevant_judgments = self.relevance_judgments.get(query_id, {})

        for i, chunk_id in enumerate(retrieved_chunks):
            if relevant_judgments.get(chunk_id, 0) >= relevant_threshold:
                return 1.0 / (i + 1)

        return 0.0

    def compute_average_precision(
        self, query_id: str, retrieved_chunks: List[str], relevant_threshold: int = 1
    ) -> float:
        """
        Compute Average Precision for a query.

        Args:
            query_id: Query identifier
            retrieved_chunks: List of chunk IDs in retrieval order
            relevant_threshold: Minimum relevance score to be considered relevant

        Returns:
            Average precision score
        """
        relevant_judgments = self.relevance_judgments.get(query_id, {})

        if not retrieved_chunks:
            return 0.0

        ap_sum = 0.0
        relevant_count = 0

        for i, chunk_id in enumerate(retrieved_chunks):
            if relevant_judgments.get(chunk_id, 0) >= relevant_threshold:
                relevant_count += 1
                precision_at_i = relevant_count / (i + 1)
                ap_sum += precision_at_i

        # Count total relevant docs
        total_relevant = sum(
            1 for score in relevant_judgments.values() if score >= relevant_threshold
        )

        if total_relevant == 0:
            return 1.0

        return ap_sum / total_relevant

    def compute_metrics_for_query(
        self,
        query_id: str,
        retrieved_chunks: List[str],
        k_values: Optional[List[int]] = None,
    ) -> RetrievalMetrics:
        """
        Compute all metrics for a single query.

        Args:
            query_id: Query identifier
            retrieved_chunks: List of chunk IDs in retrieval order
            k_values: K values to compute metrics at

        Returns:
            RetrievalMetrics object
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        relevant_judgments = self.relevance_judgments.get(query_id, {})
        total_relevant = sum(1 for score in relevant_judgments.values() if score >= 1)

        precision_at_k = {}
        recall_at_k = {}
        f1_at_k = {}

        for k in k_values:
            precision_at_k[k] = self.compute_precision_at_k(
                query_id, retrieved_chunks, k, relevant_threshold=1
            )
            recall_at_k[k] = self.compute_recall_at_k(
                query_id, retrieved_chunks, k, relevant_threshold=1
            )
            f1_at_k[k] = self.compute_f1_at_k(
                query_id, retrieved_chunks, k, relevant_threshold=1
            )

        rr = self.compute_reciprocal_rank(
            query_id, retrieved_chunks, relevant_threshold=1
        )
        ap = self.compute_average_precision(
            query_id, retrieved_chunks, relevant_threshold=1
        )

        return RetrievalMetrics(
            precision_at_k=precision_at_k,
            recall_at_k=recall_at_k,
            f1_at_k=f1_at_k,
            mean_reciprocal_rank=rr,
            average_precision=ap,
            num_relevant=total_relevant,
            num_retrieved=len(retrieved_chunks),
        )

    def compute_overall_metrics(
        self,
        query_results: Dict[str, List[str]],
        k_values: Optional[List[int]] = None,
        aggregate: str = "mean",
    ) -> Dict[str, float]:
        """
        Compute overall metrics across all queries.

        Args:
            query_results: Dict of query_id -> list of retrieved chunk IDs
            k_values: K values to compute metrics at
            aggregate: Aggregation method ('mean', 'median')

        Returns:
            Dict of overall metrics
        """
        if k_values is None:
            k_values = [1, 3, 5, 10]

        all_metrics = []

        for query_id, retrieved in query_results.items():
            metrics = self.compute_metrics_for_query(query_id, retrieved, k_values)
            all_metrics.append(metrics)

        if not all_metrics:
            return {}

        # Aggregate results
        overall = {}

        for k in k_values:
            precisions = [m.precision_at_k[k] for m in all_metrics]
            recalls = [m.recall_at_k[k] for m in all_metrics]
            f1s = [m.f1_at_k[k] for m in all_metrics]

            if aggregate == "mean":
                overall[f"precision@{k}"] = sum(precisions) / len(precisions)
                overall[f"recall@{k}"] = sum(recalls) / len(recalls)
                overall[f"f1@{k}"] = sum(f1s) / len(f1s)
            elif aggregate == "median":
                sorted_precisions = sorted(precisions)
                sorted_recalls = sorted(recalls)
                sorted_f1s = sorted(f1s)
                mid = len(sorted_precisions) // 2
                overall[f"precision@{k}"] = sorted_precisions[mid]
                overall[f"recall@{k}"] = sorted_recalls[mid]
                overall[f"f1@{k}"] = sorted_f1s[mid]

        # MRR and MAP
        mrrs = [m.mean_reciprocal_rank for m in all_metrics]
        maps = [m.average_precision for m in all_metrics]

        overall["mrr"] = sum(mrrs) / len(mrrs)
        overall["map"] = sum(maps) / len(maps)

        # Overall statistics
        total_relevant = sum(m.num_relevant for m in all_metrics)
        total_retrieved = sum(m.num_retrieved for m in all_metrics)

        overall["total_relevant"] = total_relevant
        overall["total_retrieved"] = total_retrieved
        overall["overall_precision"] = sum(
            m.precision_at_k.get(10, 0) for m in all_metrics
        ) / len(all_metrics)
        overall["overall_recall"] = sum(
            m.recall_at_k.get(10, 0) for m in all_metrics
        ) / len(all_metrics)

        return overall
