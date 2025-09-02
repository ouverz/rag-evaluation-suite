"""
Core evaluation module for RAG system performance metrics.
Provides Information Retrieval metrics for search quality assessment.
"""

from .metrics import (
    EvaluationMetric,
    MeanReciprocalRank,
    PrecisionAtK,
    MeanAveragePrecision,
    EvaluationResult,
    evaluate_search_results
)

__all__ = [
    "EvaluationMetric",
    "MeanReciprocalRank", 
    "PrecisionAtK",
    "MeanAveragePrecision",
    "EvaluationResult",
    "evaluate_search_results"
]