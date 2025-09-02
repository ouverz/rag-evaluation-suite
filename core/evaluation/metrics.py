"""
Information Retrieval metrics for RAG system evaluation.

This module implements standard IR metrics including Mean Reciprocal Rank (MRR),
Precision@K, and Mean Average Precision (MAP) for evaluating search quality.
Designed to work with the existing hybrid search engine and document rankings.
"""

from __future__ import annotations
import logging
import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Set, Union
from langchain.schema import Document

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Standardized evaluation result format."""
    metric_name: str
    score: float
    query_scores: Dict[str, float]
    metadata: Dict[str, Any]


class EvaluationMetric(ABC):
    """Abstract base class for evaluation metrics."""
    
    @abstractmethod
    def calculate(
        self, 
        search_results: Dict[str, List[str]], 
        ground_truth: Dict[str, Set[str]]
    ) -> EvaluationResult:
        """
        Calculate metric score for search results against ground truth.
        
        Args:
            search_results: Dict mapping query_id to ranked list of doc_ids
            ground_truth: Dict mapping query_id to set of relevant doc_ids
            
        Returns:
            EvaluationResult with score and per-query breakdown
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return metric name for identification."""
        pass


class MeanReciprocalRank(EvaluationMetric):
    """
    Mean Reciprocal Rank (MRR) metric implementation.
    
    MRR = (1/|Q|) * Σ(1/rank_i) where rank_i is position of first relevant doc
    Measures how well the search system ranks the first relevant document.
    """
    
    def __init__(self):
        """Initialize MRR metric calculator."""
        self.metric_name = "Mean Reciprocal Rank (MRR)"
    
    def get_name(self) -> str:
        """Return metric name."""
        return self.metric_name
    
    def _calculate_reciprocal_rank(
        self, 
        ranked_docs: List[str], 
        relevant_docs: Set[str]
    ) -> float:
        """
        Calculate reciprocal rank for a single query.
        
        Args:
            ranked_docs: Ranked list of document IDs
            relevant_docs: Set of relevant document IDs
            
        Returns:
            Reciprocal rank (1/position) of first relevant document, 0 if none
        """
        if not ranked_docs or not relevant_docs:
            return 0.0
        
        for i, doc_id in enumerate(ranked_docs, 1):
            if doc_id in relevant_docs:
                return 1.0 / i
        
        return 0.0
    
    def calculate(
        self, 
        search_results: Dict[str, List[str]], 
        ground_truth: Dict[str, Set[str]]
    ) -> EvaluationResult:
        """
        Calculate MRR across all queries.
        
        Args:
            search_results: Dict mapping query_id to ranked doc_ids
            ground_truth: Dict mapping query_id to relevant doc_ids
            
        Returns:
            EvaluationResult with MRR score and per-query breakdown
        """
        if not search_results:
            logger.warning("Empty search results provided to MRR calculation")
            return EvaluationResult(
                metric_name=self.metric_name,
                score=0.0,
                query_scores={},
                metadata={"total_queries": 0, "queries_with_results": 0}
            )
        
        query_scores = {}
        reciprocal_ranks = []
        
        for query_id in search_results:
            ranked_docs = search_results.get(query_id, [])
            relevant_docs = ground_truth.get(query_id, set())
            
            if not relevant_docs:
                logger.warning(f"No ground truth for query {query_id}")
                continue
            
            rr = self._calculate_reciprocal_rank(ranked_docs, relevant_docs)
            query_scores[query_id] = rr
            reciprocal_ranks.append(rr)
        
        # Calculate MRR
        mrr_score = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0
        
        metadata = {
            "total_queries": len(search_results),
            "queries_with_ground_truth": len(reciprocal_ranks),
            "queries_with_results": len([q for q in search_results.values() if q]),
            "zero_reciprocal_ranks": reciprocal_ranks.count(0.0)
        }
        
        logger.info(f"MRR calculation completed: {mrr_score:.4f} "
                   f"({len(reciprocal_ranks)} queries)")
        
        return EvaluationResult(
            metric_name=self.metric_name,
            score=mrr_score,
            query_scores=query_scores,
            metadata=metadata
        )


class PrecisionAtK(EvaluationMetric):
    """
    Precision@K metric implementation.
    
    P@K = (relevant documents in top K) / K
    Measures fraction of top K results that are relevant.
    """
    
    def __init__(self, k: int = 10):
        """
        Initialize Precision@K metric.
        
        Args:
            k: Number of top results to consider
        """
        if k <= 0:
            raise ValueError("k parameter must be positive")
        
        self.k = k
        self.metric_name = f"Precision@{k}"
    
    def get_name(self) -> str:
        """Return metric name."""
        return self.metric_name
    
    def _calculate_precision_at_k(
        self, 
        ranked_docs: List[str], 
        relevant_docs: Set[str],
        k: int
    ) -> float:
        """
        Calculate precision@k for a single query.
        
        Args:
            ranked_docs: Ranked list of document IDs
            relevant_docs: Set of relevant document IDs
            k: Number of top results to consider
            
        Returns:
            Precision@K score (0.0 to 1.0)
        """
        if not ranked_docs or not relevant_docs:
            return 0.0
        
        top_k_docs = ranked_docs[:k]
        relevant_in_top_k = sum(1 for doc_id in top_k_docs 
                              if doc_id in relevant_docs)
        
        return relevant_in_top_k / len(top_k_docs) if top_k_docs else 0.0
    
    def calculate(
        self, 
        search_results: Dict[str, List[str]], 
        ground_truth: Dict[str, Set[str]]
    ) -> EvaluationResult:
        """
        Calculate Precision@K across all queries.
        
        Args:
            search_results: Dict mapping query_id to ranked doc_ids
            ground_truth: Dict mapping query_id to relevant doc_ids
            
        Returns:
            EvaluationResult with Precision@K score and per-query breakdown
        """
        if not search_results:
            logger.warning("Empty search results provided to Precision@K calculation")
            return EvaluationResult(
                metric_name=self.metric_name,
                score=0.0,
                query_scores={},
                metadata={"total_queries": 0, "k_value": self.k}
            )
        
        query_scores = {}
        precision_scores = []
        
        for query_id in search_results:
            ranked_docs = search_results.get(query_id, [])
            relevant_docs = ground_truth.get(query_id, set())
            
            if not relevant_docs:
                logger.warning(f"No ground truth for query {query_id}")
                continue
            
            precision = self._calculate_precision_at_k(
                ranked_docs, relevant_docs, self.k
            )
            query_scores[query_id] = precision
            precision_scores.append(precision)
        
        # Calculate mean Precision@K
        mean_precision = (sum(precision_scores) / len(precision_scores) 
                         if precision_scores else 0.0)
        
        metadata = {
            "total_queries": len(search_results),
            "queries_with_ground_truth": len(precision_scores),
            "k_value": self.k,
            "perfect_precision_queries": precision_scores.count(1.0),
            "zero_precision_queries": precision_scores.count(0.0)
        }
        
        logger.info(f"Precision@{self.k} calculation completed: "
                   f"{mean_precision:.4f} ({len(precision_scores)} queries)")
        
        return EvaluationResult(
            metric_name=self.metric_name,
            score=mean_precision,
            query_scores=query_scores,
            metadata=metadata
        )


class MeanAveragePrecision(EvaluationMetric):
    """
    Mean Average Precision (MAP) metric implementation.
    
    MAP = (1/|Q|) * Σ(AP_q) where AP is Average Precision per query
    AP = (1/R) * Σ(P(k) * rel(k)) where P(k) is precision at rank k
    """
    
    def __init__(self):
        """Initialize MAP metric calculator."""
        self.metric_name = "Mean Average Precision (MAP)"
    
    def get_name(self) -> str:
        """Return metric name."""
        return self.metric_name
    
    def _calculate_average_precision(
        self, 
        ranked_docs: List[str], 
        relevant_docs: Set[str]
    ) -> float:
        """
        Calculate Average Precision for a single query.
        
        Args:
            ranked_docs: Ranked list of document IDs
            relevant_docs: Set of relevant document IDs
            
        Returns:
            Average Precision score (0.0 to 1.0)
        """
        if not ranked_docs or not relevant_docs:
            return 0.0
        
        relevant_found = 0
        precision_sum = 0.0
        
        for i, doc_id in enumerate(ranked_docs, 1):
            if doc_id in relevant_docs:
                relevant_found += 1
                precision_at_i = relevant_found / i
                precision_sum += precision_at_i
        
        return precision_sum / len(relevant_docs) if relevant_docs else 0.0
    
    def calculate(
        self, 
        search_results: Dict[str, List[str]], 
        ground_truth: Dict[str, Set[str]]
    ) -> EvaluationResult:
        """
        Calculate MAP across all queries.
        
        Args:
            search_results: Dict mapping query_id to ranked doc_ids
            ground_truth: Dict mapping query_id to relevant doc_ids
            
        Returns:
            EvaluationResult with MAP score and per-query breakdown
        """
        if not search_results:
            logger.warning("Empty search results provided to MAP calculation")
            return EvaluationResult(
                metric_name=self.metric_name,
                score=0.0,
                query_scores={},
                metadata={"total_queries": 0}
            )
        
        query_scores = {}
        ap_scores = []
        
        for query_id in search_results:
            ranked_docs = search_results.get(query_id, [])
            relevant_docs = ground_truth.get(query_id, set())
            
            if not relevant_docs:
                logger.warning(f"No ground truth for query {query_id}")
                continue
            
            ap = self._calculate_average_precision(ranked_docs, relevant_docs)
            query_scores[query_id] = ap
            ap_scores.append(ap)
        
        # Calculate MAP
        map_score = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
        
        metadata = {
            "total_queries": len(search_results),
            "queries_with_ground_truth": len(ap_scores),
            "perfect_ap_queries": ap_scores.count(1.0),
            "zero_ap_queries": ap_scores.count(0.0),
            "avg_relevant_docs_per_query": (
                sum(len(ground_truth.get(q, set())) 
                    for q in search_results) / len(search_results)
                if search_results else 0.0
            )
        }
        
        logger.info(f"MAP calculation completed: {map_score:.4f} "
                   f"({len(ap_scores)} queries)")
        
        return EvaluationResult(
            metric_name=self.metric_name,
            score=map_score,
            query_scores=query_scores,
            metadata=metadata
        )


def extract_document_ids_from_search_results(
    search_results: Union[List[Document], pd.DataFrame]
) -> List[str]:
    """
    Extract document IDs from search engine results.
    
    Args:
        search_results: Either list of Document objects or DataFrame
        
    Returns:
        List of document IDs in ranked order
        
    Raises:
        ValueError: If results format is not supported
    """
    if isinstance(search_results, list):
        # Handle List[Document] from search engines
        doc_ids = []
        for doc in search_results:
            if isinstance(doc, Document):
                doc_id = doc.metadata.get("id")
                if doc_id is not None:
                    doc_ids.append(str(doc_id))
            else:
                logger.warning(f"Non-Document object in results: {type(doc)}")
        return doc_ids
    
    elif isinstance(search_results, pd.DataFrame):
        # Handle DataFrame from search engines
        if "id" in search_results.columns:
            return search_results["id"].astype(str).tolist()
        else:
            raise ValueError("DataFrame missing 'id' column")
    
    else:
        raise ValueError(f"Unsupported search results format: {type(search_results)}")


def evaluate_search_results(
    queries: Dict[str, str],
    search_engine_results: Dict[str, Union[List[Document], pd.DataFrame]],
    ground_truth: Dict[str, Set[str]],
    metrics: Optional[List[EvaluationMetric]] = None
) -> Dict[str, EvaluationResult]:
    """
    Evaluate search results using multiple IR metrics.
    
    Args:
        queries: Dict mapping query_id to query string
        search_engine_results: Dict mapping query_id to search results
        ground_truth: Dict mapping query_id to set of relevant doc_ids
        metrics: List of metrics to calculate (defaults to MRR, P@10, MAP)
        
    Returns:
        Dict mapping metric name to EvaluationResult
        
    Example:
        >>> queries = {"q1": "sleep patterns children"}
        >>> results = {"q1": [doc1, doc2, doc3]}  # Documents from search
        >>> truth = {"q1": {"doc1", "doc3"}}  # Relevant docs
        >>> evaluation = evaluate_search_results(queries, results, truth)
        >>> print(f"MRR: {evaluation['Mean Reciprocal Rank (MRR)'].score}")
    """
    if metrics is None:
        metrics = [
            MeanReciprocalRank(),
            PrecisionAtK(k=10),
            MeanAveragePrecision()
        ]
    
    # Convert search results to standardized format
    standardized_results = {}
    for query_id, results in search_engine_results.items():
        try:
            doc_ids = extract_document_ids_from_search_results(results)
            standardized_results[query_id] = doc_ids
        except Exception as e:
            logger.error(f"Failed to extract doc IDs for query {query_id}: {e}")
            standardized_results[query_id] = []
    
    # Calculate metrics
    metric_results = {}
    for metric in metrics:
        try:
            result = metric.calculate(standardized_results, ground_truth)
            metric_results[metric.get_name()] = result
            logger.info(f"Calculated {metric.get_name()}: {result.score:.4f}")
        except Exception as e:
            logger.error(f"Failed to calculate {metric.get_name()}: {e}")
            metric_results[metric.get_name()] = EvaluationResult(
                metric_name=metric.get_name(),
                score=0.0,
                query_scores={},
                metadata={"error": str(e)}
            )
    
    return metric_results