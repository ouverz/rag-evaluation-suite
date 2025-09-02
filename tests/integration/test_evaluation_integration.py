"""
Integration tests for evaluation metrics with RAG search engines.

Tests the evaluation metrics with actual search engine outputs to ensure
compatibility with the existing hybrid search system.
"""

import pytest
import pandas as pd
from typing import Dict, Set
from langchain.schema import Document

from core.evaluation.metrics import (
    MeanReciprocalRank,
    PrecisionAtK,
    MeanAveragePrecision,
    evaluate_search_results
)
from core.interfaces.search_engines import SearchResult


class TestEvaluationIntegration:
    """Integration tests for evaluation metrics with search engines."""
    
    def test_metrics_with_search_result_format(self):
        """Test metrics work with SearchResult format from search engines."""
        # Simulate SearchResult from hybrid search engine
        mock_search_result = SearchResult(
            query="sleep patterns in children",
            documents=[
                Document(
                    page_content="Research on sleep patterns shows...",
                    metadata={"id": "doc1", "source": "paper1.pdf", "rrf_score": 0.8}
                ),
                Document(
                    page_content="Children's bedtime routines affect...",
                    metadata={"id": "doc2", "source": "paper2.pdf", "rrf_score": 0.6}
                ),
                Document(
                    page_content="Sleep duration correlates with...",
                    metadata={"id": "doc3", "source": "paper3.pdf", "rrf_score": 0.4}
                )
            ],
            total_results=3,
            processing_time_ms=150,
            metadata={"search_method": "hybrid_rrf"}
        )
        
        # Convert to format expected by evaluation
        search_results = {"q1": mock_search_result.documents}
        ground_truth = {"q1": {"doc1", "doc3"}}  # doc1 and doc3 are relevant
        
        # Use the main evaluation function which handles Document objects correctly
        results = evaluate_search_results(
            queries={"q1": "sleep patterns in children"},
            search_engine_results=search_results,
            ground_truth=ground_truth
        )
        
        # Test MRR result
        mrr_result = results["Mean Reciprocal Rank (MRR)"]
        assert mrr_result.score == 1.0  # First doc is relevant (rank 1)
        assert mrr_result.query_scores["q1"] == 1.0
        
        # Test Precision@10 result
        precision_result = results["Precision@10"]
        assert precision_result.score == 2.0/3.0  # 2 relevant out of 3
        assert precision_result.query_scores["q1"] == 2.0/3.0
        
        # Test MAP result
        map_result = results["Mean Average Precision (MAP)"]
        # AP = (1/1 + 2/3) / 2 = (1.0 + 0.667) / 2 = 0.833
        expected_ap = (1.0 + 2.0/3.0) / 2.0
        assert abs(map_result.score - expected_ap) < 1e-10
    
    def test_metrics_with_dataframe_from_hybrid_search(self):
        """Test metrics with DataFrame output from hybrid search."""
        # Simulate DataFrame returned by hybrid search engine
        search_df = pd.DataFrame({
            "id": ["doc1", "doc2", "doc3", "doc4"],
            "content": [
                "Sleep research shows important patterns...",
                "Bedtime routines for young children...",
                "Cognitive development and sleep...",
                "Nutrition affects sleep quality..."
            ],
            "rrf_score": [0.85, 0.72, 0.68, 0.45],
            "source_engine": ["rrf_hybrid", "rrf_hybrid", "rrf_hybrid", "rrf_hybrid"],
            "bm25_rank": [1, 2, None, 4],
            "vector_rank": [2, 3, 1, None]
        })
        
        search_results = {"q1": search_df}
        ground_truth = {"q1": {"doc1", "doc2", "doc4"}}
        
        # Use the main evaluation function
        results = evaluate_search_results(
            queries={"q1": "sleep patterns children"},
            search_engine_results=search_results,
            ground_truth=ground_truth
        )
        
        # Verify all metrics calculated successfully
        assert "Mean Reciprocal Rank (MRR)" in results
        assert "Precision@10" in results
        assert "Mean Average Precision (MAP)" in results
        
        # Check MRR (first doc is relevant)
        mrr_result = results["Mean Reciprocal Rank (MRR)"]
        assert mrr_result.score == 1.0
        
        # Check Precision@10 (3 relevant out of 4 total)
        precision_result = results["Precision@10"]
        assert precision_result.score == 3.0/4.0
        
        # Check MAP calculation
        map_result = results["Mean Average Precision (MAP)"]
        # AP = (1/1 + 2/2 + 3/4) / 3 = (1.0 + 1.0 + 0.75) / 3 = 0.917
        expected_ap = (1.0 + 1.0 + 0.75) / 3.0
        assert abs(map_result.score - expected_ap) < 1e-10
    
    def test_metrics_with_multiple_queries_realistic_scenario(self):
        """Test evaluation across multiple queries in realistic scenario."""
        # Simulate multiple query evaluation scenario
        queries = {
            "sleep_patterns": "How do sleep patterns affect child development?",
            "bedtime_routines": "What are effective bedtime routines for toddlers?",
            "sleep_duration": "How much sleep do children need by age?"
        }
        
        # Mock search results for each query
        search_results = {
            "sleep_patterns": [
                Document(page_content="Sleep patterns research...", metadata={"id": "sleep1"}),
                Document(page_content="Child development study...", metadata={"id": "dev1"}),
                Document(page_content="Cognitive effects of sleep...", metadata={"id": "cog1"})
            ],
            "bedtime_routines": [
                Document(page_content="Toddler bedtime strategies...", metadata={"id": "routine1"}),
                Document(page_content="Bedtime consistency study...", metadata={"id": "routine2"}),
                Document(page_content="Parent behavior patterns...", metadata={"id": "parent1"})
            ],
            "sleep_duration": [
                Document(page_content="Age-based sleep requirements...", metadata={"id": "duration1"}),
                Document(page_content="Sleep duration guidelines...", metadata={"id": "duration2"})
            ]
        }
        
        # Ground truth relevance judgments
        ground_truth = {
            "sleep_patterns": {"sleep1", "cog1"},  # 2 relevant docs
            "bedtime_routines": {"routine1", "routine2"},  # 2 relevant docs
            "sleep_duration": {"duration1", "duration2"}  # 2 relevant docs
        }
        
        # Evaluate with custom metrics
        custom_metrics = [
            MeanReciprocalRank(),
            PrecisionAtK(k=5),
            MeanAveragePrecision()
        ]
        
        results = evaluate_search_results(
            queries, search_results, ground_truth, metrics=custom_metrics
        )
        
        # Verify results structure
        assert len(results) == 3
        assert "Mean Reciprocal Rank (MRR)" in results
        assert "Precision@5" in results
        assert "Mean Average Precision (MAP)" in results
        
        # Check individual query results are tracked
        for metric_name, result in results.items():
            assert len(result.query_scores) == 3  # One score per query
            assert "sleep_patterns" in result.query_scores
            assert "bedtime_routines" in result.query_scores
            assert "sleep_duration" in result.query_scores
            assert result.metadata["queries_with_ground_truth"] == 3
        
        # MRR should be 1.0 (all queries have relevant doc at rank 1)
        assert results["Mean Reciprocal Rank (MRR)"].score == 1.0
        
        # MAP should also be high since relevant docs are at good positions
        assert results["Mean Average Precision (MAP)"].score > 0.8
    
    def test_evaluation_with_rrf_metadata(self):
        """Test evaluation preserves and works with RRF metadata from hybrid search."""
        # Documents with RRF metadata from hybrid search engine
        rrf_documents = [
            Document(
                page_content="Sleep research content...",
                metadata={
                    "id": "doc1",
                    "rrf_score": 0.95,
                    "bm25_rank": 1,
                    "vector_rank": 2,
                    "found_by_engines": ["bm25", "vector"],
                    "source_engine": "rrf_hybrid",
                    "fusion_method": "reciprocal_rank_fusion"
                }
            ),
            Document(
                page_content="Additional sleep content...",
                metadata={
                    "id": "doc2",
                    "rrf_score": 0.78,
                    "bm25_rank": 3,
                    "vector_rank": 1,
                    "found_by_engines": ["bm25", "vector"],
                    "source_engine": "rrf_hybrid"
                }
            ),
            Document(
                page_content="Less relevant content...",
                metadata={
                    "id": "doc3",
                    "rrf_score": 0.45,
                    "bm25_rank": None,
                    "vector_rank": 5,
                    "found_by_engines": ["vector"],
                    "source_engine": "rrf_hybrid"
                }
            )
        ]
        
        search_results = {"hybrid_query": rrf_documents}
        ground_truth = {"hybrid_query": {"doc1", "doc2"}}
        
        # Test all metrics work with RRF metadata
        results = evaluate_search_results(
            queries={"hybrid_query": "sleep research query"},
            search_engine_results=search_results,
            ground_truth=ground_truth
        )
        
        # Verify evaluation works correctly
        assert results["Mean Reciprocal Rank (MRR)"].score == 1.0  # doc1 at rank 1
        assert results["Precision@10"].score == 2.0/3.0  # 2 relevant out of 3
        
        # MAP calculation: (1/1 + 2/2) / 2 = 1.0
        assert results["Mean Average Precision (MAP)"].score == 1.0


class TestEvaluationWithEmptyResults:
    """Test evaluation handles various empty result scenarios."""
    
    def test_evaluation_with_no_search_results(self):
        """Test evaluation when search returns no results."""
        queries = {"empty_query": "query with no results"}
        search_results = {"empty_query": []}  # Empty results
        ground_truth = {"empty_query": {"doc1", "doc2"}}
        
        results = evaluate_search_results(queries, search_results, ground_truth)
        
        # All metrics should return 0.0 for empty results
        for metric_name, result in results.items():
            assert result.score == 0.0
            assert result.query_scores["empty_query"] == 0.0
    
    def test_evaluation_with_no_ground_truth(self):
        """Test evaluation when no ground truth is available."""
        queries = {"no_truth_query": "query without ground truth"}
        search_results = {
            "no_truth_query": [
                Document(page_content="content", metadata={"id": "doc1"})
            ]
        }
        ground_truth = {}  # No ground truth
        
        results = evaluate_search_results(queries, search_results, ground_truth)
        
        # All metrics should handle missing ground truth gracefully
        for metric_name, result in results.items():
            assert result.score == 0.0
            assert len(result.query_scores) == 0
            assert result.metadata["queries_with_ground_truth"] == 0