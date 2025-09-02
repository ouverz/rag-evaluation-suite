#!/usr/bin/env python3
"""
Example script demonstrating how to use the evaluation metrics module.

This script shows how to evaluate RAG system performance using Information
Retrieval metrics: Mean Reciprocal Rank (MRR), Precision@K, and Mean Average 
Precision (MAP).
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from typing import Dict, Set
from langchain.schema import Document

from core.evaluation.metrics import (
    MeanReciprocalRank,
    PrecisionAtK,
    MeanAveragePrecision,
    evaluate_search_results
)


def create_sample_data():
    """Create sample search results and ground truth for demonstration."""
    
    # Sample queries
    queries = {
        "sleep_patterns": "How do sleep patterns affect child development?",
        "bedtime_routines": "What are effective bedtime routines for children?",
        "sleep_duration": "How much sleep do children need by age?"
    }
    
    # Sample search results from RAG system (Document format)
    search_results = {
        "sleep_patterns": [
            Document(
                page_content="Sleep patterns significantly impact cognitive development in children. Research shows that consistent sleep schedules...",
                metadata={"id": "paper1_chunk3", "source": "sleep_research_2023.pdf", "rrf_score": 0.92}
            ),
            Document(
                page_content="Child development studies indicate that sleep quality affects learning outcomes and memory consolidation...",
                metadata={"id": "paper2_chunk1", "source": "child_dev_study.pdf", "rrf_score": 0.85}
            ),
            Document(
                page_content="Irregular sleep patterns can lead to behavioral issues and reduced academic performance in school-age children...",
                metadata={"id": "paper1_chunk7", "source": "sleep_research_2023.pdf", "rrf_score": 0.78}
            ),
            Document(
                page_content="Nutrition plays a role in sleep quality, but sleep patterns remain the primary factor in development...",
                metadata={"id": "paper3_chunk2", "source": "nutrition_sleep.pdf", "rrf_score": 0.45}
            )
        ],
        
        "bedtime_routines": [
            Document(
                page_content="Consistent bedtime routines help children fall asleep faster and improve sleep quality. Recommended routines include...",
                metadata={"id": "routine_study_chunk1", "source": "bedtime_routines_2024.pdf", "rrf_score": 0.95}
            ),
            Document(
                page_content="Reading stories before bed as part of a routine has been shown to reduce sleep onset time by an average of 15 minutes...",
                metadata={"id": "routine_study_chunk4", "source": "bedtime_routines_2024.pdf", "rrf_score": 0.88}
            ),
            Document(
                page_content="Screen time before bed should be avoided as it disrupts natural circadian rhythms and delays sleep onset...",
                metadata={"id": "screen_time_chunk2", "source": "digital_health_study.pdf", "rrf_score": 0.72}
            )
        ],
        
        "sleep_duration": [
            Document(
                page_content="Age-appropriate sleep duration guidelines: toddlers (11-14 hours), preschoolers (10-13 hours), school-age (9-11 hours)...",
                metadata={"id": "guidelines_chunk1", "source": "sleep_guidelines_2024.pdf", "rrf_score": 0.98}
            ),
            Document(
                page_content="Insufficient sleep duration is linked to obesity, behavioral problems, and reduced immune function in children...",
                metadata={"id": "duration_study_chunk3", "source": "sleep_duration_effects.pdf", "rrf_score": 0.82}
            ),
            Document(
                page_content="Sleep duration needs vary by individual child, but guidelines provide a helpful baseline for parents and caregivers...",
                metadata={"id": "guidelines_chunk5", "source": "sleep_guidelines_2024.pdf", "rrf_score": 0.65}
            )
        ]
    }
    
    # Ground truth relevance judgments (expert annotations)
    ground_truth = {
        "sleep_patterns": {
            "paper1_chunk3",    # Highly relevant - directly addresses question
            "paper2_chunk1",    # Relevant - discusses impact on development
            "paper1_chunk7"     # Relevant - discusses behavioral impacts
            # "paper3_chunk2" not included - more about nutrition than sleep patterns
        },
        
        "bedtime_routines": {
            "routine_study_chunk1",  # Highly relevant - core bedtime routine info
            "routine_study_chunk4",  # Relevant - specific routine practice
            "screen_time_chunk2"     # Relevant - routine guidelines (what to avoid)
        },
        
        "sleep_duration": {
            "guidelines_chunk1",     # Highly relevant - direct answer to question
            "duration_study_chunk3", # Relevant - effects of insufficient duration
            "guidelines_chunk5"      # Somewhat relevant - discusses individual variation
        }
    }
    
    return queries, search_results, ground_truth


def demonstrate_individual_metrics():
    """Demonstrate using individual evaluation metrics."""
    print("\n=== Individual Metrics Demonstration ===")
    
    # Simple example data
    search_results = {
        "query1": ["doc1", "doc2", "doc3", "doc4"],
        "query2": ["doc5", "doc6", "doc7", "doc8"]
    }
    
    ground_truth = {
        "query1": {"doc1", "doc3"},  # Relevant docs at ranks 1 and 3
        "query2": {"doc6", "doc7"}   # Relevant docs at ranks 2 and 3
    }
    
    # Mean Reciprocal Rank
    print("\n1. Mean Reciprocal Rank (MRR)")
    mrr = MeanReciprocalRank()
    mrr_result = mrr.calculate(search_results, ground_truth)
    print(f"   MRR Score: {mrr_result.score:.4f}")
    print(f"   Per-query scores: {mrr_result.query_scores}")
    print(f"   Interpretation: Average of 1/rank for first relevant document per query")
    
    # Precision@K
    print("\n2. Precision@3")
    precision = PrecisionAtK(k=3)
    precision_result = precision.calculate(search_results, ground_truth)
    print(f"   Precision@3 Score: {precision_result.score:.4f}")
    print(f"   Per-query scores: {precision_result.query_scores}")
    print(f"   Interpretation: Fraction of top-3 results that are relevant")
    
    # Mean Average Precision
    print("\n3. Mean Average Precision (MAP)")
    map_metric = MeanAveragePrecision()
    map_result = map_metric.calculate(search_results, ground_truth)
    print(f"   MAP Score: {map_result.score:.4f}")
    print(f"   Per-query scores: {map_result.query_scores}")
    print(f"   Interpretation: Average of precision values at each relevant document position")


def demonstrate_rag_evaluation():
    """Demonstrate evaluation with realistic RAG system data."""
    print("\n=== RAG System Evaluation Demonstration ===")
    
    queries, search_results, ground_truth = create_sample_data()
    
    # Evaluate using the main function
    results = evaluate_search_results(queries, search_results, ground_truth)
    
    print(f"\nEvaluation Results for {len(queries)} queries:")
    print("-" * 50)
    
    for metric_name, result in results.items():
        print(f"\n{metric_name}:")
        print(f"  Overall Score: {result.score:.4f}")
        print(f"  Query Breakdown:")
        for query_id, score in result.query_scores.items():
            query_text = queries[query_id][:50] + "..." if len(queries[query_id]) > 50 else queries[query_id]
            print(f"    {query_id}: {score:.4f} ('{query_text}')")
        
        # Show metadata
        metadata = result.metadata
        print(f"  Metadata:")
        print(f"    Total queries: {metadata.get('total_queries', 'N/A')}")
        print(f"    Queries with ground truth: {metadata.get('queries_with_ground_truth', 'N/A')}")
        if 'perfect_precision_queries' in metadata:
            print(f"    Perfect scores: {metadata['perfect_precision_queries']}")
        if 'zero_reciprocal_ranks' in metadata:
            print(f"    Zero scores: {metadata['zero_reciprocal_ranks']}")


def demonstrate_dataframe_evaluation():
    """Demonstrate evaluation with DataFrame results (as from hybrid search)."""
    print("\n=== DataFrame Results Evaluation ===")
    
    # Simulate DataFrame results from hybrid search engine
    df_results = pd.DataFrame({
        "id": ["doc1", "doc2", "doc3", "doc4", "doc5"],
        "content": [
            "Content about sleep patterns and child development...",
            "Research on bedtime routines for toddlers...",
            "Sleep duration guidelines for different ages...",
            "Effects of screen time on sleep quality...",
            "Nutrition and sleep interaction study..."
        ],
        "rrf_score": [0.95, 0.82, 0.76, 0.58, 0.43],
        "bm25_rank": [1, 3, 2, 4, 5],
        "vector_rank": [2, 1, 4, 5, 3],
        "source_engine": ["rrf_hybrid"] * 5
    })
    
    search_results = {"dataframe_query": df_results}
    ground_truth = {"dataframe_query": {"doc1", "doc2", "doc3"}}
    
    # Evaluate
    results = evaluate_search_results(
        queries={"dataframe_query": "comprehensive sleep research query"},
        search_engine_results=search_results,
        ground_truth=ground_truth
    )
    
    print(f"Results for DataFrame input:")
    for metric_name, result in results.items():
        print(f"  {metric_name}: {result.score:.4f}")


def demonstrate_custom_metrics():
    """Demonstrate using custom metric configurations."""
    print("\n=== Custom Metrics Configuration ===")
    
    queries, search_results, ground_truth = create_sample_data()
    
    # Use custom metrics with different parameters
    custom_metrics = [
        MeanReciprocalRank(),
        PrecisionAtK(k=3),   # Focus on top-3 precision
        PrecisionAtK(k=5),   # Also check top-5 precision
        MeanAveragePrecision()
    ]
    
    results = evaluate_search_results(
        queries, search_results, ground_truth, metrics=custom_metrics
    )
    
    print(f"Custom evaluation with {len(custom_metrics)} metrics:")
    for metric_name, result in results.items():
        print(f"  {metric_name}: {result.score:.4f}")


if __name__ == "__main__":
    print("RAG System Evaluation Metrics Demo")
    print("=" * 50)
    
    # Run demonstrations
    demonstrate_individual_metrics()
    demonstrate_rag_evaluation()
    demonstrate_dataframe_evaluation()
    demonstrate_custom_metrics()
    
    print("\n" + "=" * 50)
    print("Demo completed successfully!")
    print("\nKey takeaways:")
    print("- MRR focuses on finding the first relevant document quickly")
    print("- Precision@K measures relevance in top K results") 
    print("- MAP provides comprehensive ranking quality assessment")
    print("- All metrics work with Document objects and DataFrames")
    print("- Higher scores indicate better search performance")