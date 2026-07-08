from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, computed_field


class EvaluationSample(BaseModel):
    """One RAG response to score with RAGAS."""

    question: str = Field(..., min_length=1)
    answer: str = Field(..., min_length=1)
    contexts: List[str] = Field(..., min_length=1)
    reference: Optional[str] = Field(default=None, description="Ground-truth answer for retrieval-aware metrics")
    use_cache: bool = True


class EvaluationScores(BaseModel):
    """RAGAS metric scores on a 0.0-1.0 scale."""

    faithfulness: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    answer_relevancy: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    context_precision: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    context_recall: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @computed_field
    @property
    def overall(self) -> Optional[float]:
        values = [
            self.faithfulness,
            self.answer_relevancy,
            self.context_precision,
            self.context_recall,
        ]
        present = [v for v in values if v is not None]
        if not present:
            return None
        return round(sum(present) / len(present), 3)

    def to_dict(self) -> Dict[str, Optional[float]]:
        return {
            "faithfulness": self.faithfulness,
            "answer_relevancy": self.answer_relevancy,
            "context_precision": self.context_precision,
            "context_recall": self.context_recall,
            "overall": self.overall,
        }


class EvaluationResult(BaseModel):
    """Result for one evaluated RAG sample."""

    scores: EvaluationScores
    question: str
    answer_preview: str
    num_contexts: int
    evaluation_time_ms: int
    metadata: Dict[str, Any] = Field(default_factory=dict)


class EvaluationBatchRequest(BaseModel):
    samples: List[EvaluationSample] = Field(..., min_length=1, max_length=100)


class EvaluationSummary(BaseModel):
    total: int
    successful: int
    failed: int
    averages: Dict[str, Optional[float]]

    @classmethod
    def from_scores(cls, scores: List[EvaluationScores], failed: int = 0) -> "EvaluationSummary":
        keys = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "overall"]
        averages: Dict[str, Optional[float]] = {}
        for key in keys:
            values = [getattr(score, key) for score in scores if getattr(score, key) is not None]
            averages[key] = round(sum(values) / len(values), 3) if values else None

        return cls(
            total=len(scores) + failed,
            successful=len(scores),
            failed=failed,
            averages=averages,
        )


class EvaluationBatchResponse(BaseModel):
    results: List[EvaluationResult]
    summary: EvaluationSummary


class EvaluationStatus(BaseModel):
    available: bool
    module: str = "ragas"
    metrics: List[str]
    reason: Optional[str] = None
