from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import logging
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, Iterable, List, Optional

from backend.schemas.evaluation import EvaluationResult, EvaluationScores

logger = logging.getLogger(__name__)

REFERENCE_FREE_METRICS = ["faithfulness", "answer_relevancy"]
REFERENCE_REQUIRED_METRICS = ["context_precision", "context_recall"]
ALL_METRICS = REFERENCE_FREE_METRICS + REFERENCE_REQUIRED_METRICS

RagasRunner = Callable[[List[Dict[str, Any]], List[str]], Awaitable[List[Dict[str, Optional[float]]]]]


class RAGASEvaluator:
    """Evaluate RAG responses with RAGAS and cache repeated judge calls.

    The evaluator intentionally keeps RAGAS imports inside the default runner so
    the application can start, tests can run, and `/evaluate/status` can report
    clear availability even when optional evaluation dependencies are absent.
    """

    def __init__(self, ragas_runner: Optional[RagasRunner] = None, cache_size: int = 512):
        self._runner = ragas_runner or self._run_ragas
        self._cache_size = cache_size
        self._cache: OrderedDict[str, EvaluationResult] = OrderedDict()

    @staticmethod
    def availability() -> Dict[str, Any]:
        missing = [
            package
            for package in ("ragas", "langchain_openai")
            if importlib.util.find_spec(package) is None
        ]
        if missing:
            return {
                "available": False,
                "reason": f"Missing optional dependencies: {', '.join(missing)}",
                "metrics": ALL_METRICS,
            }
        return {"available": True, "reason": None, "metrics": ALL_METRICS}

    async def evaluate(
        self,
        question: str,
        answer: str,
        contexts: List[str],
        reference: Optional[str] = None,
        use_cache: bool = True,
    ) -> EvaluationResult:
        if not contexts:
            raise ValueError("RAGAS evaluation requires at least one context")

        selected_metrics = self._select_metrics(reference)
        cache_key = self._cache_key(question, answer, contexts, reference, selected_metrics)
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key].model_copy(deep=True)
            cached.metadata["cache_hit"] = True
            return cached

        sample = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
        }
        if reference:
            sample["reference"] = reference

        started = time.perf_counter()
        raw_scores = (await self._runner([sample], selected_metrics))[0]
        elapsed_ms = int((time.perf_counter() - started) * 1000)

        scores = EvaluationScores(
            faithfulness=self._normalise_score(raw_scores.get("faithfulness")),
            answer_relevancy=self._normalise_score(raw_scores.get("answer_relevancy")),
            context_precision=self._normalise_score(raw_scores.get("context_precision")),
            context_recall=self._normalise_score(raw_scores.get("context_recall")),
        )
        result = EvaluationResult(
            scores=scores,
            question=question,
            answer_preview=answer[:200],
            num_contexts=len(contexts),
            evaluation_time_ms=elapsed_ms,
            metadata={
                "cache_hit": False,
                "metrics": selected_metrics,
                "reference_provided": bool(reference),
            },
        )

        if use_cache:
            self._remember(cache_key, result)
        return result

    async def evaluate_batch(self, samples: Iterable[Dict[str, Any]]) -> List[EvaluationResult]:
        return await asyncio.gather(
            *[
                self.evaluate(
                    question=sample["question"],
                    answer=sample["answer"],
                    contexts=sample["contexts"],
                    reference=sample.get("reference"),
                    use_cache=sample.get("use_cache", True),
                )
                for sample in samples
            ]
        )

    @staticmethod
    def _select_metrics(reference: Optional[str]) -> List[str]:
        if reference and reference.strip():
            return ALL_METRICS
        return REFERENCE_FREE_METRICS

    @staticmethod
    def _cache_key(
        question: str,
        answer: str,
        contexts: List[str],
        reference: Optional[str],
        metrics: List[str],
    ) -> str:
        payload = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "reference": reference or "",
            "metrics": metrics,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()

    @staticmethod
    def _normalise_score(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return None
        if numeric < 0.0:
            numeric = 0.0
        if numeric > 1.0:
            numeric = 1.0
        return round(numeric, 3)

    def _remember(self, cache_key: str, result: EvaluationResult) -> None:
        self._cache[cache_key] = result.model_copy(deep=True)
        self._cache.move_to_end(cache_key)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    async def _run_ragas(
        self,
        samples: List[Dict[str, Any]],
        metrics: List[str],
    ) -> List[Dict[str, Optional[float]]]:
        """Run RAGAS against samples and return plain metric dictionaries."""

        def run_sync() -> List[Dict[str, Optional[float]]]:
            from datasets import Dataset
            from langchain_openai import ChatOpenAI, OpenAIEmbeddings
            from ragas import evaluate
            from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

            metric_lookup = {
                "faithfulness": faithfulness,
                "answer_relevancy": answer_relevancy,
                "context_precision": context_precision,
                "context_recall": context_recall,
            }
            selected = [metric_lookup[name] for name in metrics]

            dataset = Dataset.from_list(samples)
            result = evaluate(
                dataset,
                metrics=selected,
                llm=ChatOpenAI(model="gpt-4o-mini", temperature=0),
                embeddings=OpenAIEmbeddings(model="text-embedding-3-small"),
            )

            if hasattr(result, "to_pandas"):
                rows = result.to_pandas().to_dict(orient="records")
            else:
                rows = list(result)

            return [
                {metric: row.get(metric) for metric in ALL_METRICS}
                for row in rows
            ]

        return await asyncio.to_thread(run_sync)


_evaluator: Optional[RAGASEvaluator] = None


def get_ragas_evaluator() -> RAGASEvaluator:
    global _evaluator
    if _evaluator is None:
        _evaluator = RAGASEvaluator()
    return _evaluator
