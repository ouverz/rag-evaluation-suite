from __future__ import annotations

import logging
from fastapi import APIRouter, Depends, HTTPException, Request
from slowapi import Limiter
from slowapi.util import get_remote_address

from backend.schemas.evaluation import (
    EvaluationBatchRequest,
    EvaluationBatchResponse,
    EvaluationResult,
    EvaluationSample,
    EvaluationStatus,
    EvaluationSummary,
)
from backend.security import require_api_key
from core.evaluation.evaluator import RAGASEvaluator, get_ragas_evaluator

logger = logging.getLogger(__name__)

router = APIRouter()
limiter = Limiter(key_func=get_remote_address)


@router.get("/status", response_model=EvaluationStatus)
def evaluation_status():
    status = RAGASEvaluator.availability()
    return EvaluationStatus(
        available=status["available"],
        metrics=status["metrics"],
        reason=status.get("reason"),
    )


@router.post("", response_model=EvaluationResult)
@limiter.limit("20/minute")
async def evaluate_response(
    request: Request,
    sample: EvaluationSample,
    api_key: str = Depends(require_api_key),
):
    try:
        evaluator = get_ragas_evaluator()
        return await evaluator.evaluate(
            question=sample.question,
            answer=sample.answer,
            contexts=sample.contexts,
            reference=sample.reference,
            use_cache=sample.use_cache,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("RAGAS evaluation failed")
        raise HTTPException(status_code=500, detail=f"RAGAS evaluation failed: {exc}") from exc


@router.post("/batch", response_model=EvaluationBatchResponse)
@limiter.limit("5/minute")
async def evaluate_batch(
    request: Request,
    batch: EvaluationBatchRequest,
    api_key: str = Depends(require_api_key),
):
    evaluator = get_ragas_evaluator()
    results = []
    failed = 0

    for sample in batch.samples:
        try:
            result = await evaluator.evaluate(
                question=sample.question,
                answer=sample.answer,
                contexts=sample.contexts,
                reference=sample.reference,
                use_cache=sample.use_cache,
            )
            results.append(result)
        except Exception:
            failed += 1
            logger.exception("RAGAS batch sample failed")

    summary = EvaluationSummary.from_scores([result.scores for result in results], failed=failed)
    return EvaluationBatchResponse(results=results, summary=summary)
