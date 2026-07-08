import pytest

from core.evaluation.evaluator import RAGASEvaluator


@pytest.mark.asyncio
async def test_evaluator_scores_reference_free_metrics_without_ground_truth():
    calls = []

    async def fake_runner(samples, metrics):
        calls.append((samples, metrics))
        return [{"faithfulness": 0.8, "answer_relevancy": 0.6}]

    evaluator = RAGASEvaluator(ragas_runner=fake_runner, cache_size=8)

    result = await evaluator.evaluate(
        question="What improves toddler sleep?",
        answer="Consistent bedtime routines can improve sleep outcomes.",
        contexts=["A consistent bedtime routine was associated with improved sleep outcomes."],
    )

    assert result.scores.faithfulness == 0.8
    assert result.scores.answer_relevancy == 0.6
    assert result.scores.context_precision is None
    assert result.scores.context_recall is None
    assert result.scores.overall == pytest.approx(0.7)
    assert calls[0][1] == ["faithfulness", "answer_relevancy"]


@pytest.mark.asyncio
async def test_evaluator_adds_grounded_retrieval_metrics_when_reference_present():
    async def fake_runner(samples, metrics):
        assert metrics == [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
        assert samples[0]["reference"] == "Bedtime routines improve child sleep."
        return [{
            "faithfulness": 0.9,
            "answer_relevancy": 0.8,
            "context_precision": 0.7,
            "context_recall": 0.6,
        }]

    evaluator = RAGASEvaluator(ragas_runner=fake_runner, cache_size=8)

    result = await evaluator.evaluate(
        question="What improves child sleep?",
        answer="Bedtime routines improve child sleep.",
        contexts=["A bedtime routine improves sleep duration."],
        reference="Bedtime routines improve child sleep.",
    )

    assert result.scores.overall == pytest.approx(0.75)
    assert result.num_contexts == 1
    assert result.metadata["cache_hit"] is False


@pytest.mark.asyncio
async def test_evaluator_caches_repeated_evaluations():
    call_count = 0

    async def fake_runner(samples, metrics):
        nonlocal call_count
        call_count += 1
        return [{"faithfulness": 1.0, "answer_relevancy": 1.0}]

    evaluator = RAGASEvaluator(ragas_runner=fake_runner, cache_size=8)

    first = await evaluator.evaluate("q", "a", ["ctx"], use_cache=True)
    second = await evaluator.evaluate("q", "a", ["ctx"], use_cache=True)

    assert call_count == 1
    assert first.metadata["cache_hit"] is False
    assert second.metadata["cache_hit"] is True


@pytest.mark.asyncio
async def test_evaluator_rejects_empty_contexts():
    evaluator = RAGASEvaluator(ragas_runner=lambda samples, metrics: [])

    with pytest.raises(ValueError, match="at least one context"):
        await evaluator.evaluate("q", "a", [])
