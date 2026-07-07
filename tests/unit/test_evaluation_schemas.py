from backend.schemas.evaluation import EvaluationScores, EvaluationSummary


def test_evaluation_scores_overall_averages_available_metrics_only():
    scores = EvaluationScores(
        faithfulness=0.9,
        answer_relevancy=0.7,
        context_precision=None,
        context_recall=None,
    )

    assert scores.overall == 0.8
    assert scores.to_dict() == {
        "faithfulness": 0.9,
        "answer_relevancy": 0.7,
        "context_precision": None,
        "context_recall": None,
        "overall": 0.8,
    }


def test_evaluation_summary_aggregates_successful_results():
    summary = EvaluationSummary.from_scores([
        EvaluationScores(faithfulness=1.0, answer_relevancy=0.8, context_precision=0.6, context_recall=0.4),
        EvaluationScores(faithfulness=0.5, answer_relevancy=0.4, context_precision=None, context_recall=None),
    ], failed=1)

    assert summary.total == 3
    assert summary.successful == 2
    assert summary.failed == 1
    assert summary.averages["faithfulness"] == 0.75
    assert summary.averages["answer_relevancy"] == 0.6
    assert summary.averages["context_precision"] == 0.6
    assert summary.averages["context_recall"] == 0.4
    assert summary.averages["overall"] == 0.575
