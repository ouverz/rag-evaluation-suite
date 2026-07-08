# RAGAS Evaluation Guide

This project now includes a real RAGAS evaluation surface for answer quality and retrieval quality.

The evaluation suite is intentionally separate from the retrieval/RRF tests:

- **RRF tests** prove the hybrid retrieval fusion logic behaves correctly.
- **RAGAS evaluation** scores generated RAG answers against retrieved context and optional ground-truth references.

---

## Metrics

All scores use a **0.0–1.0** scale. Higher is better.

| Metric | What it measures | Ground truth needed? |
|---|---|---|
| `faithfulness` | Whether the answer stays grounded in retrieved context | No |
| `answer_relevancy` | Whether the answer addresses the question | No |
| `context_precision` | Whether retrieved chunks are relevant to the reference answer | Yes |
| `context_recall` | Whether retrieved chunks contain the information needed by the reference answer | Yes |
| `overall` | Average of available metrics | Depends on supplied metrics |

If `reference` is omitted or empty, the evaluator runs only `faithfulness` and `answer_relevancy`.

---

## Dependencies

The evaluation layer uses optional runtime dependencies now listed in `pyproject.toml`:

- `ragas`
- `datasets`
- `langchain-openai`

The app can still report evaluation availability through `/evaluate/status` if dependencies or credentials are missing.

---

## REST API

All endpoints use `X-API-Key` when `RAG_API_KEYS` is configured.

### Check status

```bash
curl -H "X-API-Key: your-key" http://localhost:8000/evaluate/status
```

Example response:

```json
{
  "available": true,
  "module": "ragas",
  "metrics": ["faithfulness", "answer_relevancy", "context_precision", "context_recall"],
  "reason": null
}
```

### Evaluate one response

```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "question": "What are the benefits of bedtime routines?",
    "answer": "Consistent bedtime routines can improve sleep outcomes.",
    "contexts": [
      "A nightly bedtime routine was associated with improved sleep outcomes."
    ],
    "reference": "Consistent bedtime routines improve sleep outcomes in young children.",
    "use_cache": true
  }'
```

### Evaluate a batch

```bash
curl -X POST http://localhost:8000/evaluate/batch \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-key" \
  -d '{
    "samples": [
      {
        "question": "q1",
        "answer": "a1",
        "contexts": ["ctx1"],
        "reference": "r1"
      }
    ]
  }'
```

---

## Python API

```python
import asyncio
from core.evaluation.evaluator import RAGASEvaluator

async def main():
    evaluator = RAGASEvaluator()
    result = await evaluator.evaluate(
        question="What improves toddler sleep?",
        answer="Consistent bedtime routines can improve sleep outcomes.",
        contexts=["A consistent bedtime routine was associated with improved sleep outcomes."],
        reference="Consistent bedtime routines improve sleep outcomes.",
    )
    print(result.scores.to_dict())

asyncio.run(main())
```

---

## Golden dataset

A starter golden dataset lives at:

```text
data/eval/bedtime_routines_golden.json
```

Each record contains:

```json
{
  "question": "...",
  "reference": "...",
  "expected_source": "optional source document"
}
```

The current dataset is deliberately small. It is useful as a smoke benchmark and should be expanded before claiming robust answer-quality coverage.

---

## Benchmark CLI

The CLI queries the live `/query` endpoint, sends the generated answer and retrieved contexts to `/evaluate`, then writes JSON and optional Markdown reports.

Requirements:

1. Application running: `python start_app.py`
2. System initialised: `POST /init`
3. `OPENAI_API_KEY` set for RAGAS judge calls

```bash
python scripts/benchmark_ragas.py \
  --dataset data/eval/bedtime_routines_golden.json \
  --top-k 5 \
  --output results/ragas-benchmark.json \
  --markdown results/ragas-benchmark.md
```

---

## Caching

`RAGASEvaluator` keeps a bounded in-process LRU cache keyed by:

- question
- answer
- contexts
- reference
- metric set

Use `"use_cache": false` to force a fresh judge call.

---

## Portfolio interpretation

This suite gives the project two evaluation layers:

1. **Retrieval mechanics** — deterministic tests for RRF and hybrid search behavior.
2. **Answer quality** — RAGAS scoring for faithfulness, relevancy, context precision, and context recall.

That combination is stronger than a demo because it tests both *what gets retrieved* and *what the model says with it*.
