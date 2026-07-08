#!/usr/bin/env python3
"""Generate or run a small RAGAS benchmark for the live RAG API.

Dataset format (JSON or JSONL):
[
  {
    "question": "What are the benefits of bedtime routines?",
    "reference": "Consistent routines improve sleep outcomes.",
    "expected_source": "optional.pdf"
  }
]
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List
from urllib import request, error


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    data = json.loads(text)
    if not isinstance(data, list):
        raise ValueError("Dataset must be a JSON list or JSONL records")
    return data


def post_json(url: str, payload: Dict[str, Any], api_key: str | None = None, timeout: int = 120) -> Dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    req = request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"POST {url} failed with HTTP {exc.code}: {body}") from exc


def benchmark(dataset: Iterable[Dict[str, Any]], base_url: str, api_key: str | None, top_k: int) -> Dict[str, Any]:
    rows = []
    failures = []
    started = time.time()

    for idx, sample in enumerate(dataset, start=1):
        question = sample["question"]
        try:
            query_result = post_json(
                f"{base_url}/query",
                {"query": question, "top_k": top_k},
                api_key=api_key,
            )
            contexts = [row.get("content_preview", "") for row in query_result.get("results_table", [])]
            contexts = [ctx for ctx in contexts if ctx]

            eval_result = post_json(
                f"{base_url}/evaluate",
                {
                    "question": question,
                    "answer": query_result.get("answer", ""),
                    "contexts": contexts,
                    "reference": sample.get("reference"),
                    "use_cache": True,
                },
                api_key=api_key,
            )

            rows.append({
                "question": question,
                "answer_preview": query_result.get("answer", "")[:240],
                "scores": eval_result["scores"],
                "latency_ms": query_result.get("latency_ms"),
                "num_contexts": len(contexts),
                "expected_source": sample.get("expected_source"),
            })
            print(f"[{idx}] ok overall={eval_result['scores'].get('overall')}")
        except Exception as exc:  # Keep benchmark running and report failures.
            failures.append({"question": question, "error": str(exc)})
            print(f"[{idx}] failed: {exc}")

    summary = summarize(rows, failures, elapsed_seconds=time.time() - started)
    return {"summary": summary, "results": rows, "failures": failures}


def summarize(rows: List[Dict[str, Any]], failures: List[Dict[str, Any]], elapsed_seconds: float) -> Dict[str, Any]:
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall", "overall"]
    averages = {}
    mins = {}
    maxes = {}
    for metric in metrics:
        values = [row["scores"].get(metric) for row in rows if row["scores"].get(metric) is not None]
        averages[metric] = round(statistics.mean(values), 3) if values else None
        mins[metric] = round(min(values), 3) if values else None
        maxes[metric] = round(max(values), 3) if values else None

    return {
        "questions": len(rows) + len(failures),
        "successful": len(rows),
        "failed": len(failures),
        "elapsed_seconds": round(elapsed_seconds, 2),
        "averages": averages,
        "min": mins,
        "max": maxes,
    }


def write_markdown_report(result: Dict[str, Any], path: Path) -> None:
    summary = result["summary"]
    lines = [
        "# RAGAS Benchmark Report",
        "",
        "## Summary",
        "",
        f"- Questions: {summary['questions']}",
        f"- Successful: {summary['successful']}",
        f"- Failed: {summary['failed']}",
        f"- Elapsed seconds: {summary['elapsed_seconds']}",
        "",
        "| Metric | Avg | Min | Max |",
        "|---|---:|---:|---:|",
    ]
    for metric, avg in summary["averages"].items():
        lines.append(f"| {metric} | {avg} | {summary['min'][metric]} | {summary['max'][metric]} |")

    lines.extend(["", "## Per-question results", ""])
    for row in result["results"]:
        lines.extend([
            f"### {row['question']}",
            "",
            f"- Overall: {row['scores'].get('overall')}",
            f"- Faithfulness: {row['scores'].get('faithfulness')}",
            f"- Answer relevancy: {row['scores'].get('answer_relevancy')}",
            f"- Context precision: {row['scores'].get('context_precision')}",
            f"- Context recall: {row['scores'].get('context_recall')}",
            f"- Contexts: {row['num_contexts']}",
            "",
            row["answer_preview"],
            "",
        ])

    if result["failures"]:
        lines.extend(["", "## Failures", ""])
        for failure in result["failures"]:
            lines.append(f"- **{failure['question']}**: {failure['error']}")

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a RAGAS benchmark against the live RAG API")
    parser.add_argument("--dataset", type=Path, required=True, help="JSON/JSONL dataset of questions and references")
    parser.add_argument("--base-url", default="http://localhost:8000", help="RAG API base URL")
    parser.add_argument("--api-key", default=None, help="Optional X-API-Key value")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/ragas-benchmark.json"))
    parser.add_argument("--markdown", type=Path, default=None, help="Optional markdown report path")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    result = benchmark(dataset, args.base_url.rstrip("/"), args.api_key, args.top_k)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"Wrote JSON report to {args.output}")

    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        write_markdown_report(result, args.markdown)
        print(f"Wrote markdown report to {args.markdown}")


if __name__ == "__main__":
    main()
