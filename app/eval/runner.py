"""CLI runner that evaluates the agent over the labelled dataset.

Usage:
    uv run python -m app.eval.runner
    uv run python -m app.eval.runner --dataset data/eval/questions.json --out reports/eval.csv

Writes a per-question CSV (one row per question, all metrics + judge
scores) and prints a summary table to stdout.
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path
from typing import Any

from app.agent import build_agent_graph
from app.eval.dataset import DEFAULT_DATASET_PATH, EvalQuestion, load_dataset
from app.eval.judge import judge_answer
from app.eval.metrics import (
    aggregate,
    category_correct,
    citation_accuracy,
    expected_terms_coverage,
    retrieval_recall_at_k,
)

logger = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"


def _evaluate_one(graph, question: EvalQuestion, *, with_judge: bool) -> dict[str, Any]:
    started = time.perf_counter()
    state = graph.invoke({"query": question.question})
    elapsed = time.perf_counter() - started

    answer = state.get("final_answer") or state.get("draft_answer") or ""
    docs = state.get("retrieved_docs") or []
    predicted_category = state.get("category")

    row: dict[str, Any] = {
        "id": question.id,
        "category_expected": question.category,
        "category_predicted": predicted_category,
        "category_correct": category_correct(question, predicted_category),
        "recall_at_k": round(retrieval_recall_at_k(question, docs), 4),
        "citation_accuracy": round(citation_accuracy(question, answer, docs), 4),
        "terms_coverage": round(expected_terms_coverage(question, answer), 4),
        "grounded": bool(state.get("grounded", True)),
        "latency_s": round(elapsed, 3),
        "answer": answer,
    }

    if with_judge and question.category == "tao":
        verdict = judge_answer(question.question, answer, docs)
        row["judge_groundedness"] = verdict["groundedness"]
        row["judge_relevance"] = verdict["relevance"]
        row["judge_completeness"] = verdict["completeness"]
        row["judge_comment"] = verdict["comment"]

    return row


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r.keys():
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict[str, Any]]) -> None:
    summary = aggregate(rows)
    print("\n--- Evaluation summary ---")
    for key in (
        "category_correct",
        "recall_at_k",
        "citation_accuracy",
        "terms_coverage",
        "grounded",
        "latency_s",
        "judge_groundedness",
        "judge_relevance",
        "judge_completeness",
    ):
        if key in summary:
            print(f"  {key:>22}: {summary[key]:.3f}")
    print(f"  {'n_questions':>22}: {len(rows)}")


def run(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    out_path: Path | None = None,
    *,
    with_judge: bool = True,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    questions = load_dataset(dataset_path)
    if limit:
        questions = questions[:limit]

    graph = build_agent_graph()
    rows: list[dict[str, Any]] = []
    for q in questions:
        logger.info("Evaluating %s (%s)", q.id, q.topic)
        rows.append(_evaluate_one(graph, q, with_judge=with_judge))

    out_path = out_path or DEFAULT_REPORT_DIR / "eval.csv"
    _write_csv(rows, out_path)
    print(f"Wrote {len(rows)} rows -> {out_path}")
    _print_summary(rows)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the TAO compliance agent.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument(
        "--no-judge",
        action="store_true",
        help="Skip the LLM-as-judge step (deterministic metrics only).",
    )
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N questions.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.dataset, args.out, with_judge=not args.no_judge, limit=args.limit)


if __name__ == "__main__":
    main()
