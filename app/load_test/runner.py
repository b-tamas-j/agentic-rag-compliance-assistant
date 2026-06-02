"""Async load-test runner for the agent graph.

Runs N queries through one compiled graph instance with a bounded
concurrency, then writes:

* ``reports/load_test_per_query.csv`` — one row per query (label,
  category match, total latency, sub-query count, etc.)
* ``reports/load_test_per_node.csv`` — aggregated per-node percentiles
* ``reports/load_test_per_node.png`` — bar chart of the same

Usage:
    uv run python -m app.load_test.runner
    uv run python -m app.load_test.runner --n 100 --concurrency 5
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import time
from pathlib import Path
from typing import Any

from app.load_test.scenarios import ScenarioQuery, build_query_mix
from app.load_test.timing import NodeTimings, trace_node_timings

logger = logging.getLogger(__name__)

DEFAULT_REPORT_DIR = Path(__file__).resolve().parents[2] / "reports"


def _category_predicted(state: dict[str, Any]) -> str:
    return state.get("category", "?") or "?"


async def _run_one(graph, scenario: ScenarioQuery, semaphore: asyncio.Semaphore) -> dict[str, Any]:
    async with semaphore:
        started = time.perf_counter()
        # graph.ainvoke is awaitable; falls back to a thread offload via to_thread for safety.
        try:
            state = await graph.ainvoke({"query": scenario.query})
        except Exception as exc:  # one bad query shouldn't abort the whole run
            logger.warning("query %r failed: %s", scenario.label, exc)
            return {
                "label": scenario.label,
                "category_expected": scenario.expected_category,
                "category_predicted": "ERROR",
                "category_correct": False,
                "total_latency_s": round(time.perf_counter() - started, 4),
                "error": str(exc),
            }
        elapsed = time.perf_counter() - started
        predicted = _category_predicted(state)
        return {
            "label": scenario.label,
            "category_expected": scenario.expected_category,
            "category_predicted": predicted,
            "category_correct": predicted == scenario.expected_category,
            "sub_queries": len(state.get("sub_queries") or []),
            "retrieved_docs": len(state.get("retrieved_docs") or []),
            "tool_calls": len(state.get("tool_results") or []),
            "total_latency_s": round(elapsed, 4),
        }


async def _drive(n: int, concurrency: int) -> tuple[list[dict[str, Any]], NodeTimings]:
    # Import inside the timed block so the graph captures the wrapped nodes.
    from app.agent import build_agent_graph

    with trace_node_timings() as timings:
        graph = build_agent_graph()
        queries = build_query_mix(n)
        semaphore = asyncio.Semaphore(concurrency)
        rows = await asyncio.gather(*(_run_one(graph, q, semaphore) for q in queries))
    return rows, timings


def _write_csv(rows: list[dict[str, Any]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    seen: set[str] = set()
    for r in rows:
        for k in r:
            if k not in seen:
                seen.add(k)
                fieldnames.append(k)
    with out_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _print_summary(rows: list[dict[str, Any]], timings: NodeTimings) -> None:
    correct = sum(1 for r in rows if r.get("category_correct"))
    errors = sum(1 for r in rows if r.get("category_predicted") == "ERROR")
    latencies = sorted(r["total_latency_s"] for r in rows if "total_latency_s" in r)
    if latencies:
        def pct(p: int) -> float:
            return latencies[max(0, min(len(latencies) - 1, int(round((p / 100.0) * (len(latencies) - 1)))))]

        print("\n--- End-to-end latency (per query) ---")
        print(f"  n           : {len(rows)}  (correct={correct}, errors={errors})")
        print(f"  p50         : {pct(50):.3f}s")
        print(f"  p95         : {pct(95):.3f}s")
        print(f"  p99         : {pct(99):.3f}s")
        print(f"  max         : {latencies[-1]:.3f}s")

    print("\n--- Per-node latency ---")
    print(f"  {'node':<25} {'calls':>6} {'p50':>8} {'p95':>8} {'p99':>8} {'max':>8}")
    for row in timings.summary():
        print(
            f"  {row['node']:<25} {row['calls']:>6}"
            f" {row['p50']:>8.3f} {row['p95']:>8.3f} {row['p99']:>8.3f} {row['max']:>8.3f}"
        )


def run(
    n: int = 50,
    concurrency: int = 5,
    report_dir: Path = DEFAULT_REPORT_DIR,
    *,
    render_chart: bool = True,
) -> tuple[list[dict[str, Any]], NodeTimings]:
    rows, timings = asyncio.run(_drive(n, concurrency))

    per_query = report_dir / "load_test_per_query.csv"
    per_node = report_dir / "load_test_per_node.csv"
    _write_csv(rows, per_query)
    _write_csv(timings.summary(), per_node)
    print(f"Wrote {per_query} and {per_node}")

    if render_chart:
        from app.load_test.chart import render_percentile_chart

        chart_path = render_percentile_chart(timings, report_dir / "load_test_per_node.png")
        print(f"Wrote chart {chart_path}")

    _print_summary(rows, timings)
    return rows, timings


def main() -> None:
    parser = argparse.ArgumentParser(description="Load-test the TAO compliance agent.")
    parser.add_argument("--n", type=int, default=50, help="Total number of queries.")
    parser.add_argument("--concurrency", type=int, default=5, help="Max concurrent in-flight queries.")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--no-chart", action="store_true", help="Skip the matplotlib chart.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.n, args.concurrency, args.out, render_chart=not args.no_chart)


if __name__ == "__main__":
    main()
