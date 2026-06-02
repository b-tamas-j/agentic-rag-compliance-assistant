"""Render a per-node latency chart from `NodeTimings`."""

from __future__ import annotations

from pathlib import Path

from app.load_test.timing import NodeTimings


def render_percentile_chart(timings: NodeTimings, out_path: Path) -> Path:
    """Save a grouped bar chart (p50/p95/p99 per node) to ``out_path``.

    Matplotlib is imported lazily so the rest of the load-test package
    stays importable in environments where mpl is not installed.
    """
    import matplotlib

    matplotlib.use("Agg")  # headless backend for CI / containers
    import matplotlib.pyplot as plt

    nodes = sorted(timings.node_names)
    p50 = [timings.percentiles(n)["p50"] for n in nodes]
    p95 = [timings.percentiles(n)["p95"] for n in nodes]
    p99 = [timings.percentiles(n)["p99"] for n in nodes]

    width = 0.25
    x = list(range(len(nodes)))

    fig, ax = plt.subplots(figsize=(max(8, len(nodes) * 1.4), 5))
    ax.bar([xi - width for xi in x], p50, width=width, label="p50")
    ax.bar(x, p95, width=width, label="p95")
    ax.bar([xi + width for xi in x], p99, width=width, label="p99")

    ax.set_xticks(x)
    ax.set_xticklabels(nodes, rotation=30, ha="right")
    ax.set_ylabel("Latency (s)")
    ax.set_title("Per-node latency percentiles")
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


__all__ = ["render_percentile_chart"]
