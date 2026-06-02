"""Per-node latency tracing for the load test.

LangGraph nodes are plain Python callables, so the cleanest way to time
each one is to monkey-patch the node-functions on `app.agent.nodes`
before the graph is compiled. The tracer is a small context manager
that records ``(node_name, elapsed_seconds)`` for every call and
restores the originals on exit.
"""

from __future__ import annotations

import contextlib
import statistics
import time
from collections import defaultdict
from collections.abc import Iterator
from typing import Any, Callable

from app.agent import graph as graph_mod
from app.agent import nodes as nodes_mod

_NODE_ATTRS: tuple[str, ...] = (
    "classify_query_node",
    "query_decomposer_node",
    "retrieve_documents_node",
    "tool_executor_node",
    "answer_generator_node",
    "hallucination_checker_node",
    "off_topic_handler_node",
)

# Modules that hold their own bound references to the node functions and
# therefore need patching too.
_PATCH_TARGETS = (nodes_mod, graph_mod)


class NodeTimings:
    """Collects per-node call durations across many graph invocations."""

    def __init__(self) -> None:
        self._samples: dict[str, list[float]] = defaultdict(list)

    def record(self, node: str, elapsed: float) -> None:
        self._samples[node].append(elapsed)

    @property
    def node_names(self) -> list[str]:
        return list(self._samples.keys())

    def samples(self, node: str) -> list[float]:
        return list(self._samples.get(node, []))

    def percentiles(self, node: str, ps: tuple[int, ...] = (50, 95, 99)) -> dict[str, float]:
        data = sorted(self._samples.get(node, []))
        if not data:
            return {f"p{p}": 0.0 for p in ps}
        out: dict[str, float] = {}
        for p in ps:
            k = max(0, min(len(data) - 1, int(round((p / 100.0) * (len(data) - 1)))))
            out[f"p{p}"] = data[k]
        return out

    def summary(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for node, data in self._samples.items():
            if not data:
                continue
            percs = self.percentiles(node)
            rows.append(
                {
                    "node": node,
                    "calls": len(data),
                    "mean": round(statistics.fmean(data), 4),
                    "min": round(min(data), 4),
                    **{k: round(v, 4) for k, v in percs.items()},
                    "max": round(max(data), 4),
                }
            )
        rows.sort(key=lambda r: r["node"])
        return rows


def _wrap(name: str, original: Callable[..., Any], timings: NodeTimings) -> Callable[..., Any]:
    def wrapped(state: dict[str, Any]) -> Any:
        started = time.perf_counter()
        try:
            return original(state)
        finally:
            timings.record(name, time.perf_counter() - started)

    wrapped.__name__ = getattr(original, "__name__", name)
    return wrapped


@contextlib.contextmanager
def trace_node_timings() -> Iterator[NodeTimings]:
    """Patch every agent node with a timing wrapper; restore on exit.

    Yields a fresh :class:`NodeTimings` instance that gets populated
    while the context is active. The graph must be compiled INSIDE the
    `with` block so it captures the wrapped functions.
    """
    timings = NodeTimings()
    originals: list[tuple[Any, str, Callable[..., Any]]] = []
    for attr in _NODE_ATTRS:
        node_name = attr.removesuffix("_node")
        # Take the original from the canonical location (`nodes_mod`).
        original = getattr(nodes_mod, attr)
        wrapped = _wrap(node_name, original, timings)
        for target in _PATCH_TARGETS:
            if hasattr(target, attr):
                originals.append((target, attr, getattr(target, attr)))
                setattr(target, attr, wrapped)
    try:
        yield timings
    finally:
        for target, attr, original in originals:
            setattr(target, attr, original)


__all__ = ["NodeTimings", "trace_node_timings"]
