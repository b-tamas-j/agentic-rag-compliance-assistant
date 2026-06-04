"""Small helpers for post-processing free-text LLM responses."""

from __future__ import annotations

import re

_THINK_BLOCK_RE = re.compile(r"<think\b[^>]*>.*?</think>", re.DOTALL | re.IGNORECASE)
_THINK_UNTERMINATED_RE = re.compile(r"<think\b[^>]*>.*\Z", re.DOTALL | re.IGNORECASE)
_THINK_ORPHAN_CLOSE_RE = re.compile(r"\A\s*</think>\s*", re.IGNORECASE)


def strip_think_tags(text: str) -> str:
    """Remove ``<think>...</think>`` reasoning blocks from an LLM response.

    Reasoning-tuned models (e.g. qwen3) emit a chain-of-thought wrapped in
    ``<think>...</think>`` before the actual answer. When we treat the raw
    response as a sub-query list or as a final answer, those tags leak into
    the downstream pipeline. This helper handles three cases:

    1. Fully-formed ``<think>...</think>`` blocks (possibly multi-line).
    2. An unterminated ``<think>`` that runs to the end of the response.
    3. An orphan ``</think>`` at the very start of the response.
    """
    if not text:
        return text
    text = _THINK_BLOCK_RE.sub("", text)
    text = _THINK_UNTERMINATED_RE.sub("", text)
    text = _THINK_ORPHAN_CLOSE_RE.sub("", text)
    return text.strip()


def looks_repetitive(text: str, *, window: int = 4, threshold: int = 3) -> bool:
    """Return True if ``text`` looks like a small-model degeneration loop.

    Tiny instruction models (e.g. qwen3:0.6b) sometimes get stuck repeating
    the same short phrase dozens of times instead of producing a real
    sentence. We flag a line as degenerate when any consecutive ``window``
    of words appears at least ``threshold`` times in it.
    """
    if not text:
        return False
    words = text.split()
    if len(words) < window * threshold:
        return False
    counts: dict[tuple[str, ...], int] = {}
    for i in range(len(words) - window + 1):
        ngram = tuple(words[i : i + window])
        counts[ngram] = counts.get(ngram, 0) + 1
        if counts[ngram] >= threshold:
            return True
    return False


__all__ = ["strip_think_tags", "looks_repetitive"]
