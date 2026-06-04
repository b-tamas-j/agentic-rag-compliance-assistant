"""Prompt templates for the RAG subgraph.

Kept separate from :mod:`app.rag.subgraph` so prompt tuning shows up
as small, isolated diffs.
"""

from __future__ import annotations

QUERY_TRANSFORM_PROMPT = (
    "Rewrite the question below into a concise form optimised for legal-"
    "text retrieval. Keep Hungarian domain terms intact (e.g. ÁFA, TAO, "
    "fordított adózás), but drop interrogatives and filler words. The "
    "rewrite MUST stay in Hungarian (the corpus is Hungarian). Return only "
    "the rewritten question, nothing else.\n\n"
    "Question: {query}"
)


GRADE_PROMPT = (
    "You are an expert on Hungarian accounting and tax law. From the source "
    "snippets below, decide which ones are RELEVANT for answering the "
    "question. Return the 1-based indexes of the relevant snippets in the "
    "required structured format. If none are relevant, return an empty list.\n\n"
    "Question: {query}\n\n"
    "Snippets:\n{chunks}"
)


__all__ = ["QUERY_TRANSFORM_PROMPT", "GRADE_PROMPT"]
