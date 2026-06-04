"""Prompt template for the LLM-as-judge evaluator.

Kept separate so the judge rubric can be tuned without touching
:mod:`app.eval.judge` orchestration code.
"""

from __future__ import annotations

JUDGE_PROMPT = (
    "You are an impartial judge evaluating an assistant that answers "
    "Hungarian corporate income tax (TAO) questions.\n"
    "Score the ANSWER below on three criteria, each on a 1-5 integer scale.\n\n"
    "QUESTION:\n{question}\n\n"
    "SOURCES (RAG):\n{context}\n\n"
    "ANSWER:\n{answer}\n\n"
    "Provide integer scores for:\n"
    "  - groundedness: how well the sources support the answer's claims\n"
    "  - relevance: how directly the answer addresses the question\n"
    "  - completeness: how thoroughly the answer covers the expected aspects\n"
    "Also include a brief Hungarian-language comment explaining the scores."
)


__all__ = ["JUDGE_PROMPT"]
