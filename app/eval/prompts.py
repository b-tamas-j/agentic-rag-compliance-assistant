"""Prompt template for the LLM-as-judge evaluator.

Kept separate so the judge rubric can be tuned without touching
:mod:`app.eval.judge` orchestration code.
"""

from __future__ import annotations

JUDGE_PROMPT = (
    "Bíró vagy egy magyar társasági adó (TAO) tanácsadó asszisztens kiértékelésében.\n"
    "Pontozd az alábbi VÁLASZ minőségét három szempont szerint 1-5 skálán.\n\n"
    "KÉRDÉS:\n{question}\n\n"
    "FORRÁSOK (RAG):\n{context}\n\n"
    "VÁLASZ:\n{answer}\n\n"
    "Adj integer pontszámot a groundedness (mennyire támasztják alá a források), "
    "relevance (mennyire válaszol a kérdésre) és completeness (mennyire lefedő) "
    "mezőkre, és írj rövid magyar megjegyzést."
)


__all__ = ["JUDGE_PROMPT"]
