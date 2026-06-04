"""Prompt templates for the RAG subgraph.

Kept separate from :mod:`app.rag.subgraph` so prompt tuning shows up
as small, isolated diffs.
"""

from __future__ import annotations

QUERY_TRANSFORM_PROMPT = (
    "Fogalmazd át az alábbi kérdést jogszabály-keresésre optimalizált, "
    "tömör formára magyarul. Tartsd meg a szakkifejezéseket (pl. ÁFA, TAO, "
    "fordított adózás), de hagyd el a kérdőszavakat és a felesleges szavakat. "
    "Csak az átfogalmazott kérdést add vissza, semmi mást.\n\n"
    "Kérdés: {query}"
)


GRADE_PROMPT = (
    "Te egy magyar számviteli/adójogi szakértő vagy. Az alábbi forrás"
    "részletek közül döntsd el, melyek RELEVÁNSAK a kérdés megválaszolásához. "
    "Add vissza a releváns részletek 1-alapú sorszámát a megadott "
    "szerkezetben. Ha egyik sem releváns, üres listát adj vissza.\n\n"
    "Kérdés: {query}\n\n"
    "Forrásrészletek:\n{chunks}"
)


__all__ = ["QUERY_TRANSFORM_PROMPT", "GRADE_PROMPT"]
