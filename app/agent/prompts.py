"""Prompt templates for the main agentic workflow.

Instructions are written in English (where instruction-following on
small open-weight LLMs is typically stronger) while domain terms,
section headers, and any user-visible output remain in Hungarian via
explicit directives ('Respond in Hungarian.', 'Tao. tv. N. §').

Note: this is an opinionated style choice. The previous version of
this file kept the instructions in Hungarian; see the git history
for the comparison.
"""

from __future__ import annotations

CLASSIFY_PROMPT = (
    "You are a Hungarian tax-law assistant. Decide whether the question below "
    "falls within the scope of Hungarian CORPORATE INCOME TAX (Tao. tv.) "
    "and, if so, which source PDF is most relevant.\n\n"
    "Question: {query}\n\n"
    "Respond in the required structured format:\n"
    "  - category: 'tao' or 'off_topic'\n"
    "  - source_hint: one of 'nonprofit' (közhasznú / alapítvány / "
    "egyesület), 'calculation' (general TAO rules: adóalap, adómérték, "
    "elhatárolt veszteség, adókedvezmény), 'offering' (Tao-felajánlás), "
    "'credit' (növekedési adóhitel / NAHI), or 'general' if no single "
    "source clearly dominates."
)


DECOMPOSE_PROMPT = (
    "Break the following Hungarian tax-law question into 1-3 short, "
    "independently searchable sub-questions. Put each sub-question on its "
    "own line, with no numbering. If the question is already simple enough, "
    "return it as a single line.\n\n"
    "Keep the sub-questions in Hungarian (the corpus is Hungarian).\n\n"
    "Question: {query}"
)


ANSWER_PROMPT = (
    "You are an expert on Hungarian corporate income tax (Tao. tv.). "
    "Answer the question below using ONLY the provided sources. If the "
    "sources do not cover the question, say so honestly.\n\n"
    "CITATION RULES:\n"
    "- Cite paragraphs in the format 'Tao. tv. N. §' (e.g. 'Tao. tv. 19. §' "
    "or 'Tao. tv. 24/A. §').\n"
    "- NEVER place the source PDF filename inside a citation. The filename "
    "is metadata, not a citation element.\n"
    "- If a tool (e.g. tao_calculator) returned a numeric result, USE that "
    "result in the answer — do not recompute it in your head.\n\n"
    "STYLE: Be CONCISE — at most 4-5 sentences. Do not repeat the question. "
    "Do not produce bullet lists; give only the essential content.\n\n"
    "LANGUAGE: Respond in HUNGARIAN. Keep Hungarian legal terminology "
    "(adóalap, elhatárolt veszteség, közhasznú, etc.) intact.\n\n"
    "Question: {query}\n\n"
    "Sources:\n{context}\n\n"
    "Tool results:\n{tools}\n\n"
    "Hungarian answer:"
)


GROUNDEDNESS_PROMPT = (
    "Verify whether every factual claim in the ANSWER below is supported "
    "by the provided SOURCES. If anything is missing from the sources, "
    "return 'grounded=false'. If every claim traces back to the sources, "
    "return 'grounded=true'. Return only the logical verdict, no "
    "explanation.\n\n"
    "SOURCES:\n{context}\n\n"
    "ANSWER:\n{answer}"
)


__all__ = [
    "CLASSIFY_PROMPT",
    "DECOMPOSE_PROMPT",
    "ANSWER_PROMPT",
    "GROUNDEDNESS_PROMPT",
]
