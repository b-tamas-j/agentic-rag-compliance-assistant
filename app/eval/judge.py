"""LLM-as-judge for compliance answer quality.

Asks the judge model to score the answer on three axes (1-5 each):
groundedness, relevance, completeness. Falls back to a deterministic
heuristic under the dummy provider so the eval CLI is runnable offline.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field

from app.llm import get_chat_model
from app.llm.dummy import DummyChatModel

logger = logging.getLogger(__name__)


class JudgeVerdict(BaseModel):
    groundedness: int = Field(
        ge=1, le=5, description="1=invented, 5=every claim is supported by the sources."
    )
    relevance: int = Field(ge=1, le=5, description="How directly the answer addresses the question.")
    completeness: int = Field(ge=1, le=5, description="Whether the answer covers the expected aspects.")
    comment: str = Field(default="", description="Short Hungarian justification.")


_JUDGE_PROMPT = (
    "Bíró vagy egy magyar társasági adó (TAO) tanácsadó asszisztens kiértékelésében.\n"
    "Pontozd az alábbi VÁLASZ minőségét három szempont szerint 1-5 skálán.\n\n"
    "KÉRDÉS:\n{question}\n\n"
    "FORRÁSOK (RAG):\n{context}\n\n"
    "VÁLASZ:\n{answer}\n\n"
    "Adj integer pontszámot a groundedness (mennyire támasztják alá a források), "
    "relevance (mennyire válaszol a kérdésre) és completeness (mennyire lefedő) "
    "mezőkre, és írj rövid magyar megjegyzést."
)


def _format_context(docs: list[Document]) -> str:
    if not docs:
        return "(nincs forrás)"
    parts = []
    for i, doc in enumerate(docs[:8], 1):
        section = (doc.metadata or {}).get("section", "?")
        parts.append(f"[{i}] ({section}) {doc.page_content[:400]}")
    return "\n".join(parts)


def _dummy_verdict(answer: str, docs: list[Document]) -> JudgeVerdict:
    """Cheap heuristic so the CLI runs without an LLM."""
    has_source = bool(docs)
    long_enough = len(answer or "") > 40
    groundedness = 4 if has_source else 2
    relevance = 4 if long_enough else 2
    completeness = 3 if has_source and long_enough else 2
    return JudgeVerdict(
        groundedness=groundedness,
        relevance=relevance,
        completeness=completeness,
        comment="(dummy judge — heurisztika alapján)",
    )


def judge_answer(
    question: str,
    answer: str,
    retrieved_docs: list[Document],
) -> dict[str, Any]:
    """Return the judge verdict as a plain dict (so it serialises into CSV)."""
    chat = get_chat_model("judge")
    if isinstance(chat, DummyChatModel):
        verdict = _dummy_verdict(answer, retrieved_docs)
    else:
        judge = chat.with_structured_output(JudgeVerdict)
        try:
            verdict = judge.invoke(  # type: ignore[assignment]
                _JUDGE_PROMPT.format(
                    question=question,
                    context=_format_context(retrieved_docs),
                    answer=answer,
                )
            )
        except Exception as exc:
            logger.warning("Judge failed, falling back to heuristic: %s", exc)
            verdict = _dummy_verdict(answer, retrieved_docs)
    return verdict.model_dump()


__all__ = ["JudgeVerdict", "judge_answer"]
