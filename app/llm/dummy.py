"""Deterministic dummy LLM and embedding implementations.

Used when ``LLM_PROVIDER=dummy`` so the whole pipeline can run offline,
in CI, and in load tests without booting a real model server. Outputs are
intentionally deterministic so tests can assert exact values.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from typing import Any

from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class DummyChatModel(BaseChatModel):
    """Echoes a deterministic response derived from the last user message."""

    model_name: str = "dummy-chat"

    @property
    def _llm_type(self) -> str:
        return "dummy-chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        last = messages[-1].content if messages else ""
        text = f"[dummy:{self.model_name}] echo: {last}"
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(  # pragma: no cover - not needed for the prototype
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        yield result.generations[0]


class DummyEmbeddings(Embeddings):
    """Hash-based deterministic embeddings.

    Produces a fixed-dimension float vector derived from a SHA-256 hash of
    the input text. Useful for unit-testing the RAG plumbing without
    downloading a real embedding model.
    """

    def __init__(self, dim: int = 64) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Repeat the digest as needed and map bytes to [-1, 1).
        repeats = (self.dim + len(digest) - 1) // len(digest)
        raw = (digest * repeats)[: self.dim]
        return [(b / 128.0) - 1.0 for b in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
