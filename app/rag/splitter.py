"""Paragraph-aware text splitter for Hungarian legal/tax documents.

Hungarian statutes are structured around section markers like ``§`` and
sub-paragraphs like ``(1)``, ``(2)``. Splitting purely on character count
can rip a numbered clause in half, which destroys retrieval quality and
later citation accuracy.

This splitter first cuts at section boundaries (``§``) and then, if a
section is still too long, falls back to ``RecursiveCharacterTextSplitter``
for that section only. The original section header is prepended to every
sub-chunk so retrieved snippets stay self-contained.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Matches the start of a Hungarian statute section, e.g.:
#   "12. §", "12/A. §", "1. § (1)"
SECTION_HEADER_RE = re.compile(r"(?m)^\s*(\d+(?:/[A-ZÁÉÍÓÖŐÚÜŰ])?\.\s*§)")


@dataclass(frozen=True)
class SplitterConfig:
    chunk_size: int = 1000
    chunk_overlap: int = 200


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Return ``[(header, body)]`` pairs for each ``§`` section found.

    If no section header is detected, the whole text is returned as one
    pair with an empty header so the caller can still chunk it.
    """
    matches = list(SECTION_HEADER_RE.finditer(text))
    if not matches:
        return [("", text)]

    sections: list[tuple[str, str]] = []
    # Preamble before the first §, if any.
    if matches[0].start() > 0:
        preamble = text[: matches[0].start()].strip()
        if preamble:
            sections.append(("", preamble))

    for i, m in enumerate(matches):
        header = m.group(1).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        sections.append((header, body))
    return sections


def split_legal_text(
    text: str,
    *,
    source: str,
    page: int | None = None,
    config: SplitterConfig | None = None,
) -> list[Document]:
    """Split a single page (or whole doc) into citation-friendly chunks.

    Each returned ``Document`` carries metadata that the retriever and the
    answer generator rely on: ``source`` (file name), ``page`` (1-based
    page number when available) and ``section`` (the ``§`` header).
    """
    cfg = config or SplitterConfig()
    fallback = RecursiveCharacterTextSplitter(
        chunk_size=cfg.chunk_size,
        chunk_overlap=cfg.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    docs: list[Document] = []
    for header, body in _split_into_sections(text):
        if not body:
            continue
        full = f"{header} {body}".strip() if header else body
        if len(full) <= cfg.chunk_size:
            docs.append(_make_doc(full, source=source, page=page, section=header))
            continue
        # Long section -> recursive split, but keep the header in every piece.
        for piece in fallback.split_text(body):
            chunk_text = f"{header} {piece}".strip() if header else piece
            docs.append(_make_doc(chunk_text, source=source, page=page, section=header))
    return docs


def _make_doc(text: str, *, source: str, page: int | None, section: str) -> Document:
    metadata: dict[str, object] = {"source": source, "section": section}
    if page is not None:
        metadata["page"] = page
    return Document(page_content=text, metadata=metadata)
