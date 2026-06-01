"""Document ingestion pipeline.

Reads PDFs (via pdfplumber, with a pypdf fallback for files pdfplumber
chokes on) and Markdown/text files, splits them with the paragraph-aware
:mod:`app.rag.splitter`, then upserts the chunks into Chroma.

The CLI entrypoint (``python -m app.rag.ingestion``) walks the configured
``documents_dir`` and is idempotent: re-running it does not duplicate
chunks because each chunk is keyed by ``source + page + chunk index``.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable, Iterator
from pathlib import Path

from langchain_core.documents import Document

from app.config import get_settings
from app.rag.splitter import SplitterConfig, split_legal_text

logger = logging.getLogger(__name__)

SUPPORTED_TEXT_SUFFIXES = {".md", ".txt"}
PDF_SUFFIX = ".pdf"


# ---------------------------------------------------------------------------
# File readers
# ---------------------------------------------------------------------------
def _read_text_file(path: Path) -> Iterator[tuple[str, int | None]]:
    """Yield ``(text, page)`` for a Markdown/text file (single 'page')."""
    yield path.read_text(encoding="utf-8"), None


def _read_pdf(path: Path) -> Iterator[tuple[str, int | None]]:
    """Yield ``(page_text, page_number)`` for each PDF page.

    Uses pdfplumber for layout-aware extraction and falls back to pypdf
    if pdfplumber raises (some scanned/optimised PDFs trip it up).
    """
    try:
        import pdfplumber

        with pdfplumber.open(path) as pdf:
            for idx, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                if text.strip():
                    yield text, idx
        return
    except Exception as exc:  # pragma: no cover - fallback path
        logger.warning("pdfplumber failed on %s (%s); falling back to pypdf", path.name, exc)

    from pypdf import PdfReader

    reader = PdfReader(str(path))
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            yield text, idx


def _read_file(path: Path) -> Iterator[tuple[str, int | None]]:
    suffix = path.suffix.lower()
    if suffix == PDF_SUFFIX:
        yield from _read_pdf(path)
    elif suffix in SUPPORTED_TEXT_SUFFIXES:
        yield from _read_text_file(path)
    else:
        logger.debug("Skipping unsupported file: %s", path.name)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------
def load_and_split(
    path: Path,
    *,
    splitter_config: SplitterConfig | None = None,
) -> list[Document]:
    """Read a single file from disk and return its split chunks."""
    chunks: list[Document] = []
    for page_text, page_num in _read_file(path):
        chunks.extend(
            split_legal_text(
                page_text,
                source=path.name,
                page=page_num,
                config=splitter_config,
            )
        )
    # Stable chunk id within (source, page).
    seen: dict[tuple[str, int | None], int] = {}
    for doc in chunks:
        key = (doc.metadata["source"], doc.metadata.get("page"))
        idx = seen.get(key, 0)
        doc.metadata["chunk_id"] = idx
        seen[key] = idx + 1
    return chunks


def iter_documents(documents_dir: Path) -> Iterator[Path]:
    """Yield every supported document under ``documents_dir``."""
    for path in sorted(documents_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in SUPPORTED_TEXT_SUFFIXES | {PDF_SUFFIX}:
            yield path


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------
def _stable_chunk_id(doc: Document) -> str:
    src = doc.metadata.get("source", "?")
    page = doc.metadata.get("page", "x")
    cid = doc.metadata.get("chunk_id", 0)
    return f"{src}::p{page}::c{cid}"


def ingest_documents(
    docs: Iterable[Document],
    *,
    persist_dir: Path | None = None,
    collection_name: str | None = None,
) -> int:
    """Upsert chunks into Chroma. Returns number of chunks indexed."""
    # Lazy import: keeps unit tests fast and avoids pulling Chroma when
    # callers only need splitting.
    from langchain_chroma import Chroma

    from app.llm import get_embedding_model

    settings = get_settings()
    persist_dir = persist_dir or settings.chroma_persist_dir
    collection_name = collection_name or settings.chroma_collection
    persist_dir.mkdir(parents=True, exist_ok=True)

    store = Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_model(),
        persist_directory=str(persist_dir),
    )
    docs_list = list(docs)
    if not docs_list:
        return 0
    ids = [_stable_chunk_id(d) for d in docs_list]
    store.add_documents(docs_list, ids=ids)
    return len(docs_list)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _cli() -> None:  # pragma: no cover - thin wrapper
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Ingest documents into Chroma.")
    parser.add_argument(
        "--path",
        type=Path,
        default=settings.documents_dir,
        help="Directory containing PDFs / Markdown files to ingest.",
    )
    args = parser.parse_args()
    logging.basicConfig(level=settings.log_level)

    all_chunks: list[Document] = []
    for file_path in iter_documents(args.path):
        chunks = load_and_split(file_path)
        logger.info("%s -> %d chunks", file_path.name, len(chunks))
        all_chunks.extend(chunks)

    total = ingest_documents(all_chunks)
    logger.info("Indexed %d chunks into Chroma.", total)


if __name__ == "__main__":  # pragma: no cover
    _cli()

