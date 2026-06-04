"""BM25 keyword index for the compliance corpus.

Complements the dense Chroma retriever with classic lexical search so the
agent can find chunks that share rare legal tokens (paragraph numbers,
proper nouns, specific Hungarian terms) with the query even when the
embedding model places them far apart in vector space.

The index is small enough (a few hundred chunks) to live entirely in
memory and to be persisted as a single pickle. It is rebuilt by
:mod:`app.rag.ingestion` whenever the Chroma store is rebuilt.
"""

from __future__ import annotations

import pickle
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from langchain_core.documents import Document
from rank_bm25 import BM25Okapi


# A loose tokenizer: lowercase + Unicode-NFKC + split on anything that is
# not a letter, digit, or paragraph marker. Keeps Hungarian accents intact
# (they carry meaning in this corpus) and treats "19.§" / "19. §" alike.
_TOKEN_RE = re.compile(r"[\w§]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return _TOKEN_RE.findall(text)


@dataclass
class BM25Index:
    """In-memory BM25 index plus the original chunks it was built from."""

    bm25: BM25Okapi
    documents: list[Document]

    def search(
        self,
        query: str,
        *,
        k: int = 5,
        source_filter: list[str] | None = None,
    ) -> list[Document]:
        """Return the top-``k`` documents matching ``query``.

        ``source_filter`` (if given) restricts the candidates to chunks
        whose ``metadata['source']`` is in the list. The filter is
        applied AFTER scoring so the BM25 statistics still reflect the
        full corpus, which is what we want for IDF.
        """
        if not query.strip() or not self.documents:
            return []
        tokens = _tokenize(query)
        if not tokens:
            return []
        # If none of the query tokens appear anywhere in the corpus,
        # BM25 has nothing to say -> return empty rather than emitting
        # arbitrary low-scoring chunks.
        if not any(tok in self.bm25.idf for tok in tokens):
            return []
        scores = self.bm25.get_scores(tokens)
        # Sort indexes by descending score. NOTE: on small corpora BM25's
        # IDF term ``log((N-n+0.5)/(n+0.5))`` can be negative even for
        # documents that contain query tokens. We therefore rank purely
        # by score and let the top-k cutoff filter, instead of dropping
        # everything with ``score <= 0`` (which would empty the result
        # set on tiny corpora and during early development).
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out: list[Document] = []
        for idx in ranked:
            doc = self.documents[idx]
            if source_filter and doc.metadata.get("source") not in source_filter:
                continue
            out.append(doc)
            if len(out) >= k:
                break
        return out


def build_bm25_index(documents: list[Document]) -> BM25Index:
    """Build an in-memory BM25 index over ``documents``."""
    corpus = [_tokenize(d.page_content) for d in documents]
    # rank_bm25 doesn't like an empty corpus, but an empty index is a
    # legitimate state (e.g. before first ingestion), so guard the caller
    # rather than the constructor.
    if not corpus:
        raise ValueError("Cannot build a BM25 index from an empty document list.")
    return BM25Index(bm25=BM25Okapi(corpus), documents=list(documents))


def persist_bm25_index(index: BM25Index, path: Path) -> None:
    """Pickle ``index`` to ``path`` (creating parent directories)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump(index, fh)


def load_bm25_index(path: Path) -> BM25Index | None:
    """Load a previously persisted index, or ``None`` if the file is absent."""
    if not path.exists():
        return None
    with path.open("rb") as fh:
        return pickle.load(fh)


# Module-level cache so repeated retrievals don't re-read the pickle.
_cached_index: tuple[Path, float, BM25Index] | None = None


def get_bm25_index(path: Path) -> BM25Index | None:
    """Return the cached BM25 index for ``path``, reloading on mtime change."""
    global _cached_index
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    if _cached_index and _cached_index[0] == path and _cached_index[1] == mtime:
        return _cached_index[2]
    index = load_bm25_index(path)
    if index is not None:
        _cached_index = (path, mtime, index)
    return index


def reset_bm25_cache() -> None:
    """Drop the in-memory BM25 cache (test helper)."""
    global _cached_index
    _cached_index = None
