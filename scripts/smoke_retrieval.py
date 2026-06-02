"""Quick smoke test for the populated Chroma store (run after ingestion)."""
from __future__ import annotations

from app.rag.retriever import get_vector_store


def main() -> None:
    store = get_vector_store()
    total = store._collection.count()  # noqa: SLF001  (smoke test only)
    print(f"chroma count: {total}")

    queries = [
        "Mennyi a társasági adó mértéke?",
        "Hogyan használható fel az elhatárolt veszteség?",
        "Mi a növekedési adóhitel feltétele?",
    ]
    for q in queries:
        print(f"\n=== Query: {q} ===")
        for doc in store.similarity_search(q, k=3):
            meta = doc.metadata
            src = (meta.get("source") or "?")[:60]
            section = meta.get("section") or "?"
            page = meta.get("page") or "?"
            print(f"[{src} | §{section} | p{page}]")
            print(doc.page_content[:240].replace("\n", " "))
            print("---")


if __name__ == "__main__":
    main()
