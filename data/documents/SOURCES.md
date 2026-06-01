# Source documents

This folder contains publicly available Hungarian tax guidance PDFs from
the National Tax and Customs Administration (NAV).

The PDFs are versioned in the repository (small total size, takes
precedence over reproducibility for an evaluator who clones the repo).

## Provenance

All documents originate from the official NAV information leaflets page:

- <https://nav.gov.hu/ugyfeliranytu/nezzen-utana/inf_fuz/2025>

Domain focus for this prototype: **corporate income tax (TAO)**.

## How to refresh

If you want a newer version of these documents:

1. Open the NAV URL above.
2. Download the relevant PDFs into this folder.
3. Run `uv run python -m app.rag.ingestion --path data/documents` to
   re-index Chroma. Ingestion is idempotent on the same chunk IDs.
