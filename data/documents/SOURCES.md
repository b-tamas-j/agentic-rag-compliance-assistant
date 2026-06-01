# Source documents

This folder contains publicly available Hungarian tax guidance PDFs from
the National Tax and Customs Administration (NAV).

The PDFs are versioned in the repository (~2 MB total). Reproducibility
matters more than binary bloat for a take-home project: a reviewer can
clone the repo and run the pipeline without any manual download step.

## Provenance

All documents originate from the official NAV information leaflets page:

- <https://nav.gov.hu/ugyfeliranytu/nezzen-utana/inf_fuz/2025>

## Document set (TAO focus)

| # | File | Topic |
|---|---|---|
| 13 | `13+A+nonprofit+szervezetek+adózása+2025.01.27.pdf` | Taxation of non-profit organizations |
| 41 | `41 A társasági adó legfontosabb szabályai 2025.09.01.pdf` | Core rules of corporate income tax (TAO) |
| 55 | `55 Tao-felajánlás 2025.07.21.pdf` | TAO offering (tax allocation to eligible recipients) |
| 93 | `93 Növekedési adóhitel 2025.01.17.pdf` | Growth tax credit |

Each file name keeps the NAV leaflet number prefix and the publication
date so provenance is traceable at a glance.

## How to refresh

If you want a newer version of these documents:

1. Open the NAV URL above.
2. Replace the PDFs in this folder (keep the leaflet-number prefix so
   citations stay stable).
3. Run `uv run python -m app.rag.ingestion --path data/documents` to
   re-index Chroma. Ingestion is idempotent on the same chunk IDs;
   replacing a file will refresh its chunks in the next ingestion.
