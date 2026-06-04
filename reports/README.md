# Reports

Sample outputs from running the evaluation and load-test runners against the
TAO compliance agent. Committed here so a reviewer can see real numbers
without re-running the multi-hour pipeline locally.

## Files

- `eval_aya8b.csv` — Functional evaluation over `data/eval/questions.json`
  (15 labelled Hungarian TAO questions). Generated with the mixed Ollama
  profile: main = `aya-expanse:8b`, fast + judge = `qwen3:0.6b`,
  `RAG_TOP_K=3`, `LLM_MAX_TOKENS=400`, `MAX_HALLUCINATION_RETRIES=0`.
  Total wall time: ~2 h 25 min on CPU-only (Intel Iris Xe iGPU, 16 GB RAM).
  Retrieval was pure dense Chroma similarity (the pre-BM25 baseline).
- `eval_aya8b_hybrid_smoke.csv` — Follow-up *smoke* evaluation over the
  6-question subset `data/eval/questions_smoke.json` (q01, q05, q07, q08,
  q14, q15 — one in-scope question per source PDF plus the calculation
  case and the off-topic refusal). Same model profile as above but with
  `RAG_RETRIEVAL_MODE=hybrid` (dense + BM25 fused by RRF) and the new
  classifier-driven `source_hint` Chroma metadata filter both active.
  Prompts at this point were already English-instruction / Hungarian-
  output. Total wall time: ~47 min. This run is the qualitative
  check on the post-baseline architectural changes; it is **not** a
  replacement for `eval_aya8b.csv` (sample size is too small to draw
  conclusions on the headline metrics).
- `load_test_dummy_n100_per_query.csv` — Per-query rows from a 100-query
  load test under the deterministic dummy provider.
- `load_test_dummy_n100_per_node.csv` — Per-node latency percentiles
  (count, mean, min, p50, p95, p99, max) from the same run.
- `load_test_dummy_n100_per_node.png` — Grouped bar chart of
  p50 / p95 / p99 per node from the same run.

The load test runs against the dummy provider deliberately — see
README §9.1 for the reasoning. The Chroma collection used by the dummy
run lives at `data/chroma_dummy/` (not committed); regenerate it with
`uv run python -m app.rag.ingestion` while `LLM_PROVIDER=dummy` and
`CHROMA_PERSIST_DIR=./data/chroma_dummy` are set.

## Reproducing

### Heavy evaluation (Ollama, ~2 h 25 min)

```powershell
$env:LLM_PROVIDER = "ollama"
$env:OLLAMA_MODEL = "aya-expanse:8b"
$env:OLLAMA_FAST_MODEL = "qwen3:0.6b"
$env:OLLAMA_JUDGE_MODEL = "qwen3:0.6b"
$env:RAG_TOP_K = "3"
$env:LLM_MAX_TOKENS = "400"
$env:MAX_HALLUCINATION_RETRIES = "0"
$env:OLLAMA_KEEP_ALIVE = "24h"

uv run python -m app.eval.runner --out reports/eval_aya8b.csv
```

### Hybrid smoke (Ollama, ~47 min)

```powershell
# Same env block as above, plus:
$env:RAG_RETRIEVAL_MODE = "hybrid"
$env:RAG_BM25_PATH = "./data/chroma/bm25.pkl"

# Make sure the BM25 index exists (built alongside Chroma at ingest time,
# or rebuild it on its own without re-embedding):
uv run python -m app.rag.ingestion --bm25-only

uv run python -m app.eval.runner `
    --dataset data/eval/questions_smoke.json `
    --out reports/eval_aya8b_hybrid_smoke.csv
```

### Load test (dummy provider, ~5 s)

```powershell
$env:LLM_PROVIDER = "dummy"
$env:CHROMA_PERSIST_DIR = ".\data\chroma_dummy"
uv run python -m app.rag.ingestion           # one-off, builds 64-dim index
uv run python -m app.load_test.runner --n 100 --concurrency 5 --out reports
```

See the top-level README for the summary of results, bottleneck analysis,
and optimisation suggestions.
