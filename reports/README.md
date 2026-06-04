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

## Reproducing

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
uv run python -m app.load_test.runner --n 50 --concurrency 1 --out reports
```

See the top-level README for the summary of results, bottleneck analysis,
and optimisation suggestions.
