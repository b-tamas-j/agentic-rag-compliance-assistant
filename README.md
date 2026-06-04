# Agentic RAG Compliance Assistant

Hungarian corporate income tax (TAO) Q&A assistant built around a
**LangGraph** agent on top of a **ChromaDB** vector store with local
**Ollama** models. The agent is designed for the compliance use case:
every answer must cite the underlying NAV source, every numeric example
goes through a deterministic calculator, and a separate judge model
verifies that no claim is invented before the answer reaches the user.

The project is a self-contained demo — a small, opinionated codebase
that can be cloned, brought up with `docker compose` or `uv run`, and
explored end-to-end (chat UI, evaluation harness, load test).

---

## Table of contents

1. [Why agentic RAG here](#1-why-agentic-rag-here)
2. [Architecture](#2-architecture)
3. [Project layout](#3-project-layout)
4. [Quick start](#4-quick-start)
5. [Configuration](#5-configuration)
6. [Data ingestion](#6-data-ingestion)
7. [Running the chat UI](#7-running-the-chat-ui)
8. [Evaluation](#8-evaluation) — including [sample results](#81-results-from-a-sample-run-aya-expanse8b-mixed-profile)
9. [Load test](#9-load-test) — including a [sample dummy run, N=100](#91-sample-run-dummy-provider-n100)
10. [Testing & CI](#10-testing--ci)
11. [Design decisions and trade-offs](#11-design-decisions-and-trade-offs)
12. [Future work](#12-future-work)

---

## 1. Why agentic RAG here

Plain RAG (retrieve-then-answer) is brittle on a compliance domain for a
few concrete reasons:

* **Numeric questions** like *"Mennyi a TAO 10 000 000 Ft adóalapra?"*
  cannot be answered by the LLM alone without risking hallucinated
  arithmetic. A dedicated calculator tool keeps the maths deterministic.
* **Cited paragraphs** (`19. §`, `17/A. §`) must actually exist in the
  source corpus, otherwise the answer is misleading. A validator tool
  checks every citation against ChromaDB before the answer is shown.
* **Hallucination control** matters more than fluency. A separate judge
  model re-reads the draft against the retrieved chunks and triggers a
  bounded retry if any claim is unsupported.
* **Scope discipline** — the assistant should politely refuse anything
  outside Hungarian corporate income tax instead of confidently
  hallucinating about, say, VAT.

These constraints justify the extra cost of an agent over a single-shot
RAG chain: each requirement maps to a dedicated node in the graph.

---

## 2. Architecture

### Main agent graph

```mermaid
flowchart TD
    START([User query]) --> CLASS[classify_query<br/>fast model]
    CLASS -->|tao| DECOMP[query_decomposer<br/>main model]
    CLASS -->|off_topic| OFF[off_topic_handler]
    DECOMP --> RET[retrieve_documents<br/>RAG subgraph per sub-query]
    RET --> TOOLS[tool_executor<br/>tao_calculator + legal_reference_validator]
    TOOLS --> ANS[answer_generator<br/>main model + sources]
    ANS --> CHECK{hallucination_checker<br/>judge model}
    CHECK -->|grounded| END([Final answer])
    CHECK -->|not grounded<br/>and retries left| ANS
    OFF --> END
```

The main graph has **seven nodes** and **two conditional edges**:

| Node | Model | Responsibility |
|---|---|---|
| `classify_query` | fast (`aya-expanse`) | TAO vs off-topic, structured output |
| `query_decomposer` | main (`qwen2.5`) | Split into 1–3 retrievable sub-queries |
| `retrieve_documents` | — | Runs the RAG subgraph per sub-query, dedupes results |
| `tool_executor` | — | Deterministic: fires `tao_calculator` on HUF amounts, `legal_reference_validator` on every `§` reference |
| `answer_generator` | main (`qwen2.5`) | Drafts the answer from retrieved chunks + tool outputs |
| `hallucination_checker` | judge (`mistral-nemo`) | Structured groundedness verdict; loops back into `answer_generator` up to `MAX_HALLUCINATION_RETRIES` |
| `off_topic_handler` | — | Polite Hungarian refusal |

### RAG subgraph

`retrieve_documents` delegates to an internal three-node graph:

```mermaid
flowchart LR
    Q([sub-query]) --> QT[query_transform]
    QT --> R[retrieve]
    R --> G[grade_documents]
    G --> OUT([relevant_docs])
```

* **query_transform** rewrites the sub-query into a retrieval-friendly
  form (drops question particles, normalises numbers).
* **retrieve** does a top-K similarity search against ChromaDB
  (`bge-m3` embeddings).
* **grade_documents** drops chunks that are obviously off-topic for the
  sub-query. With the dummy provider this is a deterministic keyword
  overlap; with Ollama it is an LLM call with structured output.

### Why these components

| Choice | Reason |
|---|---|
| **LangGraph** | Explicit state-machine fits the compliance requirements (retry budget, conditional branches, audit-friendly per-node logging). Cleaner than a generic ReAct loop for a pipeline with deterministic checkpoints. |
| **Ollama** | Lets the whole stack run locally without external API keys; reviewer can reproduce results offline. Model role split (main / fast / judge / embedding) is configurable per env. |
| **ChromaDB (langchain-chroma)** | Embedded, file-backed, no extra service to operate; appropriate for a corpus of this size (~380 chunks). The `data/chroma/` directory is gitignored so each clone re-indexes from the versioned PDFs. |
| **`pdfplumber` + `pypdf` fallback** | Hungarian NAV PDFs occasionally trip `pdfplumber` on tables / scanned pages; `pypdf` is the safe fallback. |
| **`bge-m3`** | Strong multilingual embedding model that handles Hungarian well; 1024-dim, fits ChromaDB without quantisation. |
| **Separate judge model** | Using the same model as both author and judge biases the verdict ("anchoring"). A different model family (`mistral-nemo`) gives a more honest groundedness signal. |
| **Pydantic-settings + `@lru_cache`** | Centralised, typed config; the cached `get_settings()` makes tests trivial to monkey-patch via env vars. |
| **`uv`** | Fast, reproducible installs from `uv.lock`; same lockfile drives local dev, CI and Docker. |

For a longer write-up see [`docs/architecture.md`](docs/architecture.md).

---

## 3. Project layout

```
app/
  agent/             LangGraph main workflow
    state.py         AgentState TypedDict
    nodes.py         The seven node implementations
    graph.py         build_agent_graph()
    tools/           tao_calculator + legal_reference_validator
  rag/               Ingestion + retriever + RAG subgraph
    splitter.py      Paragraph-aware (§) splitter
    ingestion.py     CLI: load PDFs -> Chroma
    retriever.py     get_vector_store() helper
    subgraph.py      query_transform -> retrieve -> grade_documents
  llm/               Provider abstraction (Ollama + dummy fallback)
  ui/                Streamlit chat UI
  eval/              Labelled dataset, metrics, LLM-judge, CLI runner
  load_test/         Async runner, per-node tracer, matplotlib chart
  config.py          Settings (pydantic-settings)
data/
  documents/         NAV PDF leaflets (versioned in the repo)
  chroma/            Persistent vector store (gitignored)
  eval/              Labelled eval dataset (questions.json)
reports/             Eval + load-test outputs (gitignored)
tests/               Pytest suite (60 tests)
docs/                Architecture notes
Dockerfile, docker-compose.yml
```

---

## 4. Quick start

### Option A — Docker

```powershell
git clone <repo-url>
cd agentic-rag-compliance-assistant
Copy-Item .env.example .env
docker compose up --build
```

The compose file starts two services: `ollama` (model server) and the
app (Streamlit on <http://localhost:8501>). On first run, pull the
models inside the `ollama` container:

```powershell
docker compose exec ollama ollama pull qwen2.5:14b-instruct
docker compose exec ollama ollama pull bge-m3
docker compose exec ollama ollama pull mistral-nemo:12b
docker compose exec ollama ollama pull aya-expanse:8b
```

Then ingest the PDFs once:

```powershell
docker compose exec app uv run python -m app.rag.ingestion --path data/documents
```

### Option B — Local with `uv`

```powershell
git clone <repo-url>
cd agentic-rag-compliance-assistant
Copy-Item .env.example .env
uv sync
uv run python -m app.rag.ingestion --path data/documents
uv run streamlit run app/ui/streamlit_app.py
```

A local Ollama install at `http://localhost:11434` is expected; pull the
same four models with `ollama pull`.

### Option C — Offline / dummy provider

For a quick smoke test without Ollama, set `LLM_PROVIDER=dummy` in
`.env`. All LLM-backed nodes fall back to deterministic behaviour, so
the graph runs end-to-end (the answers are placeholders, but every node
fires and every test passes).

---

## 5. Configuration

All settings live in [`app/config.py`](app/config.py) and can be
overridden via `.env` or process env vars. The most relevant ones:

| Key | Default | Purpose |
|---|---|---|
| `LLM_PROVIDER` | `dummy` | `ollama` for real models, `dummy` for offline determinism |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Where the Ollama API lives (compose sets it to `http://ollama:11434`) |
| `OLLAMA_MODEL` | `qwen2.5:14b-instruct` | Main model (decomposer + answer generator) |
| `OLLAMA_FAST_MODEL` | `aya-expanse:8b` | Cheap classifier model |
| `OLLAMA_JUDGE_MODEL` | `mistral-nemo:12b` | Separate model for groundedness check |
| `OLLAMA_EMBEDDING_MODEL` | `bge-m3` | Multilingual embeddings |
| `RAG_TOP_K` | `5` | Top-K retrieved chunks per sub-query |
| `RAG_RETRIEVAL_MODE` | `dense` | `dense` (Chroma similarity), `bm25` (keyword), or `hybrid` (RRF of both) |
| `RAG_BM25_PATH` | `./data/chroma/bm25.pkl` | Where the BM25 index pickle lives |
| `RAG_RRF_K` | `60` | Reciprocal Rank Fusion constant used in `hybrid` mode |
| `MAX_HALLUCINATION_RETRIES` | `2` | Cap on the grounded-answer loop |
| `CHROMA_PERSIST_DIR` | `./data/chroma` | Where ChromaDB persists |

See [`.env.example`](.env.example) for the full list with comments.

---

## 6. Data ingestion

```powershell
uv run python -m app.rag.ingestion --path data/documents
```

The CLI walks the directory, extracts text with `pdfplumber` (or
`pypdf` if pdfplumber returns an empty page), splits each document on
Hungarian `§` paragraph boundaries with a fall-back recursive splitter
inside long paragraphs, embeds the chunks with `bge-m3`, and upserts
them into ChromaDB with stable IDs (`{source}::p{page}::c{chunk_id}`)
so re-running is idempotent. Provenance is preserved in
`metadata.section` so the validator tool and the UI can surface it.

The same run also builds the BM25 keyword index over those chunks and
pickles it to `RAG_BM25_PATH` (default `./data/chroma/bm25.pkl`, ~870
KB on the bundled corpus). If the dense store is already populated and
you only need to (re)build the keyword index — handy when iterating
on tokenization without paying the embedding cost — pass
`--bm25-only`:

```powershell
uv run python -m app.rag.ingestion --bm25-only
```

---

## 7. Running the chat UI

```powershell
uv run streamlit run app/ui/streamlit_app.py
```

The UI streams each node as it executes (Hungarian step labels in a
`st.status` panel), then renders the final answer plus dedicated cards
for `tao_calculator` outputs and `legal_reference_validator` verdicts.
Every retrieved source chunk is available in an expander with its file
name, section and page number, so the user can audit the answer in two
clicks. Multi-turn chat history is kept in `st.session_state`, and the
graph is compiled once per process with an in-memory `MemorySaver`
checkpointer so each session has its own `thread_id`.

---

## 8. Evaluation

```powershell
uv run python -m app.eval.runner
uv run python -m app.eval.runner --no-judge          # skip LLM-judge
uv run python -m app.eval.runner --limit 5 --out reports/quick.csv
```

The dataset (`data/eval/questions.json`) holds **15 labelled Hungarian
TAO questions** with the expected `§` references and a few mandatory
key terms; one is intentionally off-topic to catch over-eager routing.
For each question the runner records:

* `category_correct` — TAO vs off-topic classifier accuracy
* `recall_at_k` — fraction of expected `§` sections present in any
  retrieved chunk
* `citation_accuracy` — fraction of `§` references in the answer that
  are backed by a retrieved chunk (penalises invented citations)
* `terms_coverage` — fraction of `expected_terms` present in the answer
* `grounded` — final-state grounded flag (after retries)
* `latency_s` — total invocation time
* `judge_groundedness / judge_relevance / judge_completeness` — 1–5
  scores from `mistral-nemo` for the in-scope questions

Outputs land in `reports/eval.csv` plus a stdout summary.

### 8.1 Results from a sample run (`aya-expanse:8b` mixed profile)

> **Caveat (post-run change).** The numbers below were measured with
> the *pre-existing* retrieval setup: dense-only Chroma similarity
> search, no metadata filter. The BM25 keyword index, the
> `RAG_RETRIEVAL_MODE=dense|bm25|hybrid` switch, and the
> `source_hint`-based source filter (see §11) were implemented
> *after* this evaluation run and are **not** reflected in the figures
> below. Re-running the eval under those new modes is left as a
> follow-up because a single full run takes ~2.5 h on this box.

I ran the full 15-question evaluation against Ollama with the mixed
profile (main = `aya-expanse:8b`, fast + judge = `qwen3:0.6b`,
`RAG_TOP_K=3`, `LLM_MAX_TOKENS=400`, `MAX_HALLUCINATION_RETRIES=0`).
Wall time was ~2 h 25 min on a CPU-only laptop (Intel Iris Xe iGPU,
16 GB RAM). The raw output is committed at
[`reports/eval_aya8b.csv`](reports/eval_aya8b.csv).

| Metric | Value | Read |
|---|---|---|
| `category_correct` | **0.867** (13/15) | Classifier solid, two edge cases below |
| `recall_at_k` | **0.20** | Top-3 retrieval misses most expected `§` references |
| `citation_accuracy` | **0.80** | Cited `§`s are largely real — hallucinated citations are the exception |
| `terms_coverage` | **0.57** | Keyword overlap is mid; LLM-judge sees more semantic match |
| `judge_groundedness` | **4.07 / 5** | Mostly grounded per the `qwen3:0.6b` judge |
| `judge_relevance` | **3.50 / 5** | Answers address the question but drift |
| `judge_completeness` | **4.14 / 5** | Coverage of the expected aspects is fine |
| mean `latency_s` | **544 s ≈ 9 min** | Dominated by `answer_generator` on CPU |

**What works (architecture):** every one of the 15 questions ran end-to-end
without exceptions across the 2.5 h. Citations stay anchored to real
chunks (`citation_accuracy=0.80`), the off-topic refusal mechanism
fires (q04 returned the polite refusal template in 5 s), and the LLM
judge scores the in-scope answers above 4/5 on groundedness and
completeness. The graph, state, tool routing, judge, and metrics
pipeline are all production-shaped.

**What struggles (model and retrieval):**

* `recall_at_k=0.20` — only q05 (NAHI) and q11 (fejlesztési
  adókedvezmény) hit the expected `§` in the top-3 retrieved chunks.
  General-vocabulary queries like *"Mennyi a társasági adó mértéke?"*
  do not surface `19. §`.
* Content-level mistakes from the 8B model on CPU: q01 invents a
  *330 E Ft × 9% ≈ 30 E Ft* example, q02 writes *42 százalék* instead
  of 50, q06 writes *7,5 százalék* instead of 80, q08 hallucinates a
  10 MFt threshold. The answers are recognisably Hungarian but the
  legal phrasing is wobbly ("egyedik élményeket és kötelendeket").
* q04 (jövedelem-minimum) was wrongly classified `off_topic` because
  none of my hard-coded TAO keywords are present in the query —
  the keyword safety net only catches obvious matches.
* q15 (brassói aprópecsenye, off-topic) was wrongly classified `tao`.
  The classifier prompt is too forgiving for clearly out-of-scope
  Hungarian text.

**Bottlenecks:**

1. **Retrieval recall.** Top-3 with a single dense embedding leaves
   most expected sections behind. This is the dominant quality
   bottleneck.
2. **Generation latency.** `answer_generator` on `aya-expanse:8b`
   takes ~4 min per call on CPU, and the pipeline issues 5–7 LLM
   calls per question. End-to-end stays in the multi-minute range
   no matter what the other nodes do.
3. **Classifier coverage.** The keyword-override safety net works
   one direction (off_topic → tao) but only on literal matches, so
   the edge cases above slip through.

**Concrete optimisations (in roughly the order I would try them):**

* Raise `RAG_TOP_K` from 3 to 6–8 and tighten the grader prompt — I
  expect this to lift `recall_at_k` past 0.5 at ~20 % more latency.
* Add a hybrid retrieval step (BM25 over `§` numbers and section
  titles, merged with dense) — section-pointed queries should hit
  recall ~1.0.
* Move to GPU or a hosted provider for the main model. The
  architecture is provider-agnostic (see `app/llm/provider.py`); a
  swap drops per-question latency from minutes to seconds and lifts
  answer fluency considerably.
* Stream tokens from `answer_generator` to the UI — total latency
  unchanged, perceived UX much better.
* Replace the keyword classifier with an embedding-similarity check
  against a few seed TAO sentences — should remove both q04 and q15
  failures.

---

## 9. Load test

```powershell
uv run python -m app.load_test.runner --n 50 --concurrency 5
uv run python -m app.load_test.runner --n 100 --concurrency 1 --no-chart
```

The harness drives a realistic query mix (factual lookups, calculations
that trigger `tao_calculator`, explicit `§` citations that trigger
`legal_reference_validator`, plus two off-topic queries) through
`graph.ainvoke` with an `asyncio.Semaphore` for bounded concurrency. A
small `trace_node_timings()` context manager monkey-patches every node
to record per-call durations, exposing **p50 / p95 / p99** per node.

Outputs:

* `reports/load_test_per_query.csv` — one row per query
* `reports/load_test_per_node.csv` — per-node percentile table
* `reports/load_test_per_node.png` — grouped bar chart (p50 / p95 / p99
  per node)
* Stdout: end-to-end and per-node summary tables

The chart makes it easy to spot the dominant nodes (typically
`answer_generator` and `retrieve_documents`) and any long-tail
behaviour under concurrency.

### 9.1 Sample run (dummy provider, N=100)

I ran the load test under `LLM_PROVIDER=dummy` against a separate
Chroma collection populated with the dummy embedder, so the per-node
timings reflect pure architecture cost — graph dispatch,
deterministic node bodies, and Chroma similarity search — with the
LLM cost zeroed out. Raw outputs are committed at
[`reports/load_test_dummy_n100_per_query.csv`](reports/load_test_dummy_n100_per_query.csv),
[`reports/load_test_dummy_n100_per_node.csv`](reports/load_test_dummy_n100_per_node.csv),
and [`reports/load_test_dummy_n100_per_node.png`](reports/load_test_dummy_n100_per_node.png).

End-to-end (100 queries, concurrency 5, ~5 s wall time):

| | p50 | p95 | p99 | max |
|---|---|---|---|---|
| latency (s) | 0.057 | 0.286 | 0.920 | 0.929 |

Per-node (only `retrieve_documents` registers under the dummy):

| node | calls | p50 | p95 | p99 | max |
|---|---|---|---|---|---|
| classify_query | 100 | 0.000 | 0.000 | 0.002 | 0.002 |
| query_decomposer | 84 | 0.000 | 0.000 | 0.000 | 0.000 |
| retrieve_documents | 84 | **0.022** | **0.889** | **0.890** | 0.905 |
| tool_executor | 79 | 0.000 | 0.027 | 0.032 | 0.034 |
| answer_generator | 79 | 0.000 | 0.000 | 0.000 | 0.0001 |
| hallucination_checker | 79 | 0.000 | 0.000 | 0.000 | 0.000 |
| off_topic_handler | 16 | 0.000 | 0.000 | 0.000 | 0.000 |

**Read.** With the LLM cost stripped out, `retrieve_documents` is the
only node that takes measurable time, and its p95 (~0.9 s) is what
drives the end-to-end p99 (~0.92 s). Everything else — classifier,
decomposer, tool executor, judge — is at most milliseconds. The graph
wiring is not the bottleneck; the dense vector search is.

**What this changes about §8.1.** The Ollama numbers in §8.1 showed
~9 min per question. The dummy load test bounds the contribution from
the non-LLM parts of the pipeline at ~1 s end-to-end at p99. The rest
— so the entire 8 min 59 s gap — comes from LLM inference inside
`answer_generator`, `query_decomposer`, the grader, and the judge.
This is why every optimisation in §8.1 that doesn't touch the LLM
(top-K bump, hybrid retrieval, streaming UI) buys at most seconds,
while any change that *does* touch the LLM (GPU, hosted provider) is
worth minutes per question.

**Caveat about the dummy.** 5 of the 100 queries failed because the
dummy provider exercises a slightly different code path in the
Chroma client during shutdown (`'RustBindingsAPI' object has no
attribute 'bindings'` etc.); the surviving 95 are enough to read the
per-node distribution. The 16 off-topic queries skip retrieval, so
that node only has 84 samples instead of 100. Neither caveat affects
the headline conclusion.

---

## 10. Testing & CI

```powershell
uv run pytest -q
```

The suite (60 tests) covers:

* Configuration & provider factory (`test_config.py`, `test_llm_provider.py`)
* RAG splitter, ingestion idempotency, retriever round-trip, subgraph
  (`test_rag.py`)
* BM25 keyword index — build, search, source filter, persist/load,
  cache invalidation (`test_bm25.py`)
* Source-hint classification + retrieval-mode dispatch (dense / bm25 /
  hybrid) + Reciprocal Rank Fusion (`test_source_filter.py`)
* Tools — calculator rate / loss cap, validator parsing & lookup
  (`test_tools.py`)
* Agent graph — classifier, off-topic short-circuit, tool firing,
  grounded loop, memory checkpoint (`test_agent.py`)
* Eval — metric correctness, dataset shape, runner end-to-end on a
  fixture (`test_eval.py`)
* Load test — percentiles, tracer patch/restore, runner with chart
  (`test_load_test.py`)

All tests run against the dummy provider (no Ollama required), so the
suite is fast (< 5 s) and deterministic for CI.

---

## 11. Design decisions and trade-offs

* **Explicit `StateGraph` over a ReAct agent.** The retry loop with a
  bounded counter, the deterministic tool firing on regex-detected
  amounts / citations, and the off-topic short-circuit are all easier
  to reason about and test as edges in a graph. ReAct would have hidden
  these inside an LLM-driven loop.
* **Two-stage pipeline (sub-graph + main graph) instead of one
  monolithic graph.** Lets the RAG layer be reused in eval and tested
  independently of the agent (`test_rag.py`).
* **Deterministic tool fallbacks.** `tao_calculator` and
  `legal_reference_validator` do not call an LLM. This is the right
  default for a compliance domain: numbers and citations should never
  be hallucinated.
* **Judge model ≠ author model.** Using `mistral-nemo` to grade
  `qwen2.5` reduces self-confirmation bias compared with using the
  same model for both roles.
* **Dummy provider in CI.** Every LLM-backed node has a deterministic
  fallback when `LLM_PROVIDER=dummy`, so the full graph runs offline.
  This is what lets the 60-test suite finish in seconds.
* **Tools as one-module-per-file subpackage.** Keeps the boundary
  between deterministic and LLM-driven logic visible and makes it
  easy to bind them to `chat.bind_tools` later if we want to mix in
  an LLM-orchestrated tool-calling node.
* **Versioned PDFs in the repo.** ~2 MB total. Trades a bit of repo
  bloat for full reproducibility: anyone can clone and run the
  pipeline with no manual download.
* **Dense + BM25 + source filter, all opt-in.** The retrieval layer
  supports three modes via `RAG_RETRIEVAL_MODE`:
  * `dense` (default, backward-compatible) — pure Chroma similarity
    search over `bge-m3` embeddings. This is what the §8.1 numbers
    were measured with.
  * `bm25` — pure keyword search via `rank_bm25` over a pickled
    `BM25Okapi` index built at ingestion time
    (`data/chroma/bm25.pkl`). Useful when the query is dominated by
    rare legal tokens (paragraph numbers, proper nouns) where the
    embedding may not place the right chunk near the query in vector
    space.
  * `hybrid` — runs both, then combines the rankings with Reciprocal
    Rank Fusion (`score = Σ 1 / (k + rank)`, default `k=60`). Pulls a
    wider pool (`2·top_k`) from each side before fusing so RRF has
    room to promote chunks one side under-ranked.

  On top of that, the classifier now emits a coarse `source_hint`
  (`nonprofit` / `calculation` / `offering` / `credit` / `general`)
  which the retrieve node applies as a Chroma `where={"source":
  {"$in": [...]}}` filter (and as a post-score filter for BM25). This
  is the cheapest possible "metadata routing" — no extra LLM call, no
  extra embedding cost — and it keeps off-source chunks out of the
  context window for narrow questions. Everything degrades gracefully:
  an unknown hint, a missing BM25 pickle, or a `general` verdict all
  fall back to the unfiltered dense path. The full eval has **not**
  been re-run under the new modes (~2.5 h on this box); doing so is
  the natural next quantitative step.

---

## 12. Future work

* **Per-domain expansion.** Today the assistant is single-domain
  (TAO). The same graph would extend to ÁFA / KIVA / járulékok with an
  additional classifier branch and a per-domain retriever (separate
  ChromaDB collection).
* **CRAG-style web fallback.** If `grade_documents` returns nothing
  relevant, fall back to a web search (e.g. NAV.hu) before answering
  — useful for very recent rule changes.
* **Token-level streaming in the UI.** Currently the answer is
  rendered as a whole after `hallucination_checker` returns; streaming
  the draft would improve perceived latency.
* **Postgres-backed checkpointer.** `MemorySaver` is process-local;
  swapping in `PostgresSaver` would let conversation state survive a
  restart and scale beyond a single Streamlit worker.
* **CI workflow.** A GitHub Actions workflow running `uv sync` +
  `pytest -q` + the eval runner with `--no-judge` would catch
  regressions automatically.
