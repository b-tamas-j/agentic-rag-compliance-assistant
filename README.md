# Agentic RAG Compliance Assistant

> Hungarian tax & accounting compliance chatbot built with **LangGraph**,
> **Ollama** and **ChromaDB**. Built as a PwC Medior AI Engineer take-home
> project.

---

## 1. Problem & objective

*To be filled in.* Short description of the domain (Hungarian VAT / corporate
income tax compliance Q&A for advisors), the user need and why an *agentic*
RAG approach is a good fit.

## 2. Architecture overview

*To be filled in.* Diagram + reasoning behind:

- Why Ollama + which model (trade-offs).
- Why ChromaDB.
- Why a dedicated RAG subgraph.
- The 5+ node main workflow with conditional routing & hallucination-check
  retry loop.
- The two tools (RAG + a non-retrieval tool).

## 3. Project layout

```
app/
  agent/      LangGraph workflow, nodes, state, tools
  rag/        ingestion, retriever, RAG subgraph
  llm/        LLM / embedding provider abstraction
  config.py   typed settings (pydantic)
ui/           Streamlit UI
data/
  documents/  source PDFs (git-ignored)
  chroma/     persistent vector store (git-ignored)
eval/         eval dataset + runner + load test
tests/        pytest suite
docs/         architecture diagram & notes
Dockerfile, docker-compose.yml
```

## 4. Installation & run

### Option A - Docker (recommended)

```powershell
git clone <repo-url>
cd agentic-rag-compliance-assistant
Copy-Item .env.example .env
docker compose up --build
```

Then open <http://localhost:8501>.

### Option B - Local Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
streamlit run ui/streamlit_app.py
```

## 5. Evaluation results

*To be filled in once `feat/evaluation` lands.*

## 6. Load test & bottleneck analysis

*To be filled in once `feat/load-test` lands.*

## 7. Future work

*To be filled in.*
