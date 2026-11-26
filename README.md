# DevDocs AI v2

[![CI](https://github.com/AnasSayed27/DevDocs_AI-v2/actions/workflows/ci.yml/badge.svg)](https://github.com/AnasSayed27/DevDocs_AI-v2/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/pytest-8%20passed-brightgreen)
![OWASP](https://img.shields.io/badge/OWASP-LLM01%20Delimited-blue)
![Auth](https://img.shields.io/badge/Auth-X--API--Key%20(HMAC%20constant--time)-blue)

A custom, zero-vector-DB Retrieval-Augmented Generation (RAG) microservice built from scratch with pure NumPy, Pydantic v2, and FastAPI.

- **Architecture & Trade-offs:** See **[`docs/DECISIONS.md`](docs/DECISIONS.md)**
- **Test & Lint Status (verified locally):** `pytest` **8 passed** (`tests/`), `ruff check .` passed, `mypy .` passed (12 source files)
- **Security Controls (`src/security.py`):**
  - **OWASP LLM01 Prompt Injection Delimiters:** `wrap_prompt` / `wrap_user_input` escapes XML/HTML characters (`html.escape(..., quote=False)`) and wraps untrusted queries in `<user_input>...</user_input>` tags.
  - **`X-API-Key` Constant-Time Verification:** `verify_api_key` validates keys via `secrets.compare_digest` and raises HTTP `401` on mismatch *(implemented & unit-tested in `src/security.py`; not yet wired as a FastAPI dependency on `POST /query` in `src/api.py`)*.

---

## Architecture Flow

```text
Client Request (POST /query, QueryRequest)
  │
  ├──► [Security — src/security.py]
  │      • verify_api_key (X-API-Key via secrets.compare_digest)
  │      • wrap_user_input (OWASP LLM01 <user_input> delimiter + escaping)
  │
  ├──► [Retrieval — src/retriever.py]  (currently _mock_retrieve stub in src/api.py)
  │      • MarkdownChunker (header-aware ## splitting)
  │      • NumpyRetriever (pure NumPy cosine similarity, zero-vector guard, [-1, 1] clip)
  │
  └──► [Grounded Generation — src/generator.py]
         ├── max(similarity_score) < 0.3 (SIMILARITY_THRESHOLD)
         │     └──► Abstain: "I do not have enough context to answer"
         └── max(similarity_score) >= 0.3
               └──► Gemini (gemini-3.5-flash-lite via google-genai, or _MockGenAIClient fallback)
                      └──► GenerationResponse (answer + source#header citations)
```

---

## Evaluation Metrics & Failure / Abstention Analysis

Measured locally via `python -m scripts.benchmark` running [`MarkdownChunker`](src/retriever.py), [`NumpyRetriever`](src/retriever.py), [`run_evaluation`](src/eval.py), and [`RAGGenerator`](src/generator.py) abstention logic with a deterministic pure-NumPy TF-IDF embedder (`5` Markdown docs, `20` `##` chunks, `25` in-domain queries, `5` out-of-domain queries):

| Metric | Measured Value | Scope / Details |
| :--- | :--- | :--- |
| **Hit@1** | `1.000` (`25/25`) | `src/eval.py` (`run_evaluation`) across 25 in-domain queries |
| **Hit@3** | `1.000` (`25/25`) | `src/eval.py` (`run_evaluation`) across 25 in-domain queries |
| **MRR** | `1.000` | `src/eval.py` (`run_evaluation`) across 25 in-domain queries |
| **Avg Latency** | `0.1449 ms` | `0.144868 ms` per query (`TfidfEmbedder` + `NumpyRetriever`) |
| **OOD Abstention** | `4/5` (`80%`) | `RAGGenerator` (`SIMILARITY_THRESHOLD = 0.3`) on 5 out-of-domain probes |
| **In-Domain False Abstention** | `4/25` (`16%`) | Rank-1 hits scoring below `0.3` under sparse TF-IDF (`min = 0.1915`, `mean = 0.4026`) |

> **Abstention & Failure Analysis:**
> - **False abstentions (`4/25` in-domain queries):** Even when `NumpyRetriever` ranked the correct chunk #1 (`Hit@1 = 1.000`), short paraphrased queries (`0.1915`–`0.2786` TF-IDF cosine similarity) fell below `SIMILARITY_THRESHOLD = 0.3` (`"When does RAGGenerator abstain instead of calling the LLM?"` at `0.1915`, `"How do I configure database connection pooling?"` at `0.2256`, `"How does MarkdownChunker split documentation sections?"` at `0.2780`, `"Which HTTP status code is returned for burst traffic throttling?"` at `0.2786`).
> - **Missed OOD abstention (`1/5` OOD queries):** `"What was the company revenue in Q4 2025?"` scored `0.3238` due to unigram overlap on common function words without stop-word filtering; the other 4 OOD probes scored `0.1742`–`0.2727` and abstained cleanly.
> - **Scope note:** [`scripts/benchmark.py`](scripts/benchmark.py) exercises the real retrieval, evaluation, and abstention modules over an in-script 5-doc corpus using pure-NumPy TF-IDF (`_StubGenAIClient` only after threshold pass); `src/api.py` itself still uses `_mock_retrieve` until a live corpus/embedder is wired into `POST /query`.

---

## Security & Engineering Features

Engineered across five incremental delivery slices without external vector database dependencies:

- **Slice 1 — Retrieval Core (`src/retriever.py`)**
  - **Structure-aware chunking:** `MarkdownChunker` splits documentation on Markdown headers (`##` by default) to preserve semantic section boundaries and attach header metadata rather than slicing arbitrarily by character count.
  - **Pure NumPy cosine similarity:** `NumpyRetriever` and `cosine_similarity` compute vectorized similarity scores via NumPy broadcasting with explicit zero-vector guards (no NaNs) and `[-1.0, 1.0]` clipping.
- **Slice 2 — API & Grounded Generation (`src/api.py`, `src/schemas.py`, `src/generator.py`)**
  - **Strict schema validation:** Pydantic v2 models (`QueryRequest`, `GenerationResponse`, `DocumentChunk`, `RetrievalResult`) enforce request/response contracts at the API boundary.
  - **Grounded citations & low-similarity abstention:** `RAGGenerator` queries Gemini (`gemini-3.5-flash-lite` via `google-genai`), formats source-indexed context blocks with `source#header` citations, and **strictly abstains** (`"I do not have enough context to answer"`) whenever the highest cosine similarity score falls below `0.3` (`SIMILARITY_THRESHOLD`).
- **Slice 3 — Evaluation Harness (`src/eval.py`, `data/golden_qs.json`)**
  - **Automated retrieval metrics:** Evaluates retrievers against golden query/source pairs (seeded in `data/golden_qs.json`, structured for a 25–30 question benchmark) and reports `Hit@1`, `Hit@3`, Mean Reciprocal Rank (`MRR`), and average latency in milliseconds.
- **Slice 4 — Auth & Security (`src/security.py`)**
  - **OWASP LLM01 prompt injection delimiters:** `wrap_prompt` / `wrap_user_input` escapes XML/HTML characters (`html.escape(..., quote=False)`) and encloses untrusted user queries in `<user_input>...</user_input>` tags to prevent delimiter breakout.
  - **Phase 5 `X-API-Key` crypto:** `verify_api_key` performs constant-time key verification using `secrets.compare_digest` and raises HTTP `401` on mismatch to prevent timing side-channel leaks.
- **Slice 5 — CI/CD & Container Hardening (`Dockerfile`, `.github/workflows/ci.yml`)**
  - **Non-root container:** Built on `python:3.12-slim` and executes Uvicorn as an unprivileged `appuser`.
  - **Automated quality & security gates:** GitHub Actions CI runs `pytest`, `ruff`, `mypy`, `pip-audit`, and Trivy (`vuln` and `secret` scanners) on pushes to `main`.

> **Implementation note:** `src/retriever.py`, `src/security.py`, `src/generator.py`, and `src/eval.py` are implemented and unit-tested as standalone modules; `src/api.py` currently exposes the `POST /query` endpoint wired to `RAGGenerator` with a Slice 2 `_mock_retrieve` stub and an automatic `_MockGenAIClient` fallback when `GEMINI_API_KEY` is unset.

---

## Quickstart Instructions

Build and run the hardened non-root Docker container:

```bash
docker build -t devdocs-ai .
docker run -p 8000:8000 devdocs-ai
```

To enable live Gemini generation instead of the built-in mock client fallback, pass `GEMINI_API_KEY`:

```bash
docker run -p 8000:8000 -e GEMINI_API_KEY="your-api-key" devdocs-ai
```

---

## Architecture & Engineering Decisions

For a detailed breakdown of the technical trade-offs behind pure NumPy retrieval, structure-aware Markdown chunking, OWASP LLM01 prompt delimiting, abstention thresholds, and container/CI hardening, see **[`docs/DECISIONS.md`](docs/DECISIONS.md)**.

---

## API Usage (`POST /query`)

Send a query request (`query` required, `top_k` defaults to `5`):

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "How does authentication work?", "top_k": 5}'
```

Example response (`GenerationResponse`):

```json
{
  "answer": "Authentication is handled via JWT tokens and API keys as documented in the architecture guide.",
  "citations": [
    "docs/architecture.md#Authentication"
  ]
}
```

---

## Local Development & Verification

Install pinned dependencies (Python 3.12) and run the test, lint, and type-check suite:

```bash
pip install -r requirements.txt
pytest
ruff check .
mypy .
```

---

## Repository Layout

```text
.
├── .github/workflows/ci.yml   # CI pipeline (pytest, ruff, mypy, pip-audit, Trivy)
├── data/golden_qs.json        # Golden QA evaluation pairs
├── docs/DECISIONS.md          # Cold-written architecture trade-offs
├── scripts/
│   └── benchmark.py           # Reproducible pure-NumPy TF-IDF RAG & abstention benchmark
├── src/
│   ├── api.py                 # FastAPI app and POST /query endpoint
│   ├── eval.py                # Hit@1, Hit@3, MRR, and latency evaluation harness
│   ├── generator.py           # Grounded Gemini generator + similarity abstention
│   ├── retriever.py           # MarkdownChunker & pure NumPy NumpyRetriever
│   ├── schemas.py             # Pydantic v2 data contracts
│   └── security.py            # Constant-time API key check & OWASP LLM01 delimiter
├── tests/                     # Unit and API tests per slice
├── Dockerfile                 # Non-root python:3.12-slim image
├── Makefile                   # Slice lock/check/close/reset workflow helpers
├── pyproject.toml             # Pytest and Ruff configuration
└── requirements.txt           # Pinned runtime and dev/security dependencies
```
