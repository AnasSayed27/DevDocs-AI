# DevDocs AI v2 - High-Level Project Overview
**Goal:** Promote an 86-LOC baseline to a ~1,000-LOC standalone custom RAG microservice.
**Timeline:** Nov 19 – Dec 3 (10-12 hrs/week budget).
**Constraints:** No external vector DBs; strictly pure NumPy math, FastAPI, and Pydantic v2.

## The 5-Slice Execution Map
*   **Slice 1:** Retrieval Core (Structure-aware chunking, raw NumPy cosine similarity).
*   **Slice 2:** API & Generation (FastAPI, Pydantic v2, Gemini grounded citations).
*   **Slice 3:** Evaluation Harness (25-30 golden questions tracking Hit@K, MRR).
*   **Slice 4:** Auth & Security (X-API-Key Phase 5 Crypto, OWASP LLM01 delimiters).
*   **Slice 5:** CI/CD & Cold Docs (Non-root Dockerfile, GitHub Actions CI, docs/DECISIONS.md).


# DevDocs AI v2 - Execution State

## Slice 1: Retrieval Core
1. Escaped bugs found after make close: 0
2. 2-strike resets triggered: [0]
3. Minutes per slice: [35 min]
Status: [DONE]

## Slice 2: API & Generation
1. Escaped bugs found after make close: 0
2. 2-strike resets triggered: [0]
3. Minutes per slice: [30]
Status: DONE

## Slice 3: Evaluation Harness
1. Escaped bugs found after make close: 0
2. 2-strike resets triggered: [0]
3. Minutes per slice: [25]
Status: DONE