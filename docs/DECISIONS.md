# Architecture Decisions

1. **Pure NumPy cosine similarity instead of a vector database**
   Used NumPy broadcasting (`src/retriever.py`) to compute cosine similarity directly in memory rather than adding an external vector DB. For a small documentation set, this keeps the service fast, lightweight, and easy to deploy with fewer moving parts.

2. **Structure-aware Markdown chunking over fixed-length splitting**
   Split documents on Markdown headers (`##` by default in `MarkdownChunker`) instead of arbitrary character or token lengths. Keeping sections intact avoids cutting related context in half and improves retrieval relevance and citations.

3. **OWASP LLM01 prompt delimiters and API key checks (`src/security.py`)**
   To guard against prompt injection, `wrap_prompt` escapes XML/HTML characters (`html.escape(..., quote=False)`) and wraps user input inside `<user_input>...</user_input>` tags so injected tags cannot break out of the prompt boundary. `verify_api_key` uses `secrets.compare_digest` for constant-time key verification.

4. **Abstention on low retrieval confidence (`src/generator.py`)**
   If the top retrieval score falls below `0.3` (`SIMILARITY_THRESHOLD`), `RAGGenerator` abstains (`"I do not have enough context to answer"`) instead of sending weak context to the model.

5. **Non-root container and automated CI checks (`Dockerfile`, `.github/workflows/ci.yml`)**
   The Docker image is built on `python:3.12-slim` and switches to a non-root `appuser` before starting Uvicorn so the app never runs as root. On every push to `main`, GitHub Actions runs `pytest`, `ruff`, `mypy`, `pip-audit`, and Trivy to catch test, type, dependency, and secret issues automatically.
