"""Reproducible local RAG benchmark using MarkdownChunker, NumpyRetriever, eval, and RAGGenerator.

Uses a pure-NumPy TF-IDF embedder over a 5-document (20-section) technical corpus,
25 in-domain golden queries, and 5 out-of-domain abstention probes. Does not modify
any repository source files, tests, or golden_qs.json.
"""

from __future__ import annotations

import json
import re
import tempfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.eval import (
    calculate_hit_at_1,
    calculate_hit_at_3,
    calculate_reciprocal_rank,
    run_evaluation,
)
from src.generator import ABSTENTION_MESSAGE, SIMILARITY_THRESHOLD, RAGGenerator
from src.retriever import MarkdownChunker, NumpyRetriever
from src.schemas import DocumentChunk

CORPUS: dict[str, str] = {
    "docs/architecture.md": """# Architecture Overview

## Authentication
User requests authenticate with short-lived JWT bearer tokens or service X-API-Key headers.
Service keys are verified in constant time with secrets.compare_digest and reject mismatches with HTTP 401.

## Rate Limiting
API gateways enforce a token bucket limit of 120 requests per minute per API key.
Burst traffic above the ceiling receives HTTP 429 with a Retry-After header.

## Response Caching
Read-heavy metadata endpoints cache serialized JSON payloads in memory for 60 seconds.
Cache keys include the normalized query string and schema version to avoid stale reads.

## Event Bus
Background audit events are published asynchronously to an internal queue after HTTP responses flush.
Consumers retry transient failures three times with exponential backoff before dead-lettering.
""",
    "docs/db.md": """# Database Operations

## Connection Pooling
PostgreSQL connections use a bounded pool with min_size=5 and max_size=25 per worker process.
Idle connections exceeding 300 seconds are recycled automatically to prevent stale sockets.

## Schema Migrations
Database migrations run as an isolated pre-deploy job using forward-only versioned SQL scripts.
Destructive column drops require a two-release expand-and-contract rollout.

## Read Replicas
Analytical reporting queries route to asynchronous read replicas to isolate OLTP latency.
Replica lag above 2 seconds triggers automatic fallback to the primary writer instance.

## Backup and Restore
Automated snapshots run every 6 hours with point-in-time recovery retained for 14 days.
Monthly restore drills verify checksums against a staging cluster before sign-off.
""",
    "docs/deploy.md": """# Deployment Guide

## Docker Build
Production images build on python:3.12-slim with pinned dependencies from requirements.txt.
The Dockerfile creates an unprivileged appuser and drops root privileges before starting Uvicorn.

## AWS ECS
Services deploy to AWS ECS Fargate behind an Application Load Balancer on port 8000.
Task definitions inject runtime secrets like GEMINI_API_KEY from AWS Secrets Manager.

## Health Checks
The load balancer polls GET /health every 15 seconds and requires two consecutive 200 responses.
Unhealthy tasks are drained for 30 seconds before ECS terminates the container.

## Rollbacks
Failed deployments automatically revert to the previous active ECS task revision when 5xx errors exceed 2 percent.
Manual rollbacks can be triggered via the release workflow using the prior immutable image digest.
""",
    "docs/rag.md": """# RAG Pipeline

## Markdown Chunking
MarkdownChunker splits documents on level-2 markdown headers so semantic sections remain intact.
Each DocumentChunk stores the originating file path and section header metadata for citations.

## Cosine Similarity
NumpyRetriever computes vectorized cosine similarity scores with NumPy broadcasting and zero-vector guards.
Scores are clipped to [-1.0, 1.0] to prevent floating-point drift from breaking ranking order.

## Abstention Threshold
RAGGenerator inspects the top retrieval similarity score before invoking the language model.
If the maximum score falls below 0.3, the generator abstains with a fixed insufficient-context message.

## Citation Formatting
Citations are deduplicated in rank order using source#header identifiers from chunks above threshold.
Only chunks meeting the similarity cutoff are included in the final GenerationResponse citations list.
""",
    "docs/security.md": """# Security Controls

## Prompt Injection Delimiters
Untrusted user queries pass through wrap_user_input to mitigate OWASP LLM01 prompt injection.
HTML/XML angle brackets are escaped before wrapping the query inside <user_input> tags.

## API Key Verification
verify_api_key compares incoming X-API-Key headers against the configured secret using HMAC constant-time equality.
Missing or invalid credentials return HTTP 401 without leaking key prefix or length timing.

## Dependency Auditing
CI executes pip-audit against pinned packages and Trivy filesystem scans for CVEs and leaked secrets.
Builds fail immediately on unignored high-severity vulnerabilities or exposed credentials.

## Container Hardening
Containers run read-only root filesystems where possible with PYTHONDONTWRITEBYTECODE enabled.
No .env files, SSH keys, or cloud credentials are baked into image layers.
""",
}

IN_DOMAIN_QUERIES: list[dict[str, str]] = [
    # docs/architecture.md
    {"query": "How does authentication work with JWT tokens and API keys?", "expected_source": "docs/architecture.md"},
    {"query": "What happens when a client exceeds the rate limiting quota?", "expected_source": "docs/architecture.md"},
    {"query": "How long are JSON responses kept in memory caching?", "expected_source": "docs/architecture.md"},
    {"query": "How are audit events retried on the event bus?", "expected_source": "docs/architecture.md"},
    {"query": "Which HTTP status code is returned for burst traffic throttling?", "expected_source": "docs/architecture.md"},
    # docs/db.md
    {"query": "How do I configure database connection pooling?", "expected_source": "docs/db.md"},
    {"query": "How are destructive SQL schema migrations rolled out?", "expected_source": "docs/db.md"},
    {"query": "When do reporting queries fall back from read replicas to primary?", "expected_source": "docs/db.md"},
    {"query": "What is the retention period for point-in-time recovery backups?", "expected_source": "docs/db.md"},
    {"query": "Why are idle PostgreSQL sockets recycled after 300 seconds?", "expected_source": "docs/db.md"},
    # docs/deploy.md
    {"query": "How do I deploy Docker containers on AWS ECS?", "expected_source": "docs/deploy.md"},
    {"query": "Which base image and non-root user does the Docker build use?", "expected_source": "docs/deploy.md"},
    {"query": "How does the load balancer perform health checks on port 8000?", "expected_source": "docs/deploy.md"},
    {"query": "When do automatic ECS rollbacks trigger on 5xx errors?", "expected_source": "docs/deploy.md"},
    {"query": "Where does ECS Fargate inject GEMINI_API_KEY at runtime?", "expected_source": "docs/deploy.md"},
    # docs/rag.md
    {"query": "How does MarkdownChunker split documentation sections?", "expected_source": "docs/rag.md"},
    {"query": "How does NumpyRetriever guard against zero vectors in cosine similarity?", "expected_source": "docs/rag.md"},
    {"query": "When does RAGGenerator abstain instead of calling the LLM?", "expected_source": "docs/rag.md"},
    {"query": "How are source#header citations deduplicated in GenerationResponse?", "expected_source": "docs/rag.md"},
    {"query": "Why are similarity scores clipped between -1.0 and 1.0?", "expected_source": "docs/rag.md"},
    # docs/security.md
    {"query": "How are OWASP LLM01 prompt injection attacks mitigated?", "expected_source": "docs/security.md"},
    {"query": "How does verify_api_key prevent timing attacks on X-API-Key?", "expected_source": "docs/security.md"},
    {"query": "Which CI scanners check for CVEs and leaked secrets?", "expected_source": "docs/security.md"},
    {"query": "What container hardening settings avoid leaking credentials in image layers?", "expected_source": "docs/security.md"},
    {"query": "How are angle brackets handled inside <user_input> tags?", "expected_source": "docs/security.md"},
]

OUT_OF_DOMAIN_QUERIES: list[str] = [
    "What is the best hydration percentage for sourdough bread?",
    "How do I configure Istio mTLS certificates in Kubernetes?",
    "What was the company revenue in Q4 2025?",
    "How do I fine-tune a ResNet-50 vision model in PyTorch?",
    "How do I set up Terraform remote state buckets on Google Cloud?",
]


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9_]+", text.lower())


class TfidfEmbedder:
    """Deterministic pure-NumPy TF-IDF embedder for local retrieval evaluation."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        tokenized_docs = [_tokenize(chunk.text) for chunk in chunks]
        vocab_set = sorted({tok for doc in tokenized_docs for tok in doc})
        self.vocab: dict[str, int] = {tok: idx for idx, tok in enumerate(vocab_set)}
        num_docs = len(tokenized_docs)

        df = np.zeros(len(self.vocab), dtype=np.float32)
        for doc in tokenized_docs:
            for tok in set(doc):
                df[self.vocab[tok]] += 1.0

        self.idf = np.log((num_docs + 1.0) / (df + 1.0)) + 1.0

    def embed_text(self, text: str) -> np.ndarray:
        vec = np.zeros(len(self.vocab), dtype=np.float32)
        tokens = _tokenize(text)
        if not tokens:
            return vec
        counts = Counter(tokens)
        total = float(len(tokens))
        for tok, count in counts.items():
            idx = self.vocab.get(tok)
            if idx is not None:
                tf = count / total
                vec[idx] = tf * float(self.idf[idx])
        norm = float(np.linalg.norm(vec))
        if norm > 0.0:
            vec = vec / norm
        return vec

    def embed_corpus(self, chunks: list[DocumentChunk]) -> np.ndarray:
        return np.vstack([self.embed_text(c.text) for c in chunks]).astype(np.float32)


class _StubGenAIClient:
    """Minimal stub client used only after similarity passes threshold."""

    def __init__(self) -> None:
        self.models = SimpleNamespace(
            generate_content=lambda model, contents: SimpleNamespace(text="Grounded answer.")
        )


def main() -> None:
    chunker = MarkdownChunker(header_level=2)
    all_chunks: list[DocumentChunk] = []
    for source, content in CORPUS.items():
        all_chunks.extend(chunker.chunk(content=content, source=source))

    embedder = TfidfEmbedder(all_chunks)
    chunk_embeddings = embedder.embed_corpus(all_chunks)
    retriever = NumpyRetriever(chunks=all_chunks, chunk_embeddings=chunk_embeddings)

    with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as tmp:
        json.dump(IN_DOMAIN_QUERIES, tmp, indent=2)
        tmp_path = Path(tmp.name)

    try:
        metrics = run_evaluation(
            retriever=retriever,
            golden_qa_path=tmp_path,
            top_k=5,
            embedder=embedder.embed_text,
        )
    finally:
        tmp_path.unlink(missing_ok=True)

    misses_at_1: list[dict[str, object]] = []
    misses_at_3: list[dict[str, object]] = []
    in_domain_abstained = 0
    in_domain_abstained_details: list[dict[str, object]] = []
    in_domain_scores: list[float] = []

    for item in IN_DOMAIN_QUERIES:
        q = item["query"]
        expected = item["expected_source"]
        results = retriever.retrieve(query_embedding=embedder.embed_text(q), top_k=5)
        top_score = results[0].similarity_score if results else 0.0
        in_domain_scores.append(top_score)

        h1 = calculate_hit_at_1(results, expected)
        h3 = calculate_hit_at_3(results, expected)
        rr = calculate_reciprocal_rank(results, expected)
        if h1 == 0.0:
            misses_at_1.append(
                {
                    "query": q,
                    "expected": expected,
                    "got_top": results[0].source if results else None,
                    "top_score": round(top_score, 4),
                    "rr": rr,
                }
            )
        if h3 == 0.0:
            misses_at_3.append({"query": q, "expected": expected})

        gen = RAGGenerator(retrieval_results=results, client=_StubGenAIClient()).generate(q)
        if gen.answer == ABSTENTION_MESSAGE:
            in_domain_abstained += 1
            in_domain_abstained_details.append(
                {
                    "query": q,
                    "expected": expected,
                    "top_score": round(top_score, 4),
                }
            )

    ood_abstained = 0
    ood_details: list[dict[str, object]] = []
    for q in OUT_OF_DOMAIN_QUERIES:
        results = retriever.retrieve(query_embedding=embedder.embed_text(q), top_k=5)
        top_score = results[0].similarity_score if results else 0.0
        gen = RAGGenerator(retrieval_results=results, client=_StubGenAIClient()).generate(q)
        abstained = gen.answer == ABSTENTION_MESSAGE
        if abstained:
            ood_abstained += 1
        ood_details.append({"query": q, "top_score": round(top_score, 4), "abstained": abstained})

    report = {
        "corpus_docs": len(CORPUS),
        "corpus_chunks": len(all_chunks),
        "in_domain_queries": len(IN_DOMAIN_QUERIES),
        "out_of_domain_queries": len(OUT_OF_DOMAIN_QUERIES),
        "metrics": metrics,
        "in_domain_top_score_min": float(min(in_domain_scores)),
        "in_domain_top_score_mean": float(sum(in_domain_scores) / len(in_domain_scores)),
        "in_domain_abstained": in_domain_abstained,
        "in_domain_abstained_details": in_domain_abstained_details,
        "ood_abstained": ood_abstained,
        "similarity_threshold": SIMILARITY_THRESHOLD,
        "misses_at_1": misses_at_1,
        "misses_at_3": misses_at_3,
        "ood_details": ood_details,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
