import math
import numpy as np
import pytest

from src.schemas import DocumentChunk, RetrievalResult
from src.retriever import chunk_markdown, cosine_similarity, rank_chunks


def test_chunking_splits_by_markdown_headers_not_blind_length():
    """Proves chunking splits by Markdown headers (##) rather than arbitrary character length."""
    markdown_content = (
        "# System Architecture\n\n"
        "Overview of the core architecture.\n\n"
        "## Authentication\n"
        "Handles user authentication via JWT tokens and API keys.\n\n"
        "## Vector Search Engine\n"
        "This is an intentionally longer section containing detailed technical specifications. "
        "It describes raw NumPy cosine similarity search, matrix transformations, and normalization routines. "
        "A blind string-length chunker with a small chunk size (e.g., 100-200 characters) would arbitrarily split "
        "this section across multiple fragments, severing sentences. Markdown-aware chunking must keep this section "
        "intact under its header.\n\n"
        "## Observability and Logging\n"
        "Structured JSON logging and telemetry metrics collection."
    )
    source_file = "docs/architecture.md"

    chunks = chunk_markdown(markdown_content, source=source_file)

    assert isinstance(chunks, list)
    assert len(chunks) == 3

    # All returned chunks must be DocumentChunk instances with correct source
    assert all(isinstance(c, DocumentChunk) for c in chunks)
    assert all(c.source == source_file for c in chunks)

    # Verify each chunk contains its respective section header / content
    section_headers = ["Authentication", "Vector Search Engine", "Observability and Logging"]
    for header in section_headers:
        matching_chunks = [c for c in chunks if header in c.text or header in str(c.metadata)]
        assert len(matching_chunks) == 1, f"Expected exactly one chunk for header: {header}"

    # Verify that the long section was not blindly sliced mid-content
    vector_search_chunk = next(
        c for c in chunks
        if "Vector Search Engine" in c.text or c.metadata.get("header") == "Vector Search Engine"
    )
    assert "severing sentences" in vector_search_chunk.text
    assert "normalization routines" in vector_search_chunk.text
    assert "Observability" not in vector_search_chunk.text


def test_cosine_similarity_ranks_perfect_match_highest():
    """Proves NumPy cosine similarity correctly ranks a perfect-match question highest."""
    chunks = [
        DocumentChunk(text="How do I configure database connection pooling?", source="docs/db.md"),
        DocumentChunk(text="Where do I set up JWT authentication tokens?", source="docs/auth.md"),
        DocumentChunk(text="Guide to deploying Docker containers on AWS ECS.", source="docs/deploy.md"),
        DocumentChunk(text="Writing unit tests with pytest fixtures.", source="docs/testing.md"),
    ]

    # Normalized 4-dimensional synthetic embeddings
    query_vector = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    doc_vectors = np.array([
        [0.6, 0.0, 0.8, 0.0],       # chunk 0: orthogonal to query (sim: 0.0)
        [0.0, 1.0, 0.0, 0.0],       # chunk 1: perfect match (sim: 1.0)
        [0.0, 0.5, 0.5, 0.7071],    # chunk 2: partial match (sim: 0.5)
        [0.1, 0.2, 0.0, 0.97],      # chunk 3: low match (sim: 0.2)
    ], dtype=np.float32)

    # Raw cosine similarity computation
    similarities = cosine_similarity(query_vector, doc_vectors)
    assert isinstance(similarities, np.ndarray)
    assert similarities[1] == pytest.approx(1.0, abs=1e-5)
    assert np.argmax(similarities) == 1

    # End-to-end ranking
    ranked_results = rank_chunks(query_vector, doc_vectors, chunks)

    assert isinstance(ranked_results, list)
    assert len(ranked_results) == len(chunks)
    assert all(isinstance(r, RetrievalResult) for r in ranked_results)

    # Perfect match must be ranked first (index 0)
    top_result = ranked_results[0]
    assert top_result.chunk.text == chunks[1].text
    assert top_result.similarity_score == pytest.approx(1.0, abs=1e-5)
    assert top_result.source == chunks[1].source

    # Results must be in monotonically non-increasing order of similarity
    scores = [r.similarity_score for r in ranked_results]
    assert scores == sorted(scores, reverse=True)


def test_retriever_handles_empty_text_and_zero_vectors():
    """Negative edge-case test: handles empty/whitespace text inputs and zero-vectors without NaN."""
    # 1. Edge-case: Empty and whitespace-only text inputs
    assert chunk_markdown("", source="empty.md") == []
    assert chunk_markdown("   \n\n\t  \n   ", source="whitespace.md") == []

    # 2. Edge-case: Zero query vector against document vectors
    zero_query = np.zeros(4, dtype=np.float32)
    normal_doc_vectors = np.array([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
    ], dtype=np.float32)

    scores_zero_query = cosine_similarity(zero_query, normal_doc_vectors)
    assert not np.any(np.isnan(scores_zero_query)), "Cosine similarity produced NaN for zero-query vector"
    assert np.all(scores_zero_query == 0.0), "Zero-query vector should produce 0.0 similarity"

    # 3. Edge-case: Normal query vector against zero document vector
    normal_query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    doc_vectors_with_zero = np.array([
        [0.0, 0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
    ], dtype=np.float32)

    scores_zero_doc = cosine_similarity(normal_query, doc_vectors_with_zero)
    assert not np.any(np.isnan(scores_zero_doc)), "Cosine similarity produced NaN for zero document vector"
    assert scores_zero_doc[0] == pytest.approx(0.0)
    assert scores_zero_doc[1] == pytest.approx(1.0)

    # 4. Edge-case: Ranking results with zero-vectors must not contain NaN scores
    chunks = [
        DocumentChunk(text="Chunk A", source="docs/a.md"),
        DocumentChunk(text="Chunk B", source="docs/b.md"),
    ]
    ranked_zero = rank_chunks(zero_query, normal_doc_vectors, chunks)
    assert len(ranked_zero) == 2
    for res in ranked_zero:
        assert isinstance(res, RetrievalResult)
        assert not math.isnan(res.similarity_score)
        assert res.similarity_score == pytest.approx(0.0)
