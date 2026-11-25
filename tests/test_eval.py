from src.eval import calculate_hit_at_1
from src.schemas import DocumentChunk, RetrievalResult


def test_calculate_hit_at_1_returns_1_on_top_match_and_0_otherwise():
    """Proves Hit@1 returns 1.0 when top retrieved chunk matches expected source, and 0.0 otherwise."""
    retrieval_results = [
        RetrievalResult(
            chunk=DocumentChunk(
                text="## Authentication\nHandles user authentication via JWT tokens and API keys.",
                source="docs/architecture.md",
            ),
            similarity_score=0.92,
            source="docs/architecture.md",
        ),
        RetrievalResult(
            chunk=DocumentChunk(
                text="How do I configure database connection pooling?",
                source="docs/db.md",
            ),
            similarity_score=0.45,
            source="docs/db.md",
        ),
    ]

    hit_score = calculate_hit_at_1(
        retrieval_results=retrieval_results,
        expected_source="docs/architecture.md",
    )
    assert isinstance(hit_score, float)
    assert hit_score == 1.0

    # Expected source is at rank 2 (not rank 1), so Hit@1 must be 0.0
    miss_score = calculate_hit_at_1(
        retrieval_results=retrieval_results,
        expected_source="docs/db.md",
    )
    assert isinstance(miss_score, float)
    assert miss_score == 0.0

    # Empty retrieval results must return 0.0
    empty_score = calculate_hit_at_1(
        retrieval_results=[],
        expected_source="docs/architecture.md",
    )
    assert isinstance(empty_score, float)
    assert empty_score == 0.0
