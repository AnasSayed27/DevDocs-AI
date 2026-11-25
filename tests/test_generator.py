from src.generator import generate_answer
from src.schemas import DocumentChunk, GenerationResponse, RetrievalResult


def test_generate_answer_abstains_on_low_similarity_scores():
    """Proves generation strictly abstains when all retrieved chunks have similarity < 0.3."""
    low_similarity_results = [
        RetrievalResult(
            chunk=DocumentChunk(
                text="Unrelated documentation about billing webhooks.",
                source="docs/billing.md",
            ),
            similarity_score=0.29,
            source="docs/billing.md",
        ),
        RetrievalResult(
            chunk=DocumentChunk(
                text="Another low-relevance snippet about CSS styling.",
                source="docs/ui.md",
            ),
            similarity_score=0.12,
            source="docs/ui.md",
        ),
    ]

    response = generate_answer(
        query="How do I configure JWT authentication?",
        retrieval_results=low_similarity_results,
    )

    assert isinstance(response, GenerationResponse)
    assert response.answer == "I do not have enough context to answer"
    assert response.citations == []
