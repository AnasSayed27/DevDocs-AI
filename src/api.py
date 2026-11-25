import os
from types import SimpleNamespace
from typing import List

from fastapi import FastAPI

from src.generator import RAGGenerator
from src.schemas import (
    DocumentChunk,
    GenerationResponse,
    QueryRequest,
    RetrievalResult,
)

app = FastAPI(title="DevDocs AI v2", version="0.2.0")


class _MockModels:
    def generate_content(self, model: str, contents: str) -> SimpleNamespace:
        return SimpleNamespace(
            text="Authentication is handled via JWT tokens and API keys as documented in the architecture guide."
        )


class _MockGenAIClient:
    def __init__(self) -> None:
        self.models = _MockModels()


def _mock_retrieve(query: str, top_k: int) -> List[RetrievalResult]:
    """Return dummy retrieval results for Slice 2 before full pipeline wiring."""
    dummy_results = [
        RetrievalResult(
            chunk=DocumentChunk(
                text=f"## Authentication\nHandles user authentication via JWT tokens and API keys. (Query: {query})",
                source="docs/architecture.md",
                metadata={"header": "Authentication"},
            ),
            similarity_score=0.85,
            source="docs/architecture.md",
        ),
    ]
    return dummy_results[:top_k]


@app.post("/query", response_model=GenerationResponse)
def query_endpoint(request: QueryRequest) -> GenerationResponse:
    """Accept a QueryRequest, run retrieval and generation, and return GenerationResponse."""
    request_data = request.model_dump()
    retrieval_results = _mock_retrieve(
        query=request_data["query"],
        top_k=request_data["top_k"],
    )

    client = None if os.environ.get("GEMINI_API_KEY") else _MockGenAIClient()
    generator = RAGGenerator(retrieval_results=retrieval_results, client=client)
    response = generator.generate(query=request_data["query"])
    return GenerationResponse.model_validate(response.model_dump())
