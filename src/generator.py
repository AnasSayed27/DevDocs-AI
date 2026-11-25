from typing import Any, List, Optional

from google import genai

from src.schemas import GenerationResponse, RetrievalResult

ABSTENTION_MESSAGE = "I do not have enough context to answer"
SIMILARITY_THRESHOLD = 0.3
DEFAULT_MODEL = "gemini-3.5-flash-lite"


class RAGGenerator:
    """Generates grounded answers from retrieved document chunks using Gemini."""

    def __init__(
        self,
        retrieval_results: List[RetrievalResult],
        similarity_threshold: float = SIMILARITY_THRESHOLD,
        model: str = DEFAULT_MODEL,
        client: Optional[Any] = None,
    ):
        self.retrieval_results = retrieval_results
        self.similarity_threshold = similarity_threshold
        self.model = model
        self._client = client

    def _get_client(self) -> Any:
        if self._client is None:
            self._client = genai.Client()
        return self._client

    def _highest_similarity(self) -> float:
        if not self.retrieval_results:
            return 0.0
        return max(result.similarity_score for result in self.retrieval_results)

    def _format_prompt(self, query: str) -> str:
        context_blocks: List[str] = []
        for idx, result in enumerate(self.retrieval_results, start=1):
            chunk_data = result.chunk.model_dump()
            header = chunk_data.get("metadata", {}).get("header")
            source_label = f"{result.source} ({header})" if header else result.source
            context_blocks.append(
                f"[{idx}] Source: {source_label}\n{chunk_data['text']}"
            )

        context_str = "\n\n".join(context_blocks)
        return (
            "You are a technical documentation assistant. Answer the user's question "
            "using ONLY the provided context chunks. Cite the sources you use.\n\n"
            f"Context:\n{context_str}\n\n"
            f"Question: {query}\n"
            "Answer:"
        )

    def _extract_citations(self) -> List[str]:
        seen = set()
        citations: List[str] = []
        for result in self.retrieval_results:
            if result.similarity_score < self.similarity_threshold:
                continue
            chunk_meta = result.chunk.model_dump().get("metadata", {})
            header = chunk_meta.get("header")
            citation = f"{result.source}#{header}" if header else result.source
            if citation not in seen:
                seen.add(citation)
                citations.append(citation)
        return citations

    def generate(self, query: str) -> GenerationResponse:
        """Generate a grounded response or abstain if similarity is below threshold."""
        if self._highest_similarity() < self.similarity_threshold:
            return GenerationResponse(answer=ABSTENTION_MESSAGE, citations=[])

        prompt = self._format_prompt(query)
        client = self._get_client()
        llm_response = client.models.generate_content(
            model=self.model,
            contents=prompt,
        )
        answer_text = (llm_response.text or "").strip() or ABSTENTION_MESSAGE
        citations = self._extract_citations()
        return GenerationResponse(answer=answer_text, citations=citations)


def generate_answer(
    query: str,
    retrieval_results: List[RetrievalResult],
    similarity_threshold: float = SIMILARITY_THRESHOLD,
    model: str = DEFAULT_MODEL,
    client: Optional[Any] = None,
) -> GenerationResponse:
    """Convenience function to generate an answer from retrieval results."""
    generator = RAGGenerator(
        retrieval_results=retrieval_results,
        similarity_threshold=similarity_threshold,
        model=model,
        client=client,
    )
    return generator.generate(query)
