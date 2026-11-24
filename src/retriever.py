import re
from typing import Any, List, Optional
import numpy as np

from src.schemas import DocumentChunk, RetrievalResult


class MarkdownChunker:
    """Structure-aware chunker for Markdown documents splitting on headers."""

    def __init__(self, header_level: int = 2):
        self.header_level = header_level

    def chunk(self, content: str, source: str = "unknown") -> List[DocumentChunk]:
        """Split markdown content by markdown headers (default: ##) preserving semantic sections."""
        if not content or not content.strip():
            return []

        header_prefix = "#" * self.header_level
        pattern = rf"(?m)^{re.escape(header_prefix)}\s+"

        if re.search(pattern, content):
            sections = re.split(pattern, content)
            chunks: List[DocumentChunk] = []
            for section in sections[1:]:  # skip preamble before first header
                section_clean = section.strip()
                if not section_clean:
                    continue
                lines = section_clean.split("\n", 1)
                header_title = lines[0].strip()
                chunk_text = f"{header_prefix} {section_clean}"
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        source=source,
                        metadata={"header": header_title},
                    )
                )
            if chunks:
                return chunks

        return [DocumentChunk(text=content.strip(), source=source)]


def chunk_markdown(content: str, source: str = "unknown") -> List[DocumentChunk]:
    """Convenience function that chunks markdown content using MarkdownChunker."""
    return MarkdownChunker().chunk(content=content, source=source)


def cosine_similarity(
    query_vector: np.ndarray,
    doc_vectors: np.ndarray,
) -> np.ndarray:
    """Compute cosine similarity between a query vector and a matrix of document vectors.

    Uses NumPy broadcasting and safely handles zero-vectors without producing NaN.
    """
    q = np.asarray(query_vector, dtype=np.float32)
    docs = np.asarray(doc_vectors, dtype=np.float32)

    if docs.ndim == 1:
        docs = docs.reshape(1, -1)

    norm_q = float(np.linalg.norm(q))
    norm_docs = np.linalg.norm(docs, axis=-1)
    dot_products = np.sum(docs * q, axis=-1)

    similarities = np.zeros(docs.shape[0], dtype=np.float32)
    valid_mask = (norm_q > 0) & (norm_docs > 0)

    if np.any(valid_mask):
        similarities[valid_mask] = dot_products[valid_mask] / (
            norm_docs[valid_mask] * norm_q
        )

    # Clip to valid cosine range [-1.0, 1.0] to guard against floating-point imprecision
    return np.clip(similarities, -1.0, 1.0)


def rank_chunks(
    query_vector: np.ndarray,
    doc_vectors: np.ndarray,
    chunks: List[DocumentChunk],
    top_k: Optional[int] = None,
) -> List[RetrievalResult]:
    """Rank document chunks by cosine similarity against the query vector."""
    if not chunks:
        return []

    scores = cosine_similarity(query_vector, doc_vectors)
    sorted_indices = np.argsort(-scores, kind="stable")

    if top_k is not None:
        sorted_indices = sorted_indices[:top_k]

    results: List[RetrievalResult] = []
    for idx in sorted_indices:
        results.append(
            RetrievalResult(
                chunk=chunks[idx],
                similarity_score=float(scores[idx]),
                source=chunks[idx].source,
            )
        )
    return results


class NumpyRetriever:
    """NumPy-based vector retriever using cosine similarity broadcasting."""

    def __init__(
        self,
        chunks: Optional[List[DocumentChunk]] = None,
        chunk_embeddings: Optional[np.ndarray] = None,
    ):
        self.chunks: List[DocumentChunk] = chunks or []
        self.chunk_embeddings: Optional[np.ndarray] = (
            np.asarray(chunk_embeddings, dtype=np.float32)
            if chunk_embeddings is not None
            else None
        )

    def retrieve(
        self,
        query_embedding: np.ndarray,
        chunk_embeddings: Optional[np.ndarray] = None,
        chunks: Optional[List[DocumentChunk]] = None,
        top_k: Optional[int] = None,
    ) -> List[RetrievalResult]:
        """Rank chunks by cosine similarity using NumPy broadcasting and return RetrievalResult list."""
        target_chunks = chunks if chunks is not None else self.chunks
        embeddings = (
            np.asarray(chunk_embeddings, dtype=np.float32)
            if chunk_embeddings is not None
            else self.chunk_embeddings
        )

        if embeddings is None:
            raise ValueError("No chunk embeddings provided to NumpyRetriever.")
        if len(target_chunks) != embeddings.shape[0]:
            raise ValueError(
                f"Mismatch between number of chunks ({len(target_chunks)}) "
                f"and embeddings ({embeddings.shape[0]})."
            )

        return rank_chunks(
            query_vector=query_embedding,
            doc_vectors=embeddings,
            chunks=target_chunks,
            top_k=top_k,
        )
