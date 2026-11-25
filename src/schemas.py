from typing import Any, Dict, List
from pydantic import BaseModel, Field, model_validator


class DocumentChunk(BaseModel):
    """Represents a chunk of text extracted from a source document."""

    text: str = Field(..., description="The textual content of the chunk")
    source: str = Field(..., description="Source document identifier or filepath")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Source metadata (e.g., section headers, line offsets)",
    )

    @model_validator(mode="before")
    @classmethod
    def handle_metadata_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "source_metadata" in data and "metadata" not in data:
                data["metadata"] = data.pop("source_metadata")
        return data

    @property
    def source_metadata(self) -> Dict[str, Any]:
        return self.metadata


class RetrievalResult(BaseModel):
    """Represents a retrieved document chunk with its similarity score and source."""

    chunk: DocumentChunk = Field(..., description="The retrieved document chunk")
    similarity_score: float = Field(..., description="Cosine similarity score")
    source: str = Field(..., description="Source document identifier or filepath")

    @model_validator(mode="before")
    @classmethod
    def populate_source_from_chunk(cls, data: Any) -> Any:
        if isinstance(data, dict) and ("source" not in data or not data["source"]):
            chunk = data.get("chunk")
            if isinstance(chunk, DocumentChunk):
                data["source"] = chunk.source
            elif isinstance(chunk, dict) and "source" in chunk:
                data["source"] = chunk["source"]
        return data


class QueryRequest(BaseModel):
    """Represents an incoming user query request."""

    query: str = Field(..., min_length=1, description="The user's natural language query")
    top_k: int = Field(default=5, ge=1, description="Maximum number of chunks to retrieve")


class GenerationResponse(BaseModel):
    """Represents the generated answer and its supporting source citations."""

    answer: str = Field(..., description="The generated answer or abstention message")
    citations: List[str] = Field(
        default_factory=list,
        description="List of source identifiers cited in the answer",
    )
