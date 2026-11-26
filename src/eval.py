import inspect
import json
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from src.schemas import RetrievalResult


def _extract_source(result: Any) -> str:
    """Extract the document source identifier from a retrieval result."""
    if isinstance(result, str):
        return result
    if isinstance(result, RetrievalResult):
        return result.source or result.chunk.source
    if isinstance(result, dict):
        source = result.get("source")
        if source:
            return str(source)
        chunk = result.get("chunk")
        if isinstance(chunk, dict) and chunk.get("source"):
            return str(chunk["source"])
        if chunk is not None and hasattr(chunk, "source") and chunk.source:
            return str(chunk.source)
        return ""
    if hasattr(result, "source") and result.source:
        return str(result.source)
    if hasattr(result, "chunk") and hasattr(result.chunk, "source"):
        return str(result.chunk.source)
    return ""


def calculate_hit_at_k(
    retrieval_results: Sequence[RetrievalResult],
    expected_source: str,
    k: int,
) -> float:
    """Return 1.0 if expected_source appears in the top k retrieval results, else 0.0."""
    if k <= 0 or not retrieval_results:
        return 0.0

    for result in retrieval_results[:k]:
        if _extract_source(result) == expected_source:
            return 1.0
    return 0.0


def calculate_hit_at_1(
    retrieval_results: Sequence[RetrievalResult],
    expected_source: str,
) -> float:
    """Return 1.0 if the top retrieved chunk's source matches expected_source, else 0.0."""
    return calculate_hit_at_k(
        retrieval_results=retrieval_results,
        expected_source=expected_source,
        k=1,
    )


def calculate_hit_at_3(
    retrieval_results: Sequence[RetrievalResult],
    expected_source: str,
) -> float:
    """Return 1.0 if expected_source appears in the top 3 retrieval results, else 0.0."""
    return calculate_hit_at_k(
        retrieval_results=retrieval_results,
        expected_source=expected_source,
        k=3,
    )


def calculate_reciprocal_rank(
    retrieval_results: Sequence[RetrievalResult],
    expected_source: str,
) -> float:
    """Return 1 / rank of the first retrieved chunk matching expected_source, else 0.0."""
    for rank, result in enumerate(retrieval_results, start=1):
        if _extract_source(result) == expected_source:
            return 1.0 / rank
    return 0.0


def _load_golden_qa(golden_qa_path: str | Path) -> list[dict[str, str]]:
    """Load and validate golden query/expected_source pairs from a JSON file."""
    path = Path(golden_qa_path)
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise TypeError(f"Golden QA dataset at {path} must be a JSON list.")

    validated: list[dict[str, str]] = []
    for idx, item in enumerate(data):
        if not isinstance(item, dict):
            raise TypeError(f"Golden QA entry at index {idx} must be a JSON object.")
        if "query" not in item or "expected_source" not in item:
            raise ValueError(
                f"Golden QA entry at index {idx} must contain 'query' and 'expected_source'."
            )
        validated.append(
            {
                "query": str(item["query"]),
                "expected_source": str(item["expected_source"]),
            }
        )
    return validated


def _invoke_retriever(
    retriever: Any,
    query_input: Any,
    top_k: int | None = None,
) -> Sequence[RetrievalResult]:
    """Invoke a retriever instance (.retrieve) or callable with a query input."""
    target = retriever.retrieve if hasattr(retriever, "retrieve") else retriever
    if not callable(target):
        raise TypeError(
            "retriever must be callable or expose a callable 'retrieve' method."
        )

    params: Mapping[str, inspect.Parameter]
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        params = {}

    accepts_var_kw = any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )
    has_top_k = "top_k" in params
    top_k_required = (
        has_top_k and params["top_k"].default is inspect.Parameter.empty
    )

    kwargs: dict[str, Any] = {}
    if top_k is not None and (has_top_k or accepts_var_kw):
        kwargs["top_k"] = top_k
    elif top_k is None and top_k_required:
        kwargs["top_k"] = 5

    if "query" in params and params["query"].kind == inspect.Parameter.KEYWORD_ONLY:
        kwargs["query"] = query_input
        return target(**kwargs)

    return target(query_input, **kwargs)


def run_evaluation(
    retriever: Any,
    golden_qa_path: str | Path = "data/golden_qs.json",
    top_k: int | None = None,
    embedder: Callable[[str], Any] | None = None,
) -> dict[str, float]:
    """Run retrieval evaluation on golden QA pairs and return Hit@1, Hit@3, MRR, and latency."""
    golden_qa = _load_golden_qa(golden_qa_path)
    if not golden_qa:
        return {
            "hit_at_1": 0.0,
            "hit_at_3": 0.0,
            "mrr": 0.0,
            "avg_latency_ms": 0.0,
        }

    total_hit_at_1 = 0.0
    total_hit_at_3 = 0.0
    total_rr = 0.0
    total_latency_ms = 0.0

    for item in golden_qa:
        query_input = embedder(item["query"]) if embedder is not None else item["query"]
        start = time.perf_counter()
        results = list(_invoke_retriever(retriever, query_input, top_k=top_k))
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        expected_source = item["expected_source"]
        total_hit_at_1 += calculate_hit_at_1(
            retrieval_results=results,
            expected_source=expected_source,
        )
        total_hit_at_3 += calculate_hit_at_3(
            retrieval_results=results,
            expected_source=expected_source,
        )
        total_rr += calculate_reciprocal_rank(
            retrieval_results=results,
            expected_source=expected_source,
        )
        total_latency_ms += elapsed_ms

    count = float(len(golden_qa))
    return {
        "hit_at_1": total_hit_at_1 / count,
        "hit_at_3": total_hit_at_3 / count,
        "mrr": total_rr / count,
        "avg_latency_ms": total_latency_ms / count,
    }
