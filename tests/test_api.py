from fastapi.testclient import TestClient

from src.api import app
from src.schemas import GenerationResponse


def test_query_endpoint_returns_200_and_valid_generation_response():
    """Tests that POST /query returns 200 OK and a valid GenerationResponse payload."""
    client = TestClient(app)

    response = client.post("/query", json={"query": "How does authentication work?"})

    assert response.status_code == 200
    parsed = GenerationResponse.model_validate(response.json())
    assert isinstance(parsed.answer, str)
    assert len(parsed.answer) > 0
    assert isinstance(parsed.citations, list)
