from tests.api.conftest import make_ready_client


def test_generate_returns_model_output_and_measurements() -> None:
    for client in make_ready_client():
        response = client.post(
            "/v1/generate",
            headers={"x-request-id": "request-123"},
            json={"prompt": "Explain tokenization.", "max_new_tokens": 32},
        )

    assert response.status_code == 200
    body = response.json()
    assert body["request_id"] == "request-123"
    assert body["response"] == "A deterministic test response."
    assert body["input_tokens"] == 5
    assert body["output_tokens"] == 6
    assert body["tokens_per_second"] == 24.0
    assert body["parameters"]["max_new_tokens"] == 32
    assert body["total_request_seconds"] >= 0


def test_generate_rejects_blank_prompt() -> None:
    for client in make_ready_client():
        response = client.post("/v1/generate", json={"prompt": "   "})

    assert response.status_code == 422


def test_generate_rejects_unknown_fields() -> None:
    for client in make_ready_client():
        response = client.post(
            "/v1/generate",
            json={"prompt": "Hello", "untrusted_option": True},
        )

    assert response.status_code == 422


def test_generate_rejects_excessive_output_limit() -> None:
    for client in make_ready_client():
        response = client.post(
            "/v1/generate",
            json={"prompt": "Hello", "max_new_tokens": 10_000},
        )

    assert response.status_code == 422

