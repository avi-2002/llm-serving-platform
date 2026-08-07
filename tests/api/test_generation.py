from tests.api.conftest import make_ready_client


def test_stream_generate_returns_sse_chunks_and_measurements() -> None:
    for client in make_ready_client():
        with client.stream(
            "POST",
            "/v1/generate/stream",
            headers={"x-request-id": "stream-123"},
            json={"prompt": "Explain streaming.", "max_new_tokens": 32},
        ) as response:
            body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert "event: start" in body
    assert '"request_id": "stream-123"' in body
    assert body.count("event: token") == 2
    assert "event: done" in body
    assert '"response": "A deterministic test response."' in body
    assert '"time_to_first_chunk_seconds":' in body


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
