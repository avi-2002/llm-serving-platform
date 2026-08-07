from fastapi.testclient import TestClient

from low_latency_llm_serving.api.app import create_app
from low_latency_llm_serving.api.runtime import ModelRuntime
from tests.api.conftest import make_ready_client


def test_health_reports_process_even_when_model_is_loading() -> None:
    runtime = ModelRuntime()
    with TestClient(create_app(runtime, auto_load=False)) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model_status": "loading"}


def test_readiness_fails_while_model_is_loading() -> None:
    runtime = ModelRuntime()
    with TestClient(create_app(runtime, auto_load=False)) as client:
        response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["detail"]["model_status"] == "loading"


def test_ready_model_exposes_readiness_and_metadata() -> None:
    for client in make_ready_client():
        ready = client.get("/ready")
        metadata = client.get("/v1/metadata")

    assert ready.status_code == 200
    assert ready.json() == {"status": "ready", "model_id": "test/tiny-model"}
    assert metadata.status_code == 200
    assert metadata.json() == {
        "model_id": "test/tiny-model",
        "device": "cpu",
        "dtype": "float32",
        "load_seconds": 0.125,
    }

