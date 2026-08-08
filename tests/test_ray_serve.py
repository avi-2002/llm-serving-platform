import pytest

from low_latency_llm_serving.ray_serve import (
    RayServeConfig,
    build_application,
    build_ingress_app,
    http_options_from_environment,
)


def test_ray_deployment_uses_deferred_fastapi_factory() -> None:
    application = build_ingress_app()
    paths = {route.path for route in application.routes}

    assert {"/health", "/ready", "/v1/metadata", "/v1/generate"} <= paths


def test_ray_generation_openapi_declares_json_body_not_query_parameters() -> None:
    application = build_ingress_app()

    operation = application.openapi()["paths"]["/v1/generate"]["post"]

    assert "requestBody" in operation
    assert not operation.get("parameters")


def test_build_application_does_not_start_ray_cluster() -> None:
    application = build_application(
        RayServeConfig(num_replicas=2, cpus_per_replica=2, torch_threads_per_replica=2)
    )

    assert application is not None


def test_ray_http_proxy_defaults_to_container_reachable_address() -> None:
    assert http_options_from_environment() == {"host": "0.0.0.0", "port": 8000}


def test_ray_http_proxy_environment_is_validated(monkeypatch) -> None:
    monkeypatch.setenv("RAY_HTTP_HOST", " ")
    with pytest.raises(ValueError, match="RAY_HTTP_HOST"):
        http_options_from_environment()

    monkeypatch.setenv("RAY_HTTP_HOST", "127.0.0.1")
    monkeypatch.setenv("RAY_HTTP_PORT", "70000")
    with pytest.raises(ValueError, match="RAY_HTTP_PORT"):
        http_options_from_environment()


@pytest.mark.parametrize(
    "config",
    [
        RayServeConfig(model_id=" "),
        RayServeConfig(device="cuda"),
        RayServeConfig(dtype="int4"),
        RayServeConfig(num_replicas=0),
        RayServeConfig(cpus_per_replica=0),
        RayServeConfig(torch_threads_per_replica=0),
        RayServeConfig(max_queued_requests=-1),
        RayServeConfig(max_batch_size=0),
        RayServeConfig(batch_wait_timeout_seconds=-0.1),
    ],
)
def test_invalid_ray_configuration_is_rejected(config: RayServeConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()
