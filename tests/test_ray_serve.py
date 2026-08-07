import pytest

from low_latency_llm_serving.ray_serve import (
    RayLLMDeployment,
    RayServeConfig,
    build_application,
)


def test_ray_deployment_uses_deferred_fastapi_factory() -> None:
    deployment_class = RayLLMDeployment.func_or_class
    deployment = object.__new__(deployment_class)
    application = deployment.__serve_build_asgi_app__()
    paths = {route.path for route in application.routes}

    assert {"/health", "/ready", "/v1/metadata", "/v1/generate"} <= paths


def test_build_application_does_not_start_ray_cluster() -> None:
    application = build_application(
        RayServeConfig(num_replicas=2, cpus_per_replica=2, torch_threads_per_replica=2)
    )

    assert application is not None


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
    ],
)
def test_invalid_ray_configuration_is_rejected(config: RayServeConfig) -> None:
    with pytest.raises(ValueError):
        config.validate()
