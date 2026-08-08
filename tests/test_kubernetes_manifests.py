from pathlib import Path

import yaml


def load_resource(filename: str) -> dict:
    return yaml.safe_load(Path("kubernetes", filename).read_text())


def test_deployment_runs_ray_with_resources_security_and_three_probes() -> None:
    deployment = load_resource("deployment.yaml")
    pod_spec = deployment["spec"]["template"]["spec"]
    container = pod_spec["containers"][0]

    assert deployment["spec"]["replicas"] == 1
    assert deployment["spec"]["strategy"]["type"] == "Recreate"
    assert container["image"] == "ghcr.io/avi-2002/llm-serving-platform:latest"
    assert container["imagePullPolicy"] == "IfNotPresent"
    assert container["command"] == ["ray-llm-api"]
    assert container["resources"]["requests"] == {"cpu": "1", "memory": "2Gi"}
    assert container["resources"]["limits"] == {"cpu": "4", "memory": "6Gi"}
    assert container["startupProbe"]["httpGet"]["path"] == "/ready"
    assert container["readinessProbe"]["httpGet"]["path"] == "/ready"
    assert container["livenessProbe"]["httpGet"]["path"] == "/health"
    assert container["securityContext"]["readOnlyRootFilesystem"] is True
    assert pod_spec["securityContext"]["runAsNonRoot"] is True
    shared_memory = next(
        volume for volume in pod_spec["volumes"] if volume["name"] == "shared-memory"
    )
    assert shared_memory["emptyDir"] == {"medium": "Memory", "sizeLimit": "2Gi"}


def test_service_selects_deployment_and_maps_port() -> None:
    service = load_resource("service.yaml")

    assert service["spec"]["type"] == "ClusterIP"
    assert service["spec"]["selector"]["app.kubernetes.io/name"] == "llm-serving"
    assert service["spec"]["ports"][0]["port"] == 80
    assert service["spec"]["ports"][0]["targetPort"] == "http"


def test_model_cache_is_persistent_and_owned_by_non_root_group() -> None:
    claim = load_resource("pvc.yaml")
    deployment = load_resource("deployment.yaml")
    pod_spec = deployment["spec"]["template"]["spec"]

    assert claim["spec"]["accessModes"] == ["ReadWriteOnce"]
    assert claim["spec"]["resources"]["requests"]["storage"] == "2Gi"
    assert pod_spec["securityContext"]["fsGroup"] == 10001
    assert any(
        volume.get("persistentVolumeClaim", {}).get("claimName")
        == "huggingface-cache"
        for volume in pod_spec["volumes"]
    )


def test_kustomization_contains_all_resources() -> None:
    kustomization = load_resource("kustomization.yaml")

    assert kustomization["namespace"] == "llm-serving"
    assert set(kustomization["resources"]) == {
        "namespace.yaml",
        "configmap.yaml",
        "pvc.yaml",
        "deployment.yaml",
        "service.yaml",
    }
