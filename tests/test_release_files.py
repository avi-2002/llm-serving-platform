from pathlib import Path

import yaml


def test_ci_runs_lint_tests_and_container_build() -> None:
    ci = Path(".github/workflows/ci.yml").read_text()

    assert "uv sync --locked" in ci
    assert "uv run ruff check ." in ci
    assert "uv run pytest -q" in ci
    assert "docker buildx build --check ." in ci


def test_publish_workflow_uses_ghcr_and_version_tags() -> None:
    workflow = Path(".github/workflows/publish-image.yml").read_text()

    assert "ghcr.io" in workflow
    assert 'tags: ["v*"]' in workflow
    assert "packages: write" in workflow
    assert "provenance: true" in workflow
    assert "sbom: true" in workflow


def test_kubernetes_uses_published_image() -> None:
    deployment = yaml.safe_load(Path("kubernetes/deployment.yaml").read_text())
    image = deployment["spec"]["template"]["spec"]["containers"][0]["image"]

    assert image == "ghcr.io/avi-2002/llm-serving-platform:latest"


def test_streamlit_cloud_entrypoint_is_lightweight() -> None:
    requirements = Path("requirements.txt").read_text()

    assert "streamlit" in requirements
    assert "requests" in requirements
    assert "torch" not in requirements
    assert Path("streamlit_app.py").is_file()
