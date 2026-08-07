from pathlib import Path


def test_dockerfile_uses_locked_non_root_multistage_build() -> None:
    dockerfile = Path("Dockerfile").read_text()

    assert "AS builder" in dockerfile
    assert "AS runtime" in dockerfile
    assert "uv sync --locked" in dockerfile
    assert "--no-dev" in dockerfile
    assert "USER appuser" in dockerfile
    assert "HEALTHCHECK" in dockerfile
    assert 'CMD ["llm-api"]' in dockerfile


def test_dockerignore_excludes_local_and_sensitive_state() -> None:
    ignored = set(Path(".dockerignore").read_text().splitlines())

    assert {".git", ".venv", ".env", "work", "outputs"} <= ignored


def test_compose_exposes_api_and_keeps_ray_optional() -> None:
    compose = Path("compose.yaml").read_text()

    assert '"8000:8000"' in compose
    assert 'profiles: ["ray"]' in compose
    assert "huggingface-cache:/data/huggingface" in compose
