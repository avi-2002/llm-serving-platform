# Phase 9: Containerization with Docker

## The problem Docker solves

Running directly on a laptop depends on that laptop's Python installation,
virtual environment, files, and shell settings. A container image packages the
application and its locked dependencies into a repeatable filesystem with one
documented startup command.

An **image** is the read-only recipe. A **container** is a running instance of
that image. Rebuilding creates a new image; stopping a container does not delete
the image or the separate model-cache volume.

## Image design

The Dockerfile uses two stages:

1. The builder stage uses `uv` and `uv.lock` to install production dependencies.
2. The runtime stage copies only the completed virtual environment, creates a
   non-root user, exposes port 8000, and starts `llm-api`.

This separates build tools from runtime and makes dependency layers reusable.
The image is Linux-based even when built on macOS. Docker Desktop runs the Linux
container through its lightweight virtual machine.

## Why MPS is not used inside Docker

Apple's MPS accelerator is a macOS framework and is not exposed to Linux
containers in Docker Desktop. The container deliberately uses CPU inference.
This also matches the CPU-oriented server environment we will use for the first
Kubernetes deployment.

## Persistent model cache

Model weights are not copied into Git or baked into the image. Compose mounts the
named `huggingface-cache` volume at `/data/huggingface`. The first container start
downloads the model; later containers reuse that volume.

## Health and readiness

The image health check calls `/ready`, not merely `/health`. The container becomes
healthy only after the model is loaded and can accept inference. Its generous
start period and retries allow for the initial model download.

## Build and run the standard API

```bash
docker compose build api
docker compose up api
```

In a second terminal:

```bash
curl http://127.0.0.1:8000/ready
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Explain what a container image is.","max_new_tokens":32}'
docker compose ps
```

Stop the service without deleting the model cache:

```bash
docker compose down
```

Do not add `--volumes` unless you intentionally want to delete the cached model.

## Observed container result

The native Linux ARM64 image built without Dockerfile warnings and ran as the
non-root `appuser`. On first startup, `/ready` correctly returned 503 while Qwen
downloaded into the named volume, then changed to healthy. A real 24-token request
completed successfully. The temporary container was removed while its 954 MB
model cache remained for reuse.

See `benchmarks/phase9_analysis.md` for the measurements and their limitations.

## Optional Ray container

The Ray service is behind a Compose profile so it does not start accidentally
beside the standard API:

```bash
docker compose --profile ray up ray-api
curl http://127.0.0.1:8001/ready
```

## Topics to study

- Images, containers, layers, registries, and build context.
- Multi-stage builds and layer-cache ordering.
- Named volumes versus bind mounts.
- Container ports versus host ports.
- PID 1, signals, and the Compose `init` option.
- Root versus non-root container processes.
- Architecture tags (`linux/arm64` and `linux/amd64`).
- Image size, software bills of materials, and vulnerability scanning.

## Official reading

- Docker multi-stage builds:
  <https://docs.docker.com/build/building/multi-stage/>
- Docker Compose:
  <https://docs.docker.com/compose/>
- Using uv in Docker:
  <https://docs.astral.sh/uv/guides/integration/docker/>
