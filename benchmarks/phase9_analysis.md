# Phase 9 container smoke-test analysis

## Verified outcome

Docker built `llm-serving-platform:phase9` successfully with no Dockerfile check
warnings. The resulting image was a native Linux ARM64 image, approximately 622
MB by Docker's image-size metadata, and ran under the non-root `appuser` account.

The runtime import smoke test confirmed PyTorch 2.10.0 CPU, Ray 2.56.1, MLflow
3.15.1, and the installed project package. The real API container then:

1. started its HTTP server;
2. returned HTTP 503 from `/ready` while Qwen downloaded and loaded;
3. stored approximately 954 MB in the named Hugging Face cache volume;
4. transitioned to Docker health status `healthy`;
5. generated a 24-token response through `/v1/generate`;
6. stopped and was removed while its model volume remained available.

The generation took 9.51 seconds and the complete API request took 9.58 seconds.
This was a functional smoke test, not a controlled host-versus-container
benchmark: the prompt, token limit, warm-up state, and Docker CPU environment did
not match earlier experiments, so the timing should not be used to claim a
container performance penalty.

## What this proves

- The lockfile resolves for Linux ARM64, not only macOS ARM64.
- The installed project and native dependencies import inside the final image.
- Runtime permissions allow a non-root process to populate the model volume.
- Readiness correctly protects inference while the model is unavailable.
- Model state survives container deletion because it belongs to a volume.

It does not yet prove an AMD64 build, registry pull, Kubernetes operation, load
behavior, or production security posture. Those require separate verification.

## User Compose verification

The subsequent learning run used `docker compose up api` on port 8000. Compose
reported the service as healthy, and a real request produced 32 output tokens in
8.54 seconds (3.75 tokens/s), with 8.56 seconds total API latency. Standard
`docker compose down` then removed the container and network while retaining the
named model volume.

The generated explanation called containers "essentially virtual machines,"
which is technically misleading: containers share an operating-system kernel,
while Docker Desktop uses a Linux VM underneath on macOS. This content-quality
issue does not invalidate the container test and reinforces why serving checks
and model-quality evaluation must remain separate.
