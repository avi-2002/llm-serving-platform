# Phase 12: release, CI/CD, and deployment

## Goal

Turn the eleven working phases into a reproducible portfolio release. The release
must be testable on every change, publishable as a container, deployable from a
known image, understandable by a reviewer, and honest about its limitations.

## The simple mental model

- **CI** is an automatic examiner: every push is linted, tested, and container-built.
- **CD** is an automatic packer: a version tag publishes the same image a server uses.
- **Registry** is an app warehouse: GHCR stores versioned container images.
- **Deployment** is the instruction telling a server which image and settings to run.
- **Secret** is private configuration supplied by the host, never committed to Git.

## What this phase adds

1. `.github/workflows/ci.yml` checks code and the Docker image on pushes and PRs.
2. `.github/workflows/publish-image.yml` publishes multi-architecture images on
   `v*` tags or a manual run.
3. The Kubernetes Deployment references the published GHCR image.
4. `streamlit_app.py` and `requirements.txt` provide a lightweight Streamlit
   Community Cloud entry point; the model is not installed in that frontend.
5. `docs/architecture.md` captures request flow, release flow, and boundaries.

## Deployment order

### 1. Verify locally

```bash
uv sync --locked
uv run ruff check .
uv run pytest -q
docker compose up --build api ui
```

### 2. Publish a release image

After CI passes on GitHub:

```bash
git tag -a v1.0.0 -m "LLM serving platform v1.0.0"
git push origin main --follow-tags
```

The publish workflow creates `ghcr.io/avi-2002/llm-serving-platform:1.0.0`
and `:latest`. Make the GHCR package public before using it without an image-pull
secret.

### 3. Deploy the backend

For a reachable Kubernetes cluster:

```bash
kubectl apply -k kubernetes
kubectl rollout status deployment/llm-serving -n llm-serving --timeout=10m
```

The included Service is deliberately `ClusterIP`. Production still needs a
provider-specific HTTPS Ingress or LoadBalancer, DNS, authentication, and rate
limits. Do not expose an unauthenticated paid inference endpoint.

### 4. Deploy the frontend

In Streamlit Community Cloud, select this GitHub repository and use
`streamlit_app.py` as the entry point. Add this secret in Advanced settings:

```toml
LLM_API_URL = "https://your-secured-backend.example.com"
```

The backend address must be public HTTPS; `localhost`, Docker service names, and
Kubernetes ClusterIP addresses are not reachable from Streamlit Cloud.

## Topics to study

- GitHub Actions events, jobs, runners, permissions, secrets, and caches
- Semantic Versioning and immutable container tags
- OCI images, manifests, image digests, SBOMs, and supply-chain provenance
- Kubernetes Ingress, TLS, Secrets, rolling updates, and Horizontal Pod Autoscaling
- Cold starts, model memory per replica, scale-to-zero, and inference cost
- API authentication, rate limiting, CORS, threat modelling, and denial of service
- SLOs, error budgets, distributed tracing, logs, metrics, and alerting

## Completion checklist

- Local lint and tests pass.
- CI passes on GitHub.
- `v1.0.0` publishes to GHCR.
- The image can answer a real inference request.
- The frontend uses a secret-backed public API URL.
- Public backend hardening is completed before a live internet deployment.
