# Phase 11: Streamlit chat interface

## Why the UI is a separate service

The inference API owns the expensive model. Streamlit is a lightweight frontend
that sends HTTP requests to that API. Keeping them separate avoids loading a
second copy of Qwen and allows either service to be changed or scaled without
rebuilding the other architecture.

```text
browser -> Streamlit UI -> FastAPI or Ray Serve -> Qwen
```

## What the interface provides

- chat-style user and assistant messages;
- backend URL and readiness check;
- maximum-token, sampling, temperature, top-p, and seed controls;
- true incremental text with the Phase 6 SSE endpoint;
- automatic completed-response fallback for the Ray/Kubernetes endpoint;
- generation time, total latency, tokens/s, output tokens, batch size, and Ray
  replica telemetry when available;
- per-browser-session message history and a clear button.

The displayed conversation is UI history only. Every prompt is currently sent as
an independent request because the backend schema accepts one prompt rather than
a list of prior messages. Multi-turn model context is a separate future feature.

## Why streaming differs by backend

The standard FastAPI route exposes `/v1/generate/stream`, so the UI can display
text while generation continues. The current Ray Serve ingress batches complete
requests and does not expose that route. When it returns HTTP 404, the UI retries
through `/v1/generate` and explains that it is using completed-response mode.

This preserves correctness instead of pretending that a completed answer is a
live token stream. Production continuous batching would require a scheduler that
supports active streaming sequences.

## Run locally with true streaming

Terminal 1:

```bash
HF_HOME="$PWD/work/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
LLM_DEVICE=cpu LLM_DTYPE=float32 uv run llm-api
```

Terminal 2:

```bash
LLM_API_URL=http://127.0.0.1:8000 uv run llm-ui
```

Open <http://127.0.0.1:8501>.

## Run both services with Compose

```bash
docker compose up --build api ui
```

Compose waits for the model API to become healthy before starting the UI. Inside
Docker networking, the UI connects to `http://api:8000`; the browser still opens
the published UI at <http://127.0.0.1:8501>.

Stop both services while retaining the model volume:

```bash
docker compose down
```

## Connect the host UI to Kubernetes Ray Serve

Scale the backend up and forward its Service:

```bash
kubectl scale deployment llm-serving --replicas=1 -n llm-serving
kubectl rollout status deployment/llm-serving -n llm-serving --timeout=300s
kubectl port-forward -n llm-serving service/llm-serving 8080:80
```

In another terminal:

```bash
LLM_API_URL=http://127.0.0.1:8080 uv run llm-ui
```

The Stream response toggle may remain enabled; the client will detect the Ray
backend and fall back safely.

## Session state in plain language

Streamlit reruns the Python script whenever a widget changes. Session state is a
small per-browser memory area that keeps messages across those reruns. It is not a
database: restarting the UI or opening a new browser session starts fresh.

## Topics to study

- Frontend/backend separation and service boundaries.
- Streamlit reruns, widgets, and session state.
- Server-Sent Events and incremental rendering.
- Docker DNS service names versus host ports.
- Timeouts, retries, error states, and graceful feature fallback.
- Authentication, rate limits, input moderation, and secure secret handling.
- Stateless UI replicas and external conversation storage.

## Verified implementation checkpoint

The Streamlit 1.61.1 app passed an isolated render test, and the API client passed
SSE parsing and validation tests. Docker built the native Linux ARM64 `phase11`
image, started the UI as the non-root user on port 8501, and returned `ok` from
`/_stcore/health`. The temporary smoke-test container was then removed.

The remaining hands-on checkpoint is to start a real backend and submit prompts
through the browser, exercising both streaming and completed-response modes.

## User verification

The Docker Compose browser test started healthy API and UI containers and streamed
a real answer. First text appeared after 2.351 seconds, while the request completed
after 17.807 seconds, letting the user begin reading 15.456 seconds earlier. Both
containers and their network were then removed cleanly while the model volume was
retained.

The answer ended mid-sentence because it reached the configured output-token
limit. This is a model-generation setting, not a Streamlit transport failure; the
UI exposes the maximum-token control so the trade-off can be adjusted.

## Official reading

- Streamlit chat elements:
  <https://docs.streamlit.io/develop/api-reference/chat>
- Streamlit Session State:
  <https://docs.streamlit.io/develop/api-reference/caching-and-state/st.session_state>
- Deploy Streamlit with Docker:
  <https://docs.streamlit.io/deploy/tutorials/docker>
