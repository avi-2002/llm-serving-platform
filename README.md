# Low-Latency LLM Model Serving Platform

A learning-first implementation of local LLM inference that will evolve into a
Ray Serve and Kubernetes serving platform. Phase 1 deliberately uses plain
PyTorch and Hugging Face Transformers so the inference fundamentals remain
visible before serving infrastructure is introduced.

## Phase 1 architecture

```text
prompt
  -> model-specific chat template
  -> tokenizer (text to token IDs)
  -> causal language model on CPU or Apple MPS
  -> autoregressive token generation
  -> decoder (token IDs to text)
  -> timing and memory measurements
```

## Model

The baseline is `Qwen/Qwen2.5-0.5B-Instruct`, a compact instruction-tuned causal
language model with approximately 0.49 billion parameters. Its small size makes
CPU/MPS comparisons practical on a 16 GB Apple M1 while exercising the same
tokenization and generation abstractions used by larger models.

Model outputs may be inaccurate. This phase measures mechanics and performance;
it does not claim factual reliability.

## Setup

Requirements: macOS on Apple Silicon, Python 3.11, and `uv`.

```bash
uv sync
```

`uv sync` creates `.venv` and installs the exact versions in `uv.lock`. Do not
install project dependencies into the global Python environment.

## Run local inference

Automatically select MPS when available:

```bash
uv run local-llm --prompt "Explain KV caching in three short sentences."
```

Force a CPU baseline:

```bash
uv run local-llm \
  --device cpu \
  --dtype float32 \
  --max-new-tokens 64 \
  --prompt "Explain KV caching in three short sentences."
```

Generate machine-readable measurements:

```bash
uv run local-llm --json --prompt "What is autoregressive decoding?"
```

Sampling is opt-in. Without `--sample`, greedy decoding makes repeatable
baseline comparisons easier.

```bash
uv run local-llm \
  --sample \
  --temperature 0.7 \
  --top-p 0.9 \
  --seed 42 \
  --prompt "Give an analogy for tokenization."
```

The first execution downloads model files from Hugging Face, so its model-load
measurement includes network transfer and must be labelled a **cold download**.
Later executions reuse the local cache and provide the meaningful process-startup
baseline.

## What the measurements mean

- **Input tokens:** tokenized system instruction, user prompt, and chat-control
  tokens consumed before generation begins.
- **Output tokens:** newly generated tokens only.
- **Model load:** tokenizer/model loading, weight materialization, device transfer,
  and accelerator synchronization.
- **Generation:** complete `generate()` time, including prefill and decoding.
- **Tokens/s:** output tokens divided by generation time.
- **Process RSS:** resident memory attributed to the Python process. On unified
  memory systems this is an approximation, not an exact GPU-memory measurement.

This implementation does **not** yet report time to first token. `generate()`
returns after the full sequence is complete. We will add streaming instrumentation
later and then separate prefill latency, TTFT, and inter-token latency.

## Verify quality checks

```bash
uv run ruff check .
uv run pytest
```

The unit tests do not download model weights. They test device/dtype selection
and generation-parameter validation.

## Initial Apple M1 baseline

For a single 33-token prompt and 32-token greedy response, CPU FP32 produced
10.60 tokens/s while MPS FP16 produced 5.30 tokens/s. The GPU was slower for this
small batch-of-one workload; accelerator dispatch and sequential decode overhead
outweighed its parallelism. See `benchmarks/phase1_comparison.md` for the measured
values, limitations, and interpretation.

## Learning checkpoints

Before moving to an HTTP service, be able to explain:

1. Why chat messages must be converted to the model's exact chat template.
2. The difference between a tokenizer, token IDs, logits, and decoded text.
3. Why a causal model emits one token at a time even when `generate()` returns a
   complete string.
4. The difference between model loading, prefill, and decoding.
5. Why accelerator work must be synchronized before measuring wall-clock time.
6. Why FP16 uses less weight memory than FP32 and can change numerical behavior.
7. Why deterministic greedy decoding is useful for performance baselines.

## Current limitations

- Single process and single model instance
- One request at a time
- No HTTP API, streaming, dynamic batching, or autoscaling
- No latency percentiles or load generator
- No quality/factuality evaluation
- No Ray Serve, containers, Kubernetes, or MLflow yet

These limitations are intentional boundaries for Phase 1.

## Phase 2: local HTTP API

Start the API on `127.0.0.1:8000`:

```bash
LLM_DEVICE=auto LLM_DTYPE=auto uv run llm-api
```

The server starts accepting operational requests while the model loads in a
background thread. In a second Terminal, inspect it with:

```bash
curl http://127.0.0.1:8000/health
curl -i http://127.0.0.1:8000/ready
curl http://127.0.0.1:8000/v1/metadata
```

Generate a response:

```bash
curl -X POST http://127.0.0.1:8000/v1/generate \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: learning-request-1' \
  -d '{
    "prompt": "Explain why an API needs input validation.",
    "max_new_tokens": 64,
    "do_sample": false
  }'
```

Interactive OpenAPI documentation is available at
<http://127.0.0.1:8000/docs> while the server is running.

### Endpoint responsibilities

- `GET /health`: confirms that the API process can respond. It remains healthy
  while the model is loading.
- `GET /ready`: returns HTTP 200 only when the model can accept inference. It
  returns HTTP 503 while loading or after a loading failure.
- `GET /v1/metadata`: reports the loaded model, device, dtype, and load time.
- `POST /v1/generate`: validates bounded generation settings and returns text,
  token counts, timings, model identity, and a request ID.

Phase 2 intentionally serializes generation through one model instance. This
protects the local runtime but does not provide scalable concurrency. Later Ray
Serve replicas, routing, backpressure, and autoscaling will address that boundary.

## Phase 3: repeated and concurrent benchmarks

With `llm-api` running in another Terminal, execute:

```bash
uv run llm-benchmark \
  --concurrency 1 2 4 \
  --requests 5 \
  --warmup 1 \
  --max-new-tokens 32 \
  --output work/phase3-results.json
```

The harness keeps the prompt and decoding configuration fixed, varies only the
number of simultaneous HTTP workers, and records both raw requests and aggregate
measurements:

- successful and failed requests;
- error rate;
- client p50, p95, and p99 latency;
- requests per second;
- output tokens per second;
- server generation and endpoint timings;
- model metadata and benchmark configuration.

Files under `work/` are intentionally ignored because repeated experiments are
scratch data. Curated, interpreted results belong under `benchmarks/`.

## Phase 4: Ray Serve replicas

Start a one-replica Ray Serve deployment using the cached model:

```bash
HF_HOME="$PWD/work/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_USAGE_STATS_ENABLED=0 \
RAY_NUM_REPLICAS=1 RAY_CPUS_PER_REPLICA=4 TORCH_THREADS_PER_REPLICA=4 \
uv run ray-llm-api
```

The Ray deployment preserves the Phase 2 routes, so `llm-benchmark` works without
changes. To test two fixed CPU replicas on the 8-core M1:

```bash
HF_HOME="$PWD/work/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 RAY_USAGE_STATS_ENABLED=0 \
RAY_NUM_REPLICAS=2 RAY_CPUS_PER_REPLICA=4 TORCH_THREADS_PER_REPLICA=4 \
uv run ray-llm-api
```

Each replica owns a separate model instance. `max_ongoing_requests=1` tells Ray
to send at most one active generation to each replica. Generation responses expose
`replica_id`, allowing benchmark results to confirm routing across replicas.

On the measured M1 baseline at concurrency 4, two replicas improved throughput
from 0.76 to 0.96 requests/s and reduced p95 latency from 7.73 to 4.31 seconds
versus one Ray replica. The gain was below 2x because both replicas shared the same
physical CPU and each generation slowed under contention. See
`benchmarks/phase4_analysis.md` for the complete interpretation and limitations.

## Phase 5: dynamic request batching

The Ray model worker can combine compatible requests into one padded tensor batch.
Batch size and maximum wait are runtime configuration, so the same code supports
an unbatched control and a batched candidate.

Unbatched control:

```bash
RAY_NUM_REPLICAS=1 RAY_MAX_BATCH_SIZE=1 RAY_BATCH_WAIT_TIMEOUT_SECONDS=0 \
RAY_CPUS_PER_REPLICA=4 TORCH_THREADS_PER_REPLICA=4 \
uv run ray-llm-api
```

Batched candidate:

```bash
RAY_NUM_REPLICAS=1 RAY_MAX_BATCH_SIZE=4 RAY_BATCH_WAIT_TIMEOUT_SECONDS=0.02 \
RAY_CPUS_PER_REPLICA=4 TORCH_THREADS_PER_REPLICA=4 \
uv run ray-llm-api
```

Use the full offline/cache environment shown in the Phase 4 commands. Benchmark
responses record actual batch size; summaries report mean and maximum observed
batch sizes. Requests with different decoding settings safely fall back to
individual generation rather than being combined incorrectly.

In the measured M1 experiment, actual batches reached 4 at concurrency 4.
Compared with the unbatched control, throughput increased from 0.75 to 1.93
requests/s (2.57x), while p95 latency fell from 9.03 to 2.10 seconds. At
concurrency 1 no batch formed, so batching offered no demonstrated low-load
advantage and retained a possible 20 ms collection cost. See
`benchmarks/phase5_analysis.md` for the controlled comparison.

## Phase 6: streaming responses

Start the standard `llm-api`, then request a streamed answer:

```bash
curl -N -X POST http://127.0.0.1:8000/v1/generate/stream \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: phase6-learning-1' \
  -d '{"prompt":"Explain KV caching simply.","max_new_tokens":64}'
```

The SSE response emits `start`, `token`, and `done` events. The final event reports
time to first displayed text chunk and total request time. This route improves
perceived responsiveness; it does not make the underlying model generate tokens
faster. See `docs/phase-6-fundamentals.md` for the concepts and experiment.

In the measured M1 smoke test, the first displayed text arrived after 0.308
seconds while the complete response took 4.381 seconds. Streaming therefore let
the user begin reading about 4.07 seconds before generation finished. See
`benchmarks/phase6_analysis.md` for the result and limitations.

## Phase 7: Prometheus observability

The standard API records bounded, dashboard-friendly metrics for traffic,
failures, active inference, latency, first streamed text, generated output, and
batch sizes. After making inference requests, inspect them with:

```bash
curl -s http://127.0.0.1:8000/metrics | grep '^llm_'
```

The `/metrics` route uses Prometheus exposition format. See
`docs/phase-7-fundamentals.md` for metric types, cardinality, PromQL topics, and
the learning experiment.

The local verification recorded both normal and streaming requests successfully,
with their in-progress gauges returning to zero. Streaming's first text arrived
in 0.257 seconds versus 4.224 seconds for the complete request. See
`benchmarks/phase7_analysis.md` for the recorded signals and interpretation.

## Phase 8: Evaluation and MLflow

Run a fixed, version-controlled quality and performance evaluation against a
running API and save it as an MLflow experiment:

```bash
uv run llm-evaluate \
  --run-name qwen-cpu-baseline \
  --output work/phase8-eval.json
```

Inspect experiment parameters, metrics, and per-case artifacts locally:

```bash
uv run mlflow ui --backend-store-uri "sqlite:///$PWD/work/mlflow.db" --port 5000
```

Visit <http://127.0.0.1:5000>. The initial code-based concept coverage and
forbidden-claim checks are deliberately transparent regression signals, not proof
of factual correctness. See `docs/phase-8-fundamentals.md`.

The first Qwen 0.5B baseline completed 5/5 cases with 46.7% concept coverage and
3.900-second mean latency. Manual review caught an incorrect streaming claim that
the initial phrase-based hallucination rule missed, so the failure was added to
the regression dataset. See `benchmarks/phase8_analysis.md`.

A second MLflow run with the expanded regression rules correctly flagged two of
five answers (40%) while concept coverage remained 46.7%. This comparison
demonstrates that evaluator changes must be versioned and interpreted separately
from model changes.
