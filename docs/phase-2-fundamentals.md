# Phase 2: HTTP API fundamentals

Phase 1 could be used only by running a Python command on the same machine.
Phase 2 adds a network contract: another program can now send JSON over HTTP and
receive a validated JSON response without knowing how PyTorch or Transformers
works internally.

## Request flow

```text
HTTP client
  -> Uvicorn web server
  -> FastAPI route
  -> Pydantic validation
  -> model-readiness check
  -> single-model generation lock
  -> Phase 1 LocalLLM
  -> validated JSON response
```

## Server, framework, and schema responsibilities

- **Uvicorn** is the ASGI server. It opens a TCP port, accepts HTTP connections,
  and passes requests to the application.
- **FastAPI** maps HTTP methods and paths to Python functions, generates OpenAPI
  documentation, and turns application exceptions into HTTP responses.
- **Pydantic** validates and converts the JSON body against explicit schemas.
- **LocalLLM** still owns tokenization, model execution, decoding, and inference
  measurements. The API does not duplicate that logic.

## Health is not readiness

`GET /health` answers: **Can the API process respond?**

`GET /ready` answers: **Can this instance serve model inference now?**

During a real verification run, `/health` returned HTTP 200 while the model was
loading, but `/ready` returned HTTP 503. After loading completed, `/ready`
returned HTTP 200. This distinction prevents a load balancer from sending model
traffic to an alive process whose weights are not ready.

The model loads in a background thread. This keeps the asynchronous HTTP event
loop responsive during expensive synchronous initialization.

## Why validation is a serving concern

The public request schema bounds:

- prompt length to 1-8,000 characters;
- output length to 1-512 new tokens;
- temperature to `(0, 2]`;
- top-p to `(0, 1]`;
- seed to an unsigned 32-bit range;
- accepted fields to the documented schema only.

Invalid input returns HTTP 422 before model execution. Bounds protect correctness
and also reduce accidental resource exhaustion, though they are not a complete
security or rate-limiting system.

## HTTP status codes used

- `200 OK`: the request succeeded.
- `422 Unprocessable Content`: JSON was syntactically valid but violated the API
  schema.
- `503 Service Unavailable`: the process is alive but the model is not ready.

## Async does not make inference parallel

The endpoints are asynchronous so the server can remain responsive while waiting
for work. PyTorch inference is still blocking compute, so it runs in a worker
thread rather than directly on the event loop.

One `asyncio.Lock` serializes generation through the single model instance. This
provides safe, predictable Phase 2 behavior but means simultaneous generation
requests wait in a queue. Writing `async def` does not make neural-network compute
parallel or create additional model capacity.

Ray Serve will later replace this local boundary with explicit replicas,
backpressure, routing, resource assignment, and autoscaling.

## Request IDs and timing boundaries

Clients may send `X-Request-ID`; otherwise the API creates a UUID. A request ID
will later connect logs, metrics, and traces for the same operation.

The response currently reports:

- `generation_seconds`: Phase 1 model generation time;
- `total_request_seconds`: time inside the endpoint, including waiting for the
  generation lock, worker-thread scheduling, inference, and response construction.

It does not yet include network transit time measured by the client.

## Real verification result

The actual cached CPU model produced 32 output tokens with:

```text
model load:            4.463 s
generation:            2.620 s
endpoint total:        2.632 s
generation throughput: 12.21 tokens/s
```

This is a functional smoke test, not a benchmark distribution.

## Exercises before the next phase

1. Start the server and call `/health` and `/ready` immediately. Explain the
   temporary difference.
2. Send an empty prompt, an unknown field, and `max_new_tokens: 10000`; inspect
   each HTTP 422 response.
3. Call `/docs` and identify which constraints appear in the OpenAPI schema.
4. Send your own `X-Request-ID` and locate it in the response.
5. Compare `generation_seconds` and `total_request_seconds`. Explain why total
   time cannot be smaller in a correct measurement.
6. Explain why a worker thread prevents event-loop blocking but does not add
   inference capacity.

## Reading

- FastAPI request-body validation: <https://fastapi.tiangolo.com/tutorial/body/>
- FastAPI lifespan events: <https://fastapi.tiangolo.com/advanced/events/>
- HTTP status codes: <https://developer.mozilla.org/en-US/docs/Web/HTTP/Status>
- ASGI specification: <https://asgi.readthedocs.io/en/latest/>
