# Phase 6: Streaming and perceived latency

## The problem

The original endpoint waits for the complete answer before returning anything.
Even if total generation is reasonably fast, a blank screen makes the service
feel slow.

The streaming endpoint returns pieces of decoded text while the model is still
generating:

```text
request -> prompt processing -> first text -> more text -> completed answer
                              ^
                              time to first chunk
```

## What we added

`POST /v1/generate/stream` uses Server-Sent Events (SSE). The response contains:

- a `start` event containing the request ID;
- multiple `token` events containing text fragments;
- a `done` event containing the complete answer, time to first chunk, and total
  request time;
- an `error` event if generation fails after streaming has begun.

The Hugging Face model still performs autoregressive generation. A
`TextIteratorStreamer` moves decoded fragments from the generation thread to the
HTTP response without waiting for the entire sequence.

## Why a chunk is not always one token

Tokens are model-level pieces such as a word, part of a word, or punctuation.
The text streamer may hold several tokens until they form clean printable text.
Therefore the metric is named `time_to_first_chunk_seconds`, not exact time to
first model token (TTFT).

## Streaming versus batching

They solve different problems:

- dynamic batching improves aggregate throughput for concurrent requests;
- streaming improves perceived responsiveness for one request.

Our Phase 5 Ray route batches complete generation calls. The Phase 6 streaming
route initially runs on the standard FastAPI server, where one model generation
is serialized at a time. A production scheduler can support continuous batching,
where active streams join and leave token-by-token batches, but that is a more
advanced architecture than Ray's request-level dynamic batching used here.

## Run the experiment

Start the standard API:

```bash
HF_HOME="$PWD/work/hf-cache" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
LLM_DEVICE=cpu LLM_DTYPE=float32 uv run llm-api
```

In a second terminal, use curl with buffering disabled:

```bash
curl -N -X POST http://127.0.0.1:8000/v1/generate/stream \
  -H 'Content-Type: application/json' \
  -H 'X-Request-ID: phase6-learning-1' \
  -d '{"prompt":"Explain KV caching simply.","max_new_tokens":64}'
```

Compare it with `POST /v1/generate`. The completed answer may take approximately
the same total time, but the streaming call should display useful text earlier.

## Topics to study

- HTTP chunked transfer and Server-Sent Events.
- Time to first token (TTFT), time per output token (TPOT), and inter-token
  latency (ITL).
- Prefill versus autoregressive decoding.
- Backpressure and client disconnection.
- Request batching versus continuous batching.
- Why reverse proxies sometimes buffer streaming responses.

## Reading

- FastAPI streaming responses: <https://fastapi.tiangolo.com/advanced/custom-response/#streamingresponse>
- Hugging Face streamers: <https://huggingface.co/docs/transformers/internal/generation_utils#streamers>
- SSE format: <https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events>
