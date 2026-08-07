# Phase 5: Dynamic batching fundamentals

## Hypothesis

Phase 4 showed that two CPU replicas competed for one M1 and slowed each model
call. Dynamic batching tests a different strategy: keep one model replica, but
process several compatible prompts in one tensor operation.

```text
without batching: request A -> model call A
                  request B -> model call B

with batching:    requests A + B -> one batched model call -> responses A + B
```

Transformers naturally computes over a batch dimension. If vectorized work uses
the CPU more efficiently, aggregate tokens per second may rise without duplicating
model weights.

## Padding and attention masks

Prompts contain different token counts, but tensors must be rectangular. Shorter
prompts are left-padded to the longest prompt in the batch:

```text
prompt A: [PAD, PAD, token, token]
prompt B: [token, token, token, token]
```

The attention mask contains zero for padding and one for real tokens, preventing
padding from being treated as meaningful context. Left padding is important for
decoder-only generation because next-token prediction occurs at the right edge of
each sequence.

## Ray's batching queue

`@serve.batch` temporarily collects individual calls. It releases a batch when:

- the maximum batch size is reached; or
- the batch wait timeout expires.

Ray then calls `LocalLLM.generate_batch()` once and splits the returned list so
each original caller receives only its own response.

The configured controls are:

- `RAY_MAX_BATCH_SIZE=1`: unbatched baseline;
- `RAY_MAX_BATCH_SIZE=4`: combine at most four requests;
- `RAY_BATCH_WAIT_TIMEOUT_SECONDS=0.02`: wait at most 20 milliseconds.

## Latency-throughput tradeoff

Batching can improve hardware utilization, but the first request may wait for
other requests. At low traffic, the timeout can add latency without forming a
larger batch. At high traffic, batches fill quickly and the wait becomes small
relative to model compute.

An optimization is accepted only if measured throughput improves enough to justify
any p50 or p95 latency penalty for the intended workload.

## Compatibility rule

A single `model.generate()` call uses one set of decoding controls for the whole
batch. This implementation combines requests only when these values match:

- maximum new tokens;
- sampling enabled/disabled;
- temperature;
- top-p;
- seed.

If settings differ, the worker processes those requests separately. Silently
applying one user's decoding settings to another user's request would be a
correctness bug.

## What measurements matter

- service requests per second;
- aggregate output tokens per second;
- p50/p95/p99 client latency;
- observed mean and maximum batch size;
- errors;
- process memory and model replicas;

Per-response `tokens_per_second` is less intuitive for a batch because every
member shares the same batch execution time. Aggregate service tokens per second
is the main throughput measurement.

## Controlled experiment

Run one replica in both cases, with the same four PyTorch threads, model, prompt,
16-token response limit, greedy decoding, and client concurrency levels. Change
only batch size and wait timeout.

The unbatched run must be repeated using the new batching-capable code rather than
reusing Phase 4 numbers, because code-path changes can otherwise confound the
comparison.

## Observed result on the M1

The telemetry confirmed actual batches of 1, 2, and 4 as concurrency increased.
At concurrency 4, batching raised throughput from 0.75 to 1.93 requests/s (2.57x)
and reduced p95 latency from 9.03 to 2.10 seconds. One four-request model call
took about 2.06 seconds versus about 1.16 seconds for a one-request call, so the
larger operation cost more but completed four times the useful work.

At concurrency 1 no batch formed. Small differences there are measurement noise,
and the configured 20 ms collection window remains a possible low-traffic cost.
See `benchmarks/phase5_analysis.md` for the complete results and limitations.

## Topics to study

- Tensor batch dimensions and vectorization.
- Padding side and attention masks for decoder-only models.
- Static, dynamic, and continuous batching.
- Batch wait time versus latency SLO.
- Head-of-line blocking caused by different sequence lengths.
- Memory growth from larger batches and longer KV caches.
- Throughput-oriented versus latency-oriented serving.

## Official reading

- Ray dynamic request batching:
  <https://docs.ray.io/en/latest/serve/advanced-guides/dyn-req-batch.html>
- Hugging Face padding and truncation:
  <https://huggingface.co/docs/transformers/pad_truncation>
- PyTorch CPU thread control:
  <https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html>
