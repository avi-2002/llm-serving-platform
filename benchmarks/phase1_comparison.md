# Phase 1 CPU versus MPS comparison

These are single cached runs on an Apple M1 MacBook Air with 16 GB unified
memory. They establish an initial observation, not a statistically reliable
performance claim.

| Measurement | CPU FP32 | MPS FP16 | Observation |
|---|---:|---:|---|
| Model load | 3.560 s | 3.286 s | MPS was 7.7% faster |
| Generation | 3.018 s | 6.040 s | CPU was 2.00x faster |
| Output speed | 10.60 tokens/s | 5.30 tokens/s | CPU was 2.00x faster |
| Process RSS | 2,119.9 MiB | 586.6 MiB | MPS-reported RSS was 72.3% lower |

Both runs used the same model, prompt, 33 input tokens, greedy decoding, and 32
output tokens. CPU used FP32 while MPS used FP16, so this comparison represents
the practical automatic configurations rather than an isolation of device alone.

## Interpretation

The GPU result is not a failure. This workload is a batch of one using a very
small model and only 32 output tokens. Kernel dispatch, device coordination, and
sequential token decoding can outweigh parallel-compute benefits. A GPU is more
likely to help when the model, batch, prompt, or concurrent workload provides
enough parallel work to amortize that overhead.

Process RSS is not total system memory or exact GPU memory on Apple Silicon.
CPU and GPU share unified physical memory, and framework/driver allocations may
not be attributed to the Python process in the same way. The lower MPS RSS is
therefore directionally useful but must not be advertised as exact model memory.

## What may change this conclusion

- repeated runs with warm-up and latency percentiles;
- longer input and output sequences;
- batching or concurrent requests;
- a larger model;
- matching precision across devices where supported;
- different PyTorch, Transformers, or macOS versions;
- a serving engine with optimized attention and KV-cache management.

Later benchmark phases will control these variables and report distributions
instead of drawing conclusions from one run.
