# Phase 1: Local inference fundamentals

## What actually happens during generation

1. The chat template converts role-based messages into the control-token format
   used during instruction tuning.
2. The tokenizer maps that formatted text to integer token IDs.
3. During **prefill**, the model processes all input tokens and creates an initial
   key-value (KV) cache.
4. During **decode**, the model produces logits for the next token, selects a token,
   appends it to the sequence, updates the KV cache, and repeats.
5. The tokenizer decodes only the newly generated token IDs back to text.

An LLM does not directly produce a sentence. It repeatedly predicts a probability
distribution over its vocabulary and selects one next token.

## Key objects in the code

- `AutoTokenizer`: text/chat formatting to token IDs and token IDs back to text.
- `AutoModelForCausalLM`: a decoder-only model with a next-token prediction head.
- `input_ids`: integer token IDs shaped as `[batch_size, sequence_length]`.
- `attention_mask`: marks real tokens so padding cannot affect attention.
- `generate()`: a decoding loop built on repeated model forward passes.
- `torch.inference_mode()`: disables gradient tracking because inference does not
  perform backpropagation.

## Greedy decoding and sampling

Greedy decoding selects the highest-probability token at every step. It is usually
repeatable and is therefore the Phase 1 performance baseline.

Sampling draws from the token distribution:

- `temperature < 1` sharpens the distribution; `temperature > 1` flattens it.
- `top_p` keeps the smallest token set whose cumulative probability reaches the
  threshold, then samples from that set.
- a random seed improves repeatability but does not guarantee identical results
  across all hardware and library versions.

Do not compare performance runs when decoding settings, prompt, output limit,
device, precision, or software versions differ.

## Latency boundaries

The current `generation_seconds` covers both prefill and all decoding because the
non-streaming `generate()` call returns only after the sequence completes.

It does not yet expose:

- time to first token (TTFT), mostly influenced by queueing and prefill;
- time per output token (TPOT) or inter-token latency;
- request queueing time;
- latency percentiles across repeated/concurrent requests.

Those measurements require streaming and a benchmark harness in later phases.

## Why accelerator synchronization matters

Accelerator operations are normally asynchronous: Python can continue before the
GPU has completed queued work. Measuring the clock without synchronization can
therefore under-report execution time. The project calls `torch.mps.synchronize()`
immediately before starting and after stopping the generation timer.

CPU operations used here are synchronous from the caller's perspective.

## Precision and memory

With `N` parameters, weight storage is approximately:

```text
FP32: N * 4 bytes
FP16: N * 2 bytes
INT8: N * 1 byte
INT4: N * 0.5 bytes
```

This does not include the KV cache, activations, temporary buffers, Python/runtime
overhead, or tokenizer. Process RSS is only a rough operational measurement on an
Apple unified-memory system.

## Exercises before Phase 2

1. Run the same greedy command twice. Explain why the response is identical but
   timing can differ.
2. Change `max_new_tokens` from 16 to 64. Observe why total generation time rises.
3. Compare CPU FP32 with MPS FP16 using the same prompt and token limit.
4. Enable sampling, then change temperature while keeping the seed fixed.
5. Print the rendered chat template and identify its system, user, assistant, and
   end-of-message control tokens.
6. Explain why the 107-second first load was excluded from the cached-startup
   baseline: it included network download time.

## Reading

- Qwen2.5-0.5B-Instruct model card:
  <https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct>
- Hugging Face chat templates:
  <https://huggingface.co/docs/transformers/chat_templating>
- Hugging Face generation API:
  <https://huggingface.co/docs/transformers/main_classes/text_generation>
- PyTorch MPS backend:
  <https://docs.pytorch.org/docs/stable/notes/mps.html>
