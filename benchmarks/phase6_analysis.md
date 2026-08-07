# Phase 6 streaming analysis

## Result

The local Qwen streaming request produced its first visible text after 0.308
seconds and completed after 4.381 seconds.

| Measurement | Result |
|---|---:|
| Time to first displayed chunk | 0.308 s |
| Total request time | 4.381 s |
| Text appeared before completion by | 4.073 s |
| First chunk as portion of total time | 7.0% |

## Layman's interpretation

Without streaming, the screen would remain blank for the full 4.38 seconds.
With streaming, the user sees the answer begin after 0.31 seconds and can start
reading while the remaining text is generated.

Streaming did not make the model finish sooner. It changed when completed pieces
were delivered to the user, greatly improving perceived responsiveness.

## Measurement caution

Hugging Face's text streamer emits printable text fragments, which may contain
one or several model tokens. Therefore this implementation measures time to first
displayed chunk, not exact time to first model token. Exact TTFT requires
instrumenting the token-generation loop beneath text decoding.

This was one functional smoke test. Stable performance claims require repeated
runs, percentile calculations, and a comparison with the non-streaming route
using identical prompts and settings.
