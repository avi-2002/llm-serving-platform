# Phase 8 evaluation baseline analysis

## Automated result

All five API calls completed. Mean concept coverage was 46.7%, mean latency was
3.900 seconds, and p95 latency was 4.324 seconds. MLflow stored the run as
`7999322cc8104608bfdbe4d3f8c5c07b`.

| Case | Concept coverage | Manual observation |
|---|---:|---|
| Latency versus throughput | 66.7% | Broadly correct, but missed service-specific work terminology |
| Dynamic batching | 33.3% | Vague and confused dynamic batching with smaller batches |
| Streaming | 0.0% | Incorrectly claimed that streaming speeds computation |
| Health versus readiness | 100.0% | Covered expected terms, though the 64-token response was truncated |
| Prometheus counter | 33.3% | Failed to clearly state that a counter only increases except for reset |

## The most important lesson

The first automated result reported a 0% hallucination signal, but manual review
found a clearly false streaming claim. The rule searched for one exact forbidden
phrase, while the model expressed the same bad idea using different words.

Therefore `hallucination_signal_rate` means "fraction caught by our current
rules," not "fraction of answers containing hallucinations." The regression
dataset now includes the newly observed bad phrasings. This is evaluation-driven
development: failures discovered by humans become permanent future test cases.

## Baseline conclusion

The 0.5B model is operationally reliable and reasonably fast on this hardware,
but this small evaluation does not support a claim of strong answer quality. Its
46.7% concept coverage and manually identified errors make it a useful serving
baseline, not a trusted technical expert.

The next run should use the updated dataset and a distinct MLflow run name. That
will show how experiment comparison works and should correctly flag the repeated
false statements under deterministic decoding.
