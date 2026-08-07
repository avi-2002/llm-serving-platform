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

## Regression run v2

The second MLflow run, `02dc7fa7a34b48c2adae2f91df2910e2`, used the expanded
rules. It retained 5/5 request success and 46.7% concept coverage, but now reported
a 40% hallucination signal rate. It correctly flagged the streaming and
Prometheus-counter cases found during manual review.

| Metric | Baseline | Regression v2 |
|---|---:|---:|
| Successful cases | 5/5 | 5/5 |
| Mean concept coverage | 46.7% | 46.7% |
| Hallucination signal rate | 0.0% | 40.0% |
| Mean latency | 3.900 s | 3.672 s |
| p95 latency | 4.324 s | 4.146 s |
| Output tokens | 282 | 282 |

The higher warning rate means the evaluator became more sensitive; the model did
not become worse. The identical token count and concept coverage are consistent
with deterministic greedy responses. The small latency difference is normal
run-to-run variation and is not evidence of an optimization.
