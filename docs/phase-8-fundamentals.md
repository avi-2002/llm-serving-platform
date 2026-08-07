# Phase 8: Evaluation and MLflow experiment tracking

## Why serving speed is not enough

A fast model that gives poor answers is not useful. Phase 8 evaluates performance
and basic quality signals together using the same fixed questions on every run.

The version-controlled dataset contains five questions about concepts this project
implements. Each case defines groups of acceptable terms and known bad claims.

## Scores

- **Success rate:** requests that completed without an API error.
- **Concept coverage:** fraction of expected concept groups represented in the
  response. Alternatives make this less brittle than exact-answer matching.
- **Hallucination signal rate:** responses containing a known forbidden claim.
- **Mean and p95 latency:** response-time measurements for the evaluation set.
- **Output tokens:** generation volume, needed when comparing latency fairly.

These are transparent regression signals, not semantic proof. A response can use
all expected words and still be wrong, or be correct with different wording.
Human review and an independently validated LLM judge are later improvements.

## What MLflow records

Each execution becomes an MLflow run containing:

- parameters: model, device, dtype, generation limit, seed, and dataset;
- metrics: coverage, hallucination signal, success, latency, and token totals;
- artifacts: the exact input dataset and detailed per-case JSON output;
- a run ID so experiments can be compared and reproduced.

The local MLflow metadata is stored in `work/mlflow.db`, while artifacts are kept
under `work/mlartifacts`. Both are intentionally excluded
from Git because it is generated experiment state. Curated conclusions belong in
`benchmarks/`.

## Run the evaluation

Start `llm-api` in one terminal. In a second terminal:

```bash
uv run llm-evaluate \
  --run-name qwen-cpu-baseline \
  --output work/phase8-eval.json
```

Open the tracking UI:

```bash
uv run mlflow ui --backend-store-uri "sqlite:///$PWD/work/mlflow.db" --port 5000
```

Then visit <http://127.0.0.1:5000>, open `llm-serving-evaluation`, and inspect the
run's metrics, parameters, and artifacts.

## Topics to study

- Offline evaluation versus online production monitoring.
- Golden datasets and regression testing.
- Exact match, lexical overlap, semantic similarity, and LLM-as-a-judge.
- Judge bias, self-evaluation bias, and human calibration.
- MLflow experiments, runs, parameters, metrics, tags, and artifacts.
- Quality/latency/cost trade-offs and model promotion thresholds.

## Observed baseline

The Qwen 0.5B CPU run completed 5/5 cases with 46.7% mean concept coverage, 3.900
second mean latency, and 4.324 second p95 latency. Although the first automatic
hallucination signal was 0%, manual review found a false claim that streaming
speeds model computation. The original exact-phrase rule did not recognize the
model's alternative wording.

We added the observed bad phrasings to the dataset for future regression runs.
See `benchmarks/phase8_analysis.md` for the per-case review.

The second MLflow run correctly flagged the streaming and Prometheus-counter
answers, producing a 40% hallucination signal rate while concept coverage remained
46.7%. This means the evaluator improved; it does not mean the deterministic model
became worse between runs.

## Official reading

- MLflow Tracking: <https://mlflow.org/docs/latest/ml/tracking/>
- MLflow LLM evaluation: <https://mlflow.org/docs/latest/genai/eval-monitor/>
- MLflow evaluation datasets: <https://mlflow.org/docs/latest/genai/datasets/>
