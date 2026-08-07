"""Repeatable API evaluation with transparent scores and MLflow tracking."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean
from typing import Any
from urllib.error import HTTPError, URLError

import mlflow
from mlflow import MlflowClient

from low_latency_llm_serving.benchmark import _post_json, fetch_metadata, percentile


@dataclass(frozen=True)
class EvaluationCase:
    id: str
    prompt: str
    expected_concepts: list[list[str]]
    forbidden_claims: list[str]


@dataclass(frozen=True)
class CaseResult:
    id: str
    prompt: str
    response: str
    success: bool
    latency_seconds: float | None
    output_tokens: int | None
    concept_coverage: float
    matched_concepts: int
    expected_concept_groups: int
    forbidden_claim_matches: list[str]
    hallucination_signal: bool
    error: str | None


def load_dataset(path: Path) -> list[EvaluationCase]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list) or not raw:
        raise ValueError("evaluation dataset must be a non-empty JSON list")
    cases = [EvaluationCase(**item) for item in raw]
    if len({case.id for case in cases}) != len(cases):
        raise ValueError("evaluation case IDs must be unique")
    for case in cases:
        if not case.prompt.strip() or not case.expected_concepts:
            raise ValueError(f"case {case.id} has incomplete expectations")
        if any(not group for group in case.expected_concepts):
            raise ValueError(f"case {case.id} has an empty concept group")
    return cases


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.casefold()).strip()


def score_response(case: EvaluationCase, response: str) -> tuple[float, list[str]]:
    normalized = normalize_text(response)
    matched = sum(
        any(normalize_text(term) in normalized for term in alternatives)
        for alternatives in case.expected_concepts
    )
    forbidden = [
        claim for claim in case.forbidden_claims if normalize_text(claim) in normalized
    ]
    return matched / len(case.expected_concepts), forbidden


def evaluate_case(
    case: EvaluationCase,
    *,
    base_url: str,
    max_new_tokens: int,
    timeout_seconds: float,
    seed: int,
) -> CaseResult:
    try:
        status, body = _post_json(
            f"{base_url.rstrip('/')}/v1/generate",
            {
                "prompt": case.prompt,
                "max_new_tokens": max_new_tokens,
                "do_sample": False,
                "seed": seed,
            },
            f"phase8-{case.id}",
            timeout_seconds,
        )
        if status != 200:
            raise RuntimeError(f"HTTP {status}: {body}")
        response = str(body["response"])
        coverage, forbidden = score_response(case, response)
        return CaseResult(
            id=case.id,
            prompt=case.prompt,
            response=response,
            success=True,
            latency_seconds=float(body["total_request_seconds"]),
            output_tokens=int(body["output_tokens"]),
            concept_coverage=coverage,
            matched_concepts=round(coverage * len(case.expected_concepts)),
            expected_concept_groups=len(case.expected_concepts),
            forbidden_claim_matches=forbidden,
            hallucination_signal=bool(forbidden),
            error=None,
        )
    except (
        HTTPError,
        URLError,
        TimeoutError,
        OSError,
        ValueError,
        KeyError,
        RuntimeError,
    ) as exc:
        return CaseResult(
            id=case.id,
            prompt=case.prompt,
            response="",
            success=False,
            latency_seconds=None,
            output_tokens=None,
            concept_coverage=0.0,
            matched_concepts=0,
            expected_concept_groups=len(case.expected_concepts),
            forbidden_claim_matches=[],
            hallucination_signal=False,
            error=f"{type(exc).__name__}: {exc}",
        )


def summarize(results: list[CaseResult]) -> dict[str, float | int]:
    successful = [result for result in results if result.success]
    latencies = [
        result.latency_seconds
        for result in successful
        if result.latency_seconds is not None
    ]
    return {
        "cases": len(results),
        "successful_cases": len(successful),
        "success_rate": len(successful) / len(results) if results else 0.0,
        "mean_concept_coverage": mean(result.concept_coverage for result in successful)
        if successful
        else 0.0,
        "hallucination_signal_rate": mean(
            float(result.hallucination_signal) for result in successful
        )
        if successful
        else 0.0,
        "mean_latency_seconds": mean(latencies) if latencies else 0.0,
        "p95_latency_seconds": percentile(latencies, 95) if latencies else 0.0,
        "total_output_tokens": sum(result.output_tokens or 0 for result in successful),
    }


def log_to_mlflow(
    result: dict[str, Any],
    *,
    tracking_uri: str,
    experiment_name: str,
    run_name: str,
    dataset_path: Path,
    result_path: Path,
    artifact_location: str | None = None,
) -> str:
    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = (
        experiment.experiment_id
        if experiment is not None
        else client.create_experiment(
            experiment_name, artifact_location=artifact_location
        )
    )
    mlflow.set_experiment(experiment_id=experiment_id)
    with mlflow.start_run(run_name=run_name) as run:
        config, metadata = result["config"], result["model_metadata"]
        mlflow.log_params(
            {
                "model_id": metadata["model_id"],
                "device": metadata["device"],
                "dtype": metadata["dtype"],
                "max_new_tokens": config["max_new_tokens"],
                "seed": config["seed"],
                "dataset": dataset_path.name,
            }
        )
        mlflow.log_metrics(result["summary"])
        mlflow.log_artifact(str(dataset_path), artifact_path="evaluation_inputs")
        mlflow.log_artifact(str(result_path), artifact_path="evaluation_results")
        mlflow.set_tag("evaluation_type", "transparent_code_based_signals")
        return run.info.run_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate the LLM API and record the run in MLflow."
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument(
        "--dataset", type=Path, default=Path("evaluation/phase8_dataset.json")
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("work/phase8-eval.json"))
    parser.add_argument("--experiment", default="llm-serving-evaluation")
    parser.add_argument("--run-name", default="qwen-baseline")
    parser.add_argument("--tracking-uri")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 1 <= args.max_new_tokens <= 512:
        raise ValueError("max-new-tokens must be between 1 and 512")
    cases = load_dataset(args.dataset)
    metadata = fetch_metadata(args.base_url, args.timeout)
    results = [
        evaluate_case(
            case,
            base_url=args.base_url,
            max_new_tokens=args.max_new_tokens,
            timeout_seconds=args.timeout,
            seed=args.seed,
        )
        for case in cases
    ]
    evaluation: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_metadata": metadata,
        "config": {
            "base_url": args.base_url,
            "max_new_tokens": args.max_new_tokens,
            "seed": args.seed,
        },
        "summary": summarize(results),
        "cases": [asdict(result) for result in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evaluation, indent=2) + "\n")
    tracking_uri = args.tracking_uri or f"sqlite:///{Path('work/mlflow.db').resolve()}"
    run_id = log_to_mlflow(
        evaluation,
        tracking_uri=tracking_uri,
        experiment_name=args.experiment,
        run_name=args.run_name,
        dataset_path=args.dataset,
        result_path=args.output,
        artifact_location=Path("work/mlartifacts").resolve().as_uri(),
    )
    summary = evaluation["summary"]
    print("\nPhase 8 evaluation summary")
    print(f"success: {summary['successful_cases']}/{summary['cases']}")
    print(f"mean concept coverage: {summary['mean_concept_coverage']:.1%}")
    print(f"hallucination signal rate: {summary['hallucination_signal_rate']:.1%}")
    print(f"mean latency: {summary['mean_latency_seconds']:.3f}s")
    print(f"p95 latency: {summary['p95_latency_seconds']:.3f}s")
    print(f"MLflow run ID: {run_id}")
    print(f"Raw results: {args.output}")


if __name__ == "__main__":
    main()
