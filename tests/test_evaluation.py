from pathlib import Path

import pytest

from low_latency_llm_serving.evaluation import (
    CaseResult,
    EvaluationCase,
    load_dataset,
    score_response,
    summarize,
)


def test_score_response_matches_concepts_and_forbidden_claims() -> None:
    case = EvaluationCase(
        "batching",
        "What?",
        [["batch", "group"], ["request"], ["together"]],
        ["always faster"],
    )
    coverage, forbidden = score_response(
        case, "Group each request together. It is always faster."
    )
    assert coverage == 1.0
    assert forbidden == ["always faster"]


def test_load_phase8_dataset_has_unique_complete_cases() -> None:
    cases = load_dataset(Path("evaluation/phase8_dataset.json"))
    assert len(cases) == 5
    assert len({case.id for case in cases}) == 5
    assert all(case.expected_concepts for case in cases)


def test_summarize_reports_quality_latency_and_failures() -> None:
    results = [
        CaseResult("one", "one", "answer", True, 1.0, 10, 1.0, 2, 2, [], False, None),
        CaseResult(
            "two", "two", "risky", True, 3.0, 20, 0.5, 1, 2, ["bad claim"], True, None
        ),
    ]
    summary = summarize(results)
    assert summary["success_rate"] == 1.0
    assert summary["mean_concept_coverage"] == 0.75
    assert summary["hallucination_signal_rate"] == 0.5
    assert summary["mean_latency_seconds"] == 2.0
    assert summary["p95_latency_seconds"] == pytest.approx(2.9)
    assert summary["total_output_tokens"] == 30
