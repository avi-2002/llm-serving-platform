import json

import pytest

from low_latency_llm_serving.api_client import LLMAPIClient


def test_api_client_rejects_non_http_url() -> None:
    with pytest.raises(ValueError, match="http"):
        LLMAPIClient("api:8000")


def test_sse_parser_preserves_event_names_and_json_data() -> None:
    lines = iter(
        [
            "event: start",
            'data: {"request_id": "one"}',
            "",
            b"event: token",
            b'data: {"text": "Hello "}',
            b"",
            "event: token",
            'data: {"text": "world"}',
            "",
            "event: done",
            'data: {"total_request_seconds": 1.25}',
        ]
    )

    events = list(LLMAPIClient._parse_sse(lines))

    assert [event["event"] for event in events] == ["start", "token", "token", "done"]
    assert (
        "".join(event["data"]["text"] for event in events if event["event"] == "token")
        == "Hello world"
    )
    assert events[-1]["data"]["total_request_seconds"] == 1.25


def test_sse_parser_rejects_invalid_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        list(LLMAPIClient._parse_sse(iter(["event: token", "data: nope", ""])))
