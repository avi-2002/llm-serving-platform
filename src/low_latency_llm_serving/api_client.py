"""Small HTTP client shared by the Streamlit UI and its tests."""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

import requests


class APIClientError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class StreamingUnavailable(APIClientError):
    """Raised when a backend does not expose the optional streaming route."""


class LLMAPIClient:
    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 180.0,
        session: requests.Session | None = None,
    ) -> None:
        base_url = base_url.rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            raise ValueError("API URL must start with http:// or https://")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = base_url
        self.timeout_seconds = timeout_seconds
        self.session = session or requests.Session()

    def ready(self) -> dict[str, Any]:
        try:
            response = self.session.get(
                f"{self.base_url}/ready", timeout=min(self.timeout_seconds, 10)
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise self._request_error("readiness check failed", exc) from exc

    def generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> dict[str, Any]:
        try:
            response = self.session.post(
                f"{self.base_url}/v1/generate",
                json=self._payload(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                ),
                headers={"X-Request-ID": f"ui-{uuid4()}"},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            raise self._request_error("generation request failed", exc) from exc

    def stream_generate(
        self,
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> Iterator[dict[str, Any]]:
        try:
            response = self.session.post(
                f"{self.base_url}/v1/generate/stream",
                json=self._payload(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    do_sample=do_sample,
                    temperature=temperature,
                    top_p=top_p,
                    seed=seed,
                ),
                headers={
                    "Accept": "text/event-stream",
                    "X-Request-ID": f"ui-stream-{uuid4()}",
                },
                timeout=self.timeout_seconds,
                stream=True,
            )
            if response.status_code == 404:
                response.close()
                raise StreamingUnavailable(
                    "this backend does not provide /v1/generate/stream",
                    status_code=404,
                )
            response.raise_for_status()
        except StreamingUnavailable:
            raise
        except requests.RequestException as exc:
            raise self._request_error("streaming request failed", exc) from exc

        try:
            yield from self._parse_sse(response.iter_lines(decode_unicode=True))
        finally:
            response.close()

    @staticmethod
    def _parse_sse(lines: Iterator[str | bytes]) -> Iterator[dict[str, Any]]:
        event_name = "message"
        data_lines: list[str] = []
        for raw_line in lines:
            line = raw_line.decode() if isinstance(raw_line, bytes) else raw_line
            if line == "":
                if data_lines:
                    yield {
                        "event": event_name,
                        "data": json.loads("\n".join(data_lines)),
                    }
                event_name = "message"
                data_lines = []
            elif line.startswith("event:"):
                event_name = line.removeprefix("event:").strip()
            elif line.startswith("data:"):
                data_lines.append(line.removeprefix("data:").lstrip())
        if data_lines:
            yield {"event": event_name, "data": json.loads("\n".join(data_lines))}

    @staticmethod
    def _payload(
        prompt: str,
        *,
        max_new_tokens: int,
        do_sample: bool,
        temperature: float,
        top_p: float,
        seed: int,
    ) -> dict[str, object]:
        return {
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "seed": seed,
        }

    @staticmethod
    def _request_error(message: str, exc: requests.RequestException) -> APIClientError:
        response = getattr(exc, "response", None)
        status_code = response.status_code if response is not None else None
        detail = response.text if response is not None else str(exc)
        return APIClientError(
            f"{message}: {detail or type(exc).__name__}", status_code=status_code
        )
