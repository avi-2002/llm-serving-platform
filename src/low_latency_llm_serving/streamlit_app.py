"""Interactive Streamlit frontend for the LLM serving API."""

import os
from typing import Any

import streamlit as st

from low_latency_llm_serving.api_client import (
    APIClientError,
    LLMAPIClient,
    StreamingUnavailable,
)

st.set_page_config(page_title="LLM Serving Platform", page_icon="⚡", layout="wide")


def configured_api_url() -> str:
    """Read local environment first, then a Streamlit Cloud secret."""
    if value := os.getenv("LLM_API_URL"):
        return value
    try:
        if value := st.secrets.get("LLM_API_URL"):
            return str(value)
    except FileNotFoundError:
        pass
    return "http://127.0.0.1:8000"


def metric_caption(metrics: dict[str, Any]) -> str:
    parts = []
    mappings = (
        ("time_to_first_chunk_seconds", "first text", "s"),
        ("generation_seconds", "generation", "s"),
        ("total_request_seconds", "total", "s"),
        ("tokens_per_second", "tokens/s", ""),
        ("output_tokens", "output tokens", ""),
        ("batch_size", "batch", ""),
    )
    for key, label, suffix in mappings:
        value = metrics.get(key)
        if value is not None:
            formatted = f"{value:.3f}" if isinstance(value, float) else str(value)
            parts.append(f"{label}: {formatted}{suffix}")
    if replica := metrics.get("replica_id"):
        parts.append(f"replica: {str(replica)[:8]}")
    return " · ".join(parts)


def generation_settings() -> dict[str, Any]:
    with st.sidebar:
        st.header("Serving controls")
        api_url = st.text_input(
            "Backend API URL",
            value=configured_api_url(),
        )
        max_new_tokens = st.slider("Maximum new tokens", 8, 256, 64, 8)
        do_sample = st.toggle("Enable sampling", value=False)
        temperature = st.slider(
            "Temperature", 0.1, 2.0, 0.7, 0.1, disabled=not do_sample
        )
        top_p = st.slider("Top-p", 0.1, 1.0, 0.9, 0.05, disabled=not do_sample)
        seed = st.number_input("Seed", min_value=0, value=42, step=1)
        streaming = st.toggle(
            "Stream response",
            value=True,
            help="The local FastAPI backend streams. Ray Serve falls back to a completed response.",
        )
        if st.button("Check backend", use_container_width=True):
            try:
                ready = LLMAPIClient(api_url).ready()
                st.success(f"Ready: {ready['model_id']}")
            except (APIClientError, ValueError) as exc:
                st.error(str(exc))
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    return {
        "api_url": api_url,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "temperature": temperature,
        "top_p": top_p,
        "seed": int(seed),
        "streaming": streaming,
    }


def complete_response(
    client: LLMAPIClient, prompt: str, settings: dict[str, Any]
) -> tuple[str, dict[str, Any]]:
    result = client.generate(
        prompt,
        max_new_tokens=settings["max_new_tokens"],
        do_sample=settings["do_sample"],
        temperature=settings["temperature"],
        top_p=settings["top_p"],
        seed=settings["seed"],
    )
    return str(result["response"]), result


st.title("⚡ Low-Latency LLM Serving Platform")
st.caption(
    "Chat with the served Qwen model and inspect latency, throughput, batching, "
    "and Ray replica telemetry. Each message is currently an independent request."
)

settings = generation_settings()
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if caption := metric_caption(message.get("metrics", {})):
            st.caption(caption)

if prompt := st.chat_input("Ask the served model a question", max_chars=8_000):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        placeholder = st.empty()
        response_text = ""
        metrics: dict[str, Any] = {}
        try:
            client = LLMAPIClient(settings["api_url"])
            if settings["streaming"]:
                try:
                    for item in client.stream_generate(
                        prompt,
                        max_new_tokens=settings["max_new_tokens"],
                        do_sample=settings["do_sample"],
                        temperature=settings["temperature"],
                        top_p=settings["top_p"],
                        seed=settings["seed"],
                    ):
                        if item["event"] == "token":
                            response_text += str(item["data"]["text"])
                            placeholder.markdown(response_text + "▌")
                        elif item["event"] == "done":
                            metrics = item["data"]
                        elif item["event"] == "error":
                            raise APIClientError(str(item["data"].get("detail")))
                except StreamingUnavailable:
                    st.info("Ray backend detected; using completed-response mode.")
                    response_text, metrics = complete_response(client, prompt, settings)
            else:
                response_text, metrics = complete_response(client, prompt, settings)
            placeholder.markdown(response_text)
            if caption := metric_caption(metrics):
                st.caption(caption)
            st.session_state.messages.append(
                {"role": "assistant", "content": response_text, "metrics": metrics}
            )
        except (APIClientError, ValueError, KeyError) as exc:
            placeholder.empty()
            st.error(f"The backend request failed: {exc}")
