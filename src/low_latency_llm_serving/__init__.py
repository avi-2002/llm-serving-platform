"""Learning-first components for a low-latency LLM serving platform.

Heavy inference imports are lazy so the lightweight Streamlit frontend can be
deployed separately from the PyTorch backend.
"""

from typing import Any

__all__ = ["DEFAULT_MODEL_ID", "LocalLLM"]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from low_latency_llm_serving import inference

        return getattr(inference, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
