"""Streamlit Community Cloud entry point for the frontend-only deployment."""

import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
runpy.run_module("low_latency_llm_serving.streamlit_app", run_name="__main__")
