from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_streamlit_app_renders_chat_and_controls_without_backend_call() -> None:
    app = AppTest.from_file(
        Path("src/low_latency_llm_serving/streamlit_app.py").resolve(),
        default_timeout=10,
    ).run()

    assert not app.exception
    assert app.title[0].value == "⚡ Low-Latency LLM Serving Platform"
    assert app.chat_input[0].placeholder == "Ask the served model a question"
    assert any(button.label == "Check backend" for button in app.button)
