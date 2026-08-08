"""Console entry point for the Streamlit application."""

import os
import sys
from pathlib import Path


def main() -> None:
    from streamlit.web import cli as streamlit_cli

    app_path = Path(__file__).with_name("streamlit_app.py")
    host = os.getenv("UI_HOST", "0.0.0.0")
    port = os.getenv("UI_PORT", "8501")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        f"--server.address={host}",
        f"--server.port={port}",
        "--server.headless=true",
        "--browser.gatherUsageStats=false",
    ]
    raise SystemExit(streamlit_cli.main())
