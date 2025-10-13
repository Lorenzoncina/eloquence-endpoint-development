#!/usr/bin/env bash
set -euo pipefail

# deps are already installed in the container, run uvicorn directly
python whisper_api_python.py