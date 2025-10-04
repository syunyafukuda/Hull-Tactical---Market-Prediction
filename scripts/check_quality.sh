#!/usr/bin/env bash
set -e  # エラーが出たら止める

echo "=== Running Ruff (lint/format check) ==="
uv run ruff check .

echo "=== Running Pyright (type check) ==="
uv run pyright

echo "=== Running Pytest (unit tests) ==="
uv run pytest --cov=src --cov-report=term-missing
