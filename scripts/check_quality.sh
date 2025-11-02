#!/usr/bin/env bash
set -e  # エラーが出たら止める

echo "=== Running Ruff (lint/format check) ==="
python3 -m ruff check .

echo "=== Running Pyright (type check) ==="
python3 -m pyright

echo "=== Running Pytest (unit tests) ==="
python3 -m pytest --cov=src --cov-report=term-missing
