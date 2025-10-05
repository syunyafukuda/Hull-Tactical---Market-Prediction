#!/usr/bin/env bash
set -euo pipefail  # エラーが出たら止める＆未定義変数で停止＆パイプエラー伝播

echo "=== Guard: ensure no data/artifacts are tracked ==="
if git ls-files data/ artifacts/ | grep -qE '.'; then
	echo "Error: Files under data/ or artifacts/ are tracked by Git. Please do not commit them. (データや成果物（data/ または artifacts/ 配下）のファイルがGitに含まれています。コミットしないでください。)" >&2
	git ls-files data/ artifacts/ >&2 || true
	exit 1
fi

echo "=== Running Ruff (lint/format check) ==="
uv run ruff check .

echo "=== Running Pyright (type check) ==="
uv run pyright

echo "=== Running Pytest (unit tests) ==="
uv run pytest --cov=src --cov-report=term-missing
