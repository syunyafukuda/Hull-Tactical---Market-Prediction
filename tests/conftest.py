"""Pytest 設定: テスト実行時に src を import パスへ追加します。

CI/ローカルの実行環境差異にかかわらず、`from hull_tactical ...` で
インポートできるようにするための最小設定です。
"""

import os
import sys

# Debug: Print to check if conftest is being executed
# print(f"[conftest.py] Executing conftest.py, sys.path before: {sys.path[:2]}")


def _add_src_to_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
        # print(f"[conftest.py] Added {src} to sys.path")
    # else:
        # print(f"[conftest.py] {src} already in sys.path")


_add_src_to_path()
# print(f"[conftest.py] sys.path after: {sys.path[:2]}")
