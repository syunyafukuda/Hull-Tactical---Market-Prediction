"""Pytest 設定: テスト実行時に src を import パスへ追加します。

CI/ローカルの実行環境差異にかかわらず、`from hull_tactical ...` で
インポートできるようにするための最小設定です。
"""

import os
import sys


def _add_src_to_path() -> None:
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)


_add_src_to_path()
