from __future__ import annotations

import importlib.util
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd


def load_module_from_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


@contextmanager
def patched_argv(argv: Iterable[str]):
    old = sys.argv[:]  # copy
    sys.argv = [sys.argv[0]] + list(argv)
    try:
        yield
    finally:
        sys.argv = old


def run_train_cli(script_path: Path, *, data_dir: Path, out_dir: Path, extra_args: List[str] | None = None) -> int:
    mod = load_module_from_path("train_mod", script_path)
    args = [
        "--data-dir", str(data_dir),
        "--out-dir", str(out_dir),
    ] + (extra_args or [])
    with patched_argv(args):
        return int(mod.main())


def run_predict_cli(
    script_path: Path,
    *,
    data_dir: Path,
    artifacts_dir: Path,
    out_parquet: Path,
    out_csv: Path | None = None,
    extra_args: List[str] | None = None,
) -> int:
    mod = load_module_from_path("predict_mod", script_path)
    args = [
        "--data-dir", str(data_dir),
        "--artifacts-dir", str(artifacts_dir),
        "--out-parquet", str(out_parquet),
    ]
    if out_csv is not None:
        args += ["--out-csv", str(out_csv)]
    args += (extra_args or [])
    with patched_argv(args):
        return int(mod.main())


def make_dummy_data(dir_path: Path, n: int = 40) -> None:
    dir_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    train = pd.DataFrame({
        "date_id": np.arange(n),
        "f1": rng.normal(size=n),
        "f2": rng.normal(size=n),
        "market_forward_excess_returns": rng.normal(size=n),
    })
    test = pd.DataFrame({
        "date_id": np.arange(n, n + 10),
        "f1": rng.normal(size=10),
        "f2": rng.normal(size=10),
        "is_scored": True,
    })
    train.to_csv(dir_path / "train.csv", index=False)
    test.to_csv(dir_path / "test.csv", index=False)