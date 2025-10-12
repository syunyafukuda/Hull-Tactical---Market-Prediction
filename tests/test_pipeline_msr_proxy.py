from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_harness(root: Path):
    path = root / "tests" / "common" / "submit_harness.py"
    spec = importlib.util.spec_from_file_location("submit_harness", str(path))
    assert spec and spec.loader, "Failed to load module spec or loader for submit_harness.py"
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_msr_proxy_train_and_predict_cli(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    harness = _load_harness(root)
    train_script = root / "scripts" / "MSR-proxy" / "train_msr_proxy.py"
    predict_script = root / "scripts" / "MSR-proxy" / "predict_msr_proxy.py"

    # ダミーデータ生成
    data_dir = tmp_path / "data"
    harness.make_dummy_data(data_dir)

    # 学習
    artifacts_dir = tmp_path / "artifacts"
    rc_train = harness.run_train_cli(train_script, data_dir=data_dir, out_dir=artifacts_dir)
    assert rc_train == 0
    assert (artifacts_dir / "model_msr_proxy.pkl").exists()
    assert (artifacts_dir / "model_meta.json").exists()

    # 推論
    out_parquet = tmp_path / "submission.parquet"
    out_csv = tmp_path / "submission.csv"
    rc_pred = harness.run_predict_cli(
        predict_script,
        data_dir=data_dir,
        artifacts_dir=artifacts_dir,
        out_parquet=out_parquet,
        out_csv=out_csv,
    )
    assert rc_pred == 0
    assert out_parquet.exists()
    assert out_csv.exists()
