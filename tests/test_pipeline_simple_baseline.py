import importlib.util
from pathlib import Path


def _load_harness(root: Path):
    path = root / "tests" / "common" / "submit_harness.py"
    spec = importlib.util.spec_from_file_location("submit_harness", str(path))
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[attr-defined]
    return mod


def test_simple_baseline_train_and_predict_cli(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    harness = _load_harness(root)
    train_script = root / "scripts" / "simple_baseline" / "train_simple.py"
    predict_script = root / "scripts" / "simple_baseline" / "predict_simple.py"

    # ダミーデータ生成
    data_dir = tmp_path / "data"
    harness.make_dummy_data(data_dir)

    # 学習
    artifacts_dir = tmp_path / "artifacts"
    rc_train = harness.run_train_cli(train_script, data_dir=data_dir, out_dir=artifacts_dir)
    assert rc_train == 0
    assert (artifacts_dir / "model_simple.pkl").exists()
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
