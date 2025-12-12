from __future__ import annotations

import numpy as np

from scripts.utils_msr import (
    PostProcessParams,
    evaluate_msr_proxy,
    grid_search_msr,
    msr_downside_ratio,
    msr_ratio,
    returns_from_signal,
    to_signal,
    vmsr_ratio,
)


def test_to_signal_and_returns_basic():
    yhat = np.array([-0.1, 0.0, 0.1], dtype=float)
    params = PostProcessParams(mult=1.0, lo=0.8, hi=1.2)
    sig = to_signal(yhat, params)
    # signal = clip(1 + yhat, 0.8, 1.2)
    assert np.allclose(sig, np.clip(1.0 + yhat, 0.8, 1.2))

    ytrue = np.array([0.02, -0.01, 0.03], dtype=float)
    r = returns_from_signal(sig, ytrue)
    # r = (sig-1) * y
    assert np.allclose(r, (sig - 1.0) * ytrue)


def test_msr_variants_monotonicity():
    # 正の平均・適度な分散でMSRが正になること
    rng = np.random.default_rng(0)
    r = rng.normal(loc=1e-3, scale=1e-2, size=1000)
    y = rng.normal(scale=1e-2, size=1000)
    assert msr_ratio(r) > 0
    assert msr_downside_ratio(r) > 0
    # vMSRは lam>0 で分母が増えやすく、lam=0 よりも小さくなりがち
    v0 = vmsr_ratio(r, y, lam=0.0)
    v1 = vmsr_ratio(r, y, lam=0.5)
    assert v1 <= v0 + 1e-12


def test_evaluate_and_grid_search_shapes():
    yhat = np.linspace(-0.01, 0.01, 50)
    ytrue = np.linspace(-0.02, 0.02, 50)
    params = PostProcessParams(mult=1.5, lo=0.9, hi=1.1)
    met = evaluate_msr_proxy(yhat, ytrue, params, eps=1e-8, lam=0.0)
    assert set(["rmse", "msr", "msr_down", "vmsr", "vmsr_lam", "mean", "std", "std_down"]).issubset(met.keys())

    best, rows = grid_search_msr(
        y_pred=yhat,
        y_true=ytrue,
        mult_grid=[0.5, 1.0],
        lo_grid=[0.9, 1.0],
        hi_grid=[1.0, 1.1],
        eps=1e-8,
        optimize_for="msr",
        lam_grid=[0.0, 0.25],
    )
    assert isinstance(best, PostProcessParams)
    assert len(rows) > 0
