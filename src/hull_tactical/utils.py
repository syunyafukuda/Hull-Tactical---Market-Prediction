from __future__ import annotations


def moving_average(values: list[float] | tuple[float, ...], window: int) -> list[float]:
    """単純移動平均 (SMA) を計算します。

    Args:
        values: 数値のシーケンス。
        window: ウィンドウサイズ（> 0 かつ len(values) 以下）。

    Returns:
        長さが len(values) - window + 1 の移動平均のリスト。

    Raises:
        ValueError: ウィンドウサイズが不正な場合。
    """
    n = len(values)
    if window <= 0:
        raise ValueError("window must be > 0")
    if window > n:
        raise ValueError("window must be <= len(values)")

    # 累積和を用いて O(n) でSMAを計算
    cumsum = [0.0]
    total = 0.0
    for v in values:
        total += float(v)
        cumsum.append(total)

    out: list[float] = []
    for i in range(window, n + 1):
        s = cumsum[i] - cumsum[i - window]
        out.append(s / window)
    return out
