"""注意: このファイルは暫定のモックテストです。
将来的に本実装のテストへ置き換えるか、不要であれば削除してください。
"""

from hull_tactical.utils import moving_average


def test_moving_average_happy_path():
    # ハッピーパス: 単純移動平均の基本計算を検証
    assert moving_average([1, 2, 3, 4, 5], window=3) == [2.0, 3.0, 4.0]


def test_moving_average_errors():
    # エラーパス1: window <= 0 のときは ValueError
    try:
        moving_average([1, 2], window=0)
        assert False, "window <= 0 の場合は ValueError が発生すべき"
    except ValueError:
        pass

    # エラーパス2: window > len(values) のときは ValueError
    try:
        moving_average([1, 2], window=3)
        assert False, "window > len(values) の場合は ValueError が発生すべき"
    except ValueError:
        pass
