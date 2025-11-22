# MSR=0問題のトラブルシューティング記録

**発生日**: 2025-11-22  
**影響範囲**: SU3スイープ実行時（全48構成）  
**解決済み**: ✅

---

## 問題の症状

SU3のパラメータスイープ実行後、全48構成で以下の異常が検出された：

```
OOF RMSE: 0.01111～0.01112（正常変動）
OOF MSR: 0.0（全構成で固定）← 異常
```

**期待動作**: MSRは構成間で変動するはず（Sharpe-like指標として性能差を示す）  
**実際の動作**: MSRが全ての構成で0.0を返す

---

## 根本原因

`src/feature_generation/su3/sweep_oof.py` の306行目と322行目で、シグナル後処理パラメータが誤って設定されていた：

```python
# 誤った設定（問題の原因）
signal_params = PostProcessParams(mult=1.0, lo=1.0, hi=1.0)
```

### なぜMSR=0になるのか

MSRの計算ロジック（`scripts/utils_msr.py`）：

```python
# 1. シグナル生成
signal = clip(pred * mult + 1.0, lo, hi)

# 2. トレードリターン計算
r = (signal - 1.0) * target_returns

# 3. MSR計算
msr = mean(r) / (std(r) + eps)
```

**lo=1.0, hi=1.0の場合の挙動**:
- `signal = clip(pred + 1.0, 1.0, 1.0)` → **常に1.0（定数）**
- `r = (1.0 - 1.0) * target = 0` → **常にゼロ（トレードなし状態）**
- `msr = 0 / 0 = 0` → **MSRが無意味な値になる**

シグナルが定数1.0に固定されると「ポジションを取らない」ことと同等になり、トレードリターンが常にゼロ、結果としてMSRも意味をなさない。

---

## 診断プロセス

### 1. 症状確認
```bash
# スイープ結果JSONから統計確認
python tmp/simple_diagnose.py
# → 全48構成でMSR=0.0確認
```

### 2. コード読解
```python
# sweep_oof.py 306行目を確認
signal_params = PostProcessParams(mult=1.0, lo=1.0, hi=1.0)
# ↑ lo=hi=1.0 → シグナルが定数化
```

### 3. 設計意図の確認
```python
# SU2のデフォルト値確認（scripts/utils_msr.py）
@dataclass
class PostProcessParams:
    mult: float = 1.0
    lo: float = 0.0    # デフォルトは0.0
    hi: float = 2.0    # デフォルトは2.0
```

SU2では`grid_search_msr()`で最適パラメータを探索していたが、SU3では固定値を使用していた。しかし、その固定値が誤っていた。

---

## 修正内容

### 変更箇所

**ファイル**: `src/feature_generation/su3/sweep_oof.py`

**変更1: 306行目（fold別MSR計算）**
```python
# 修正前
signal_params = PostProcessParams(mult=1.0, lo=1.0, hi=1.0)

# 修正後
# NOTE: lo=0.0, hi=2.0 を使用（lo=hi=1.0だとシグナルが定数になりMSR=0）
signal_params = PostProcessParams(mult=1.0, lo=0.0, hi=2.0)
```

**変更2: 322行目（OOF全体のMSR計算）**
```python
# 修正前
signal_params = PostProcessParams(mult=1.0, lo=1.0, hi=1.0)

# 修正後
# NOTE: lo=0.0, hi=2.0 を使用（lo=hi=1.0だとシグナルが定数になりMSR=0）
signal_params = PostProcessParams(mult=1.0, lo=0.0, hi=2.0)
```

### 修正根拠

1. **デフォルト値準拠**: `PostProcessParams`のデフォルト値（lo=0.0, hi=2.0）に合わせる
2. **SU2との一貫性**: SU2のgrid_search結果も0.0～2.0の範囲で探索している
3. **シグナル可変性**: lo≠hiにすることで予測値に応じたシグナル変動を許容
4. **MSR計算の正常化**: トレードリターンが変動するため、MSRが意味のある値になる

---

## 検証結果

### 修正前（問題発生時）
```python
# テストスクリプト（tmp/test_msr_fix.py）での確認
PostProcessParams(mult=1.0, lo=1.0, hi=1.0):
  MSR: 0.0000000000
  mean(r): 0.0
  std(r): 0.0
  ✓ 問題を再現
```

### 修正後（正常動作）
```python
PostProcessParams(mult=1.0, lo=0.0, hi=2.0):
  MSR: -0.1186586097
  mean(r): -0.000113
  std(r): 0.000952
  ✓ MSRが正常に計算される
```

### 再スイープ結果
```
構成数: 48
MSR範囲: -0.009764 ～ 0.005772
ベスト構成:
  reappear_top_k=20, temporal_top_k=10, holiday_top_k=10
  OOF RMSE: 0.011107
  OOF MSR: 0.005772
  特徴量数: 444
```

**結論**: ✅ MSRが構成間で変動し、性能差が可視化されるようになった

---

## 教訓と今後の対応

### 1. パラメータのデフォルト値確認の重要性
- データクラスのデフォルト値を確認せず、独自の値を設定した際に問題が発生
- **対策**: デフォルト値から変更する場合は、その理由をコメントで明示する

### 2. SU間の一貫性確保
- SU2では`grid_search_msr()`で最適化、SU3では固定値使用という非対称性
- **対策**: 共通の設定方針を`configs/feature_generation.yaml`で定義し、SU間で統一

### 3. 診断スクリプトの有効性
- 複雑な診断スクリプトよりも、JSONファイルの直接確認が効率的だった
- **対策**: 異常検出時は、まずシンプルなアプローチ（ファイル確認、統計確認）から開始

### 4. テストでのエッジケース確認
- 「lo=hi」のような退化ケースをテストで事前に検出すべきだった
- **対策**: `test_su3.py`に異常パラメータのテストケースを追加

---

## 関連ファイル

### 修正対象
- `src/feature_generation/su3/sweep_oof.py`（306行目、322行目）

### 診断スクリプト
- `tmp/simple_diagnose.py`（スイープ結果の統計確認）
- `tmp/test_msr_fix.py`（修正の検証）

### 参照コード
- `scripts/utils_msr.py`（MSR計算ロジック、PostProcessParamsデフォルト値）
- `src/feature_generation/su2/sweep_oof.py`（SU2の実装、grid_search参考）

### 結果ファイル
- `results/ablation/SU3/sweep_2025-11-22_110535.json`（修正後のスイープ結果）
- `results/ablation/SU3/sweep_summary.csv`（サマリー）

---

## チェックリスト（今後の類似問題防止）

新しいSUや評価スクリプトを実装する際の確認事項：

- [ ] シグナルパラメータ（mult, lo, hi）がデフォルト値と異なる場合、理由をコメント
- [ ] MSR計算を使う場合、lo≠hiであることを確認（退化ケース防止）
- [ ] SU2の`grid_search_msr()`と同じパラメータ範囲を使用しているか確認
- [ ] スイープ結果で全構成のMSRが同じ値になっていないか確認
- [ ] テストに異常パラメータケース（lo=hi, mult=0など）を追加

---

**最終更新**: 2025-11-22  
**担当者**: Feature Generation Team  
**ステータス**: 解決済み、ドキュメント化完了
