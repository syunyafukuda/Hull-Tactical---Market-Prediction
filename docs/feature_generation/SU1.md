---

# 欠損構造特徴でやること（網羅）

## SU1 運用ルール
- ① SU1 は生データを入力とする（欠補や変換より先に適用）
- ② 未来参照は禁止（`t+1` 以降を使う距離・補間を作らない）
- ③ 距離・連続長の `clip` は 60 以下に固定
- ④ 全 NaN 列／先頭連続 NaN を保護し、派生特徴は NaN もしくは 0 に統一
- ⑤ 全グループ（M/E/I/P/S/V）を対象とする
- ⑥ 列名は先頭が英大文字連続＋数字（例: `M1`, `V10`）で始まる前提。異なる命名は個別マッピングを追加する。

## 適用ポリシー

### 列フラグ
- `m/<col>` = 当日がNaNなら1、観測なら0。uint8。全特徴量列（M/E/I/P/S/V）に適用。
- `m_any_day` 当日NaN総数
- `m_rate_day` 当日NaN比率
- `m_cnt/ALL`, `m_rate/ALL` = 監視用の全列集計（`m_any_day`, `m_rate_day` の別名）

### 群集約
- `m_cnt/<grp>`　当日、群（M/E/I/P/S/V）内でNaNの列本数。群に属する全列の m/<col>（NaN=1）を行方向に合計。
- `m_rate/<grp>` 当日、群内でNaNとなっている列の比率。m_cnt/<grp> を 群内列数で割る。m_rate/<grp> = m_cnt/<grp> / n_cols_grp。
- `avg_gapff/<grp>` 当日、群内の「直近非NaNからの距離（gap_ffill）」の平均。群内の全 gap_ffill/<col> を行方向に平均。全NaN列は平均計算から除外。clip≤60。
- `avg_run_na/<grp>`当日、群内の「連続NaN長（run_na）」の平均。群内の全 run_na/<col> を行方向に平均。全NaN列は平均計算から除外。clip≤60。

### 直近観測距離・連続長
- `gap_ffill/<col>` = 直近非NaNからの経過営業日数。先頭で観測が無い区間は上限60でクリップ。型int16。欠損を1、観測を0とするランレングス更新。観測日に0、NaN連続中は前日値+1。先頭未観測区間は60固定。未来の非NaNまでの距離は使わない。
- `run_na/<col>` 連続NaN長。当日まで連続してNaNである長さ。int16、上限60。当日NaNならprev+1、観測なら0にリセット。
- `run_obs/<col>` 連続観測長。int16、上限60。当日観測ならprev+1、NaNなら0にリセット。

## ガード（リーク防止）
- すべて「過去のみ」。next非NaNまでの距離のような未来参照は禁止。
- 交差やPCAはCV折ごとにfit→transform。テストは学習統計でtransformのみ。
- 日付派生は決定可能情報のみ。

## 検証
- 単体テスト: 全NaN列、先頭連続NaN、島状NaN の3系で期待値一致。
- 指標: OOFでMSR/Sharpeの差分、予測分布の平均と分散の変化、PSI(train後期 vs test)。
- 採用条件: OOF MSRがベース比 +1σ、LB非劣化。劣化時はロールバック。

---

## トラブルシューティング

### MSR=0問題（2025-11-22解決）

**症状**: スイープ実行時に全構成でMSR=0.0となる異常

**原因**: シグナルパラメータの誤設定
```python
# 問題のコード
signal_params = PostProcessParams(mult=1.0, lo=1.0, hi=1.0)
# → lo=hi=1.0でシグナルが定数1.0に固定
# → r=(signal-1.0)*target=0（常にゼロ）
# → MSR=0/0=0
```

**修正**:
```python
# 正しい設定（デフォルト値準拠）
signal_params = PostProcessParams(mult=1.0, lo=0.0, hi=2.0)
# → シグナルが[0.0, 2.0]の範囲で可変
# → トレードリターンが変動し、MSRが正常計算される
```

**教訓**:
1. シグナルパラメータ（mult, lo, hi）がデフォルトと異なる場合は理由をコメント
2. lo≠hiであることを確認（退化ケース防止）
3. スイープ結果で全構成のMSRが同じ値になっていないか確認

詳細は [`docs/feature_generation/troubleshooting/MSR_zero_issue.md`](./troubleshooting/MSR_zero_issue.md) を参照。

