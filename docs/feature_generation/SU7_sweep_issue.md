# SU7 列数・変換セットスイープ ISSUE（案）

## 背景 / Context

- 現状の SU7 ラインは、MSR-proxy の per-feature FI に基づき、以下 8 本のベース列に対して SU7 変換を適用している：
  - `M3, M4, M11, S2, S5, P5, P8, V9`
- 変換セットは SU7 仕様（`docs/feature_generation/SU7.md`）に記載のとおり、1日リターン系列 $r_t$ に対して
  - diff
  - lag
  - rolling mean / rolling std
  - RSI
  - sign
 などの deterministic transform を適用している。
- 直近の SU7 ライン（`src/feature_generation/su7/train_su7.py`）の実行結果は、
  - **RMSE** は SU1+SU5 ベースラインより改善（0.01241 → 0.01208 程度）
  - 一方で **MSR / vMSR** はベースラインの方が高く、SU7 追加によってややディフェンシブなプロファイルになっている
 という状況。
- SU7 の寄与をより適切に評価するために、
  - ベース列の本数（`su7_base_cols` のサイズ）
  - 変換セットの構成（特に RSI / sign / rolling window 等）
 について、いくつかの候補パターンをスイープするラインを設計・実装したい。

## 目的 / Goal

- SU7 の列数・変換セットのバリエーションを複数パターン試し、
  - SU1+SU5 ベースライン（MSR-proxy）
  - 現行 SU7 ライン（`su7_base_cols`=8本, 現行変換セット）
  との比較に加えて、
  - 列数を絞ったパターン（例: 4本, 6本）
  - 列数を広げたパターン（例: 10本, 12本）
  - 変換セットを簡略化 / 強化したパターン
  などを OOF で比較できるようにする。
- 評価指標としては **RMSE を第一優先** としつつ、
  - 補助指標として MSR / vMSR / MSR_down もモニターし、
  - 「RMSE を悪化させずに MSR 系指標が極端に悪化しない」構成を探る。
- SU7 の「MSR / vMSR に対する純粋な寄与」を把握し、
  - 本番採用する SU7 構成（列数・変換セット）を 1〜2 パターンに絞ること
  を目標とする。

## スコープ / Scope

- 対象は **SU7 ラインのみ** とし、SU1, SU5, MSR-proxy 側の仕様・実装は変更しない。
- スイープの対象は主に以下：
  1. `configs/feature_generation.yaml` における `su7.su7_base_cols` の本数
  2. `SU7Config` / `SU7FeatureGenerator` の変換セット構成（ON/OFF や window 長など）
- 学習・評価パイプラインは、基本的に既存の `src/feature_generation/su7/train_su7.py` を再利用し、
  - 設定差分を与えて複数パターンを回すラッパースクリプト、または
  - 追加の CLI オプション（例: `--su7-config-variant`）
  のいずれかで実現することを想定。

## 具体的なタスク / Tasks

1. **スイープ候補パターンの設計**
   - FI ランキング（`artifacts/MSR-proxy/feature_importances_per_feature.csv`）を参照しつつ、
     - 列数を絞るパターンの候補例：
       - Case A: 上位 4 本のみ（例: `M3, M4, S2, P5`）
       - Case B: 上位 6 本のみ（例: `M3, M4, M11, S2, S5, P5`）
     - 列数を広げるパターンの候補例：
       - Case C: 上位 10 本
       - Case D: 上位 12 本
     - 変換セットを変えるパターンの候補例：
       - Case E: RSI を OFF（diff/lag/rolling/sign のみ）
       - Case F: rolling window を長くする or 追加 window を増やす
   - 各ケースについて、想定される列数（`12 * B` 列）と計算コストを簡単に見積もる。

2. **設定の表現方法の設計**
   - `configs/feature_generation.yaml` に SU7 用のバリアントを表現する方法を検討：
     - 例1: `su7` セクション内に `variants:` を追加し、
       - `variants.case_a.su7_base_cols`
       - `variants.case_a.transforms`（RSI ON/OFF 等）
       を記述する。
     - 例2: 別 YAML（例: `configs/su7_sweep.yaml`）にバリアント一覧を持たせる。
   - 最終的にどの方式を採用するかを決め、最小限の schema を docs に記載。

3. **スイープ実行ロジックの実装**
   - アプローチ案 A: 専用スクリプトを追加
    - 例: `src/feature_generation/su7/run_su7_sweep.py`
     - 各バリアントについて：
       1. 設定を読み込み（`feature_generation.yaml` / `su7_sweep.yaml`）
       2. 一時的な `SU7Config` を構築
       3. `train_su7.main(...)` 相当を呼び出し、出力先を `artifacts/SU7/<variant_name>/` に分ける
   - アプローチ案 B: `train_su7.py` にオプション追加
     - 例: `--su7-variant-name` / `--su7-config-override-path`
     - ここでは **既存のシンプルな CLI を壊さないこと** を前提に、追加オプションは任意指定にする。

4. **実行 & アーティファクト整理**
   - 各バリアントについて SU7 ラインを実行し、
     - `model_meta.json`
     - `cv_fold_logs.csv`
     - `oof_predictions.csv`
     を `artifacts/SU7/<variant_name>/` に保存。
   - `model_meta.json` には少なくとも以下を含める：
     - `su7_base_cols`
     - 有効な変換セット（例: `use_rsi: true/false` など）
     - OOF 指標（`oof_rmse`, `oof_best_metrics`）

5. **比較レポートの作成**
   - 簡単な Python スクリプト or Notebook を用意し、
     - MSR-proxy（SU1+SU5 ベースライン）
     - 現行 SU7 ライン
     - 各バリアント（Case A〜F など）
     の
       - `oof_rmse`
       - `msr`
       - `msr_down`
       - `vmsr`
     を一覧表にまとめる。
   - 可能であれば、指標の差分（SU7無し vs SU7有り）を計算して、
     - `Δrmse` / `Δmsr` / `Δvmsr` 等を列として追加。

## DoD（Definition of Done）

- [ ] SU7 の列数・変換セットに関する **スイープ候補パターン** が明文化されていること。
- [ ] `configs/feature_generation.yaml` または専用 YAML で、各バリアントを一意に指定できる設定スキーマが定義されていること。
- [ ] 既存の `src/feature_generation/su7/train_su7.py` を壊さずに、
      SU7 バリアントを複数実行できるスイープ用ロジック（スクリプト or CLI オプション）が実装されていること。
- [ ] 各バリアントについて、`artifacts/SU7/<variant_name>/` 配下に
      `model_meta.json` / `cv_fold_logs.csv` / `oof_predictions.csv` が生成されること。
- [ ] MSR-proxy ベースライン / 現行 SU7 / 各バリアントの OOF 指標を比較する簡易レポート（表 or Notebook）が存在すること。
- [ ] 上記レポートに基づき、「本番で採用を検討したい SU7 構成（列数・変換セット）」が 1〜2 パターン程度に絞り込まれていること。

## 備考 / Notes

- 本 ISSUE はあくまで **SU7 の構成スイープと評価** が目的であり、
  MSR-proxy 側のアルゴリズムや SU1/SU5 の仕様変更はスコープ外とする。
- SU7 の計算コストが重い場合は、
  - スイープ対象のバリアント数を段階的に増やす（まずは 2〜3 パターン）
  - 現行の CV 設定（fold 数や `n_estimators`）を一時的に軽くする
  などで調整してよい。
