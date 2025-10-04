# EDA 概要（Hull Tactical Market Prediction）

最終更新: 2025-10-04

## 1. コンペ概要

- 公式ページ: https://www.kaggle.com/competitions/hull-tactical-market-prediction
- タスク: 市場に関する将来リターンの予測
- データ構造（手元の train/test からの推定）:
  - 時系列キー: `date_id`
  - 特徴量群: `D*`, `E*`, `I*`, `M*`, `P*`, `S*`, `V*` など多数の数値列
  - train のみ: 目的変数候補 `market_forward_excess_returns` または `forward_returns`、関連指標に `risk_free_rate`
  - test: `is_scored`（採点対象フラグ）と `lagged_*` 系の説明変数
- 提出・評価: 評価は RMSE（採点対象は test の is_scored==true 行）。詳細は「1.1 目的変数と評価」を参照。

参考（ローカルのヘッダ確認による件数）:

- train 列総数: 98（`date_id` とターゲット/指標列を含む）
- 特徴量群の件数（先頭文字プレフィクス別）:
	- D: 9, E: 20, I: 9, M: 18, P: 13, S: 12, V: 13
- test 列総数: 99、`lagged_*` 列は 3、本番採点対象フラグ `is_scored` を含む

データページ: https://www.kaggle.com/competitions/hull-tactical-market-prediction/data

注意: 公式規約により、生データの再配布は禁止。Kaggle 提出時は `/kaggle/input/hull-tactical-market-prediction/...` を参照。

### 1.1 目的変数と評価

- 目的変数: `market_forward_excess_returns`
	- 定義の概念整理:
		- 翌日リターンを $r_{t+1}$（データ上は `forward_returns`）、リスクフリーを $rf_{t+1}$（`risk_free_rate`）とする。
		- 超過リターンは $r^{excess}_{t+1} = r_{t+1} - rf_{t+1}$。
		- 長期トレンドを除くため、5年ローリング平均（過去のみ）を $\mathrm{MA}_{5y}(\cdot)$ とし、
			\[ \tilde{y}_{t+1} = r^{excess}_{t+1} - \mathrm{MA}_{5y}(r^{excess}) \]
		- 外れ値の影響を抑えるため、中央値 $m$ と $\mathrm{MAD}$（Median Absolute Deviation）を用い、$k=4$（MAD=4）でウィンザー化：
			- 下限/上限を \( L = m - 4\,\mathrm{MAD} \), \( U = m + 4\,\mathrm{MAD} \) とし、
			- \( y_{t+1} = \mathrm{clip}(\tilde{y}_{t+1}, L, U) \) として境界に置換（除外ではなく「丸め」）。
	- 実務メモ: 上記処理はデータ提供側で反映済みであり、学習では `market_forward_excess_returns` をそのまま目的変数として使用すればよい。

- 評価指標: RMSE（Root Mean Squared Error）
	- 採点対象は test のうち `is_scored == true` の行。
	- 数式: \( \mathrm{RMSE} = \sqrt{\tfrac{1}{N} \sum_{i=1}^{N} (\hat{y}_i - y_i)^2} \)
	- 参考: Public LB は一部期間のコピーを用いる旨の注意がある（詳細は公式 Evaluation を参照）。

- リーク防止の注意
	- 5年平均や中央値/MADなど、時点 $t$ の統計量は必ず「過去のみ」を用いて算出すること（例: `shift(1)` + `rolling`/`expanding`）。
	- 例（概念）: \( m_t = \mathrm{median}(\{y_s\}_{s \le t-1}) \), \( \mathrm{MAD}_t = \mathrm{median}(|y_s - m_t|) \, (s \le t-1) \)
	- `lagged_*` は過去情報であることを確認し、学習時の特徴量作成も「未来情報を見ない」ように徹底する。

## 2. データ運用ポリシー（本リポジトリ）

- 競技データは Kaggle API で取得し、Git には含めない。
- `data/` 配下でローカル管理。毎回スクリプトで再現。
- 取得: `./scripts/fetch_data.sh`
- `.gitignore` にて `data/` と `artifacts/` を除外。CI/ローカルの品質チェックで誤コミットを検知し失敗させる。

## 3. EDA でやるべきこと（大枠）

1) データ確認・基本整形
	- 列名・型・欠損率の把握（train/test 両方）
	- タイムキー `date_id` の整合性、重複・ギャップ確認
	- ラベル（`market_forward_excess_returns` もしくは `forward_returns`）の確定と存在確認

2) 記述統計・分布確認
	- 数値列の基本統計（平均・分散・歪度・尖度）
	- ターゲットの分布・外れ値確認（winsorize の要否検討）
	- 欠損のパターン（特定列群だけ欠けるか、date_id 依存か）

3) 時系列特性の把握
	- ターゲットの時系列プロット、ローリング平均・分散
	- 自己相関/偏自己相関（ACF/PACF）
	- regime らしき変化点の有無（簡易的なCUSUMやローリング相関など）

4) 相関・多重共線性
	- 特徴量群ごとの相関ヒートマップ（高相関の把握）
	- VIF の簡易確認（必要なら）

5) リーク回避の観点
	- `lagged_*` 系の説明変数とターゲットのタイムアライメント確認
	- 学習/検証分割は時系列で行う（ランダム分割は禁止）

6) 検証戦略の案（ドラフト）
	- 時系列K-Fold（例: expanding window / rolling window）
	- 期間ごとの安定性評価（スコアのドリフト検知）

7) スケーリング・変換・前処理の仮説
	- ログ変換、標準化、クリッピングの要否
	- カテゴリ列があればエンコーディング方針（現状は数値列主体）

8) ベースライン作成の準備
	- 簡易モデル（線形回帰 / ラッソ / 木系）のベースライン
	- 最小の特徴量セットでのサニティチェック

9) 提出のドライラン
	- 公式フォーマットでの出力確認（列名・ファイル名・順序）
	- Private/Public の挙動注意（リーク検査）

## 4. 実装タスク（チェックリスト）

- [ ] 公式 Evaluation/Submission の要点をこのファイルに転記（評価指標・提出形式）
- [ ] `scripts/fetch_data.sh` 実行 → `data/raw/` の構成確認をノートに記録
- [ ] 欠損率サマリ関数を notebooks か `src/hull_tactical/` に用意
- [ ] 記述統計・分布の可視化（ヒスト/ KDE / 箱ひげ）
- [ ] ターゲットの時系列可視化とローリング指標
- [ ] 相関ヒートマップ（高相関上位の列名抽出も）
- [ ] 簡易ベースラインの構築（CVを時系列で）
- [ ] 提出ファイルの雛形作成（ノート or スクリプト）

## 5. 参照

- README の「データ運用ポリシー（EDA）」「品質チェック（CI相当）」「ブランチ戦略」
- ノート: `notebooks/eda.ipynb`

## 6. 公式評価コンポーネントの入出力仕様（概要）

ソース: `data/raw/kaggle_evaluation/`

- Gateway の役割（`default_gateway.py` / `core/templates.py` / `core/base_gateway.py`）
	- `test.csv` を読み込み、行ID列（デフォルトは `batch_id`。存在しない場合は test の先頭列名）でバッチ分割し、各バッチを `predict` に送る。
	- 受け取った予測の行数が、対応する row_ids の行数と一致するかを検証（不一致はエラー）。
	- 予測と row_ids を結合して `submission.parquet` を出力。
		- 行ID列は最初の列に配置される（複数列可）。
		- 目的列名は、予測が無名 Series/DF の場合は Gateway 側の `target_column_name` が適用される。デフォルトは `prediction`。
		- pandas / polars どちらもサポート（内部は Parquet/LZ4）
	- レスポンスタイムアウト（1バッチ）: 5分（`set_response_timeout_seconds(60*5)`）。
	- サーバ起動猶予: 15分（`STARTUP_LIMIT_SECONDS`）。

- InferenceServer（`default_inference_server.py` / `core/templates.py` / `core/relay.py`）
	- 参加者は `predict` エンドポイントを実装（`InferenceServer` に登録）。
	- Gateway から gRPC でデータを受け取り、同じ行数の予測を返す必要がある。

- 評価との関係
	- Gateway は全 test 行に対する `submission.parquet` を作る。実際のスコア計算はプラットフォーム側で `is_scored == true` の行に対して RMSE を算出。
	- したがって、提出では「全行」分の予測を用意し、評価側で `is_scored` により自動的にフィルタされる想定。

- 実装メモ
	- ローカルで I/O を模擬する場合、行ID列（`batch_id` または先頭列名）を先頭に、予測列（`prediction`）を続けた Parquet を作れば整合。
	- 予測のデータ型は DataFrame/Series（pandas or polars）を推奨。Series なら列名を明示するか、Gateway の `target_column_name` に依存（デフォルト `prediction`）。

