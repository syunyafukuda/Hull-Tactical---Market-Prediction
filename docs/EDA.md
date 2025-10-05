# EDA 概要（Hull Tactical Market Prediction）

最終更新: 2025-10-05

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

2) 記述統計・分布確認
	- 数値列の基本統計（平均・分散・歪度・尖度）
	- ターゲットの分布・外れ値確認（winsorize の要否検討）
	- 欠損のパターン（特定列群だけ欠けるか、date_id 依存か）

3) 時系列特性の把握
	- ターゲットの時系列プロット、ローリング平均・分散
	- 自己相関/偏自己相関（ACF/PACF）

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

- [x] 公式 Evaluation/Submission の要点をこのファイルに転記（評価指標・提出形式）
- [x] データ読み込み・構成確認（notebooks/eda.ipynb 実行にて確認）
- [x] 欠損率サマリ関数の用意と全件出力（省略なし表示）
- [x] 記述統計・分布の可視化（ターゲット＋全数値特徴量、ページング表示）
- [x] ターゲットの時系列可視化とローリング指標（shift(1) でリーク防止）
- [x] 自己相関/偏自己相関（ACF/PACF）の診断
- [x] 相関ヒートマップ（高相関上位の列名抽出）
- [x] VIF（多重共線性）診断（上位40列）
- [x] 外れ値検出（IQR / z>3 / MAD>4 の比率集計）
- [x] タイムキー `date_id` の整合性（重複・ギャップ）チェック
- [ ] 簡易ベースラインの構築（CVを時系列で）
- [ ] 提出ファイルの雛形作成（ノート or スクリプト）
- [ ] 高相関・高VIF列の自動削減（しきい値/重要度併用、fold内fit）
- [ ] PCA/PLS 等の次元圧縮（fold内fit）
- [ ] スケーリング・単調変換比較（Standard/Robust/Quantile、log1p/Box-Cox/Yeo-Johnson；fold内fit）
- [ ] レジーム検出（高低ボラ）とモデル比較、時系列CVの詳細設計
- [ ] D*（ダミー群）の寄与評価（Permutation/SHAP）と削減判断
- [ ] 分布シフト検定（train 後半 vs test; KS検定等）と対策
- [ ] 診断用：test 単体の相関ヒートマップ（学習意思決定には未使用）

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

## 7. 実行状況（notebooks/eda.ipynb 2025-10-05）

- データ読み込み/プレビュー: PASS（train/test の全列名表示、head 表示）
- `date_id` 整合性: PASS（重複・欠番チェック、結果「問題なし」）
- 欠損サマリ（train/test）: PASS（train は全件、test は上位）
- 記述統計・分布（ターゲット＋全数値）: PASS（describe 省略なし、ヒストは12列ごとにページング）
- 分散・歪度・尖度（全数値）: PASS（var/std/skew/kurt/欠損率を一覧化）
- 外れ値検出: PASS（IQR / z>3 / MAD>4 の外れ値比率を列ごとに集計）
- ターゲットのローリング統計（5年相当）: PASS（shift(1) でリーク防止）
- ACF/PACF: PASS（ラグ40まで、95%CI付き。上位ラグを標準出力に表示）
- 相関ヒートマップ: PASS（サブセット上限40列、上位相関ペアを出力）
- VIF: PASS（上位40列で算出）
  - サマリ例: `vif>10`: 11 列, `vif>5`: 13 列, `max_vif`: ∞
  - Top by VIF: D1=∞, D2=∞, I5≈1686, I9≈1660, E2≈53, E3≈50, E12≈27, E11≈26, E17≈18, E16≈11.9

補足:
- `lagged_*` 列（test: 3列）はいずれも1日ラグの過去情報（未来情報は含まれない）。
- 目的変数は提供側でデトレンド＋MAD=4ウィンザー化済みで、外れ値影響が抑制されている。

## 8. 主要所見（現時点）

- 欠損: 初期期間に多く、プレフィックス群で似た欠損パターン。連続値はロバスト補完、二値は欠損フラグ併用が有効。
- 分布: 多くがゼロ付近に集中しファットテール・歪みあり。標準化/ロバストスケール/分位変換や単調変換の検討が必要。
- 時系列: ボラが時期で変動（レジーム感）。CVは時系列で、shift(1)+rolling の徹底が必須。
- ACF/PACF: 短期ラグの自己相関の有無を確認（具体の上位ラグはノート出力参照）。ラグ/移動統計の特徴量で捉える方針。
- 相関: 群内で強い相関塊。冗長性が高く、多重共線性対策が必要。
- VIF: D1/D2 が ∞、I5/I9 が極端に高いなど、強い共線性を確認。しきい値削減やPCA/PLSの検討が必要。
- 外れ値: 複数指標で外れ値比率の高い列を特定。列限定のクリッピング/ウィンザー化やロバスト損失の適用候補。

## 9. この EDA から生まれたサブタスク（Backlog）

- [ ] 高相関・高VIF列の削減ルール整備（|r|>0.95 や VIF>10 など）と自動化（fold内fit）
- [ ] PCA/PLS の導入（fold内fit）と性能・解釈の比較
- [ ] スケーリング/単調変換の比較実験（Standard/Robust/Quantile, log1p/Box-Cox/Yeo-Johnson）
- [ ] レジーム検出（高/低ボラ）の定義とサンプルウェイト/二段階学習の比較
- [ ] ラグ/移動統計の特徴拡充（説明変数に対して shift(1)+rolling を体系的に）
- [ ] D*（ダミー群）の寄与評価（Permutation/SHAP）と削減判断
- [ ] 分布シフト診断（train 後半 vs test; KS検定/PSI）と対策
- [ ] ベースライン構築（L1/L2/ElasticNet, 木系モデル）＋時系列CVのプロトタイプ
- [ ] 提出パイプラインの雛形作成（submission.parquet 生成のローカル検証）

