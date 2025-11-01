---

欠損構造特徴でやること（網羅）

### 列フラグ
- `m/<col>` = 1 if col is NaN else 0
- `m_any_day` 当日NaN総数、`m_rate_day` 当日NaN比率

### 群集約
- `m_cnt/<grp>`, `m_rate/<grp>`

### 直近観測距離・連続長
- `gap_ffill/<col>` = 直近非NaNからの経過営業日数（先頭は大きめ値にclip、例60）
- `run_na/<col>` 連続NaN長
- `run_obs/<col>` 連続観測長
- `days_since_first_obs/<col>`, `days_since_last_obs/<col>`（未来は見ない）

### ローリング欠損率（過去のみ）
- 窓 `W∈{3,5,10,20,60}`: `roll_m_rate_W/<col>` = 過去W日でのNaN比
- 拡張: `exp_m_rate/<col>` = 0..tの累積NaN比（burn-in=20日）

### パターン変化フラグ
- `na_to_obs/<col>` = 1 if 前日NaNかつ当日観測
- `obs_to_na/<col>` = 1 if 前日観測かつ当日NaN
- `reappear_gap/<col>` = 直前の観測→観測間隔（NaNを跨いだ戻り間隔）

### 代入影響量（ポリシー追跡）
- `imp_used/<col>` = 代入実施フラグ（当日が補完で埋まったか）
- `imp_delta/<col>` = `x_imputed - x_raw`（観測日は0）
- `imp_absdelta/<col>` = `|imp_delta|`
- 代入種別ワンホット（例 `ffill`, `bfill`, `missforest`, `holiday_bridge`）。同日複数不可

### 共欠損構造（群内限定）
- `co_miss_cnt/<a>_<b>` = 当日 a と b 同時NaN のブール
- `co_miss_rolrate_W/<a>_<b>` = 過去W日での同時NaN比
- 次元圧縮案: 欠損フラグ行列の群内PCA成分 `miss_pca1..k`（fitはCV折内）

### 曜日・月次傾向（決定可能情報のみ）
- `dow_m_rate/<col>` = 過去同曜日のNaN比（expanding）
- `month_m_rate/<col>` = 過去同月のNaN比
- 祝日橋渡し×欠損の交差: `holiday_bridge * m/<col>`

### 欠損順位・相対位置
- 当日、群内での「欠けやすさ」順位: `rank_miss_prob_day/<grp>`
- 列内での観測再開の相対位置: `pos_since_reappear/<col>` = `gap_ffill` を `[0,1]` に正規化

### マスク付き統計の下準備用メタ
- `valid_share_W/<col>` = 過去W日での有効観測比
- `stable_window/<col>` = 最小Wで `valid_share_W≥τ` となるW（τ例0.8）
- 下流で「窓不足による不安定」を弾くフィルタ信号

### ヘッド/テイル保護とクリップ
- 先頭連続NaN時: `gap_ffill` は `W_max+1` でクリップ
- すべてNaN列は派生特徴を全てNaNまたは0に固定し除外候補タグ付け
- すべて観測列は `m/*` 系を0に統一

### 推奨パラメータ
- 窓: `W = {3,5,10,20,60}`。`min_periods = W`。
- クリップ: `gap_ffill`, `run_na ≤ 60`。`imp_delta` は ±p99でwinsorize。
- burn-in: 20日。burn-in未満は `exp_*` をNaNのまま。

### ガード（リーク防止）
- すべて「過去のみ」。next非NaNまでの距離のような未来参照は禁止。
- 交差やPCAはCV折ごとにfit→transform。テストは学習統計でtransformのみ。
- 日付派生は決定可能情報のみ。

### 検証
- 単体テスト: 全NaN列、先頭連続NaN、島状NaN の3系で期待値一致。
- 指標: OOFでMSR/Sharpeの差分、予測分布の平均と分散の変化、PSI(train後期 vs test)。
- 採用条件: OOF MSRがベース比 +1σ、LB非劣化。劣化時はロールバック。

### 実行順（軽→重）
- コア: 1,2,3,4,5,9,10
- 拡張: 6,7,8（共欠損と曜日・月傾向、順位系）
- 圧縮: 群内 miss-PCA（必要最低限の成分のみ）

### コスト最適化
- 2値フラグは `uint8`、距離は `int16` に収める。
- ペア特徴は高相関列上位Kのみ。`K=10` 想定。
- 列数が閾値超過時は `valid_share_W` で弱列を先に間引く。

### 提出ユニットとブランチ名例
- **SU1_単列コア**
	- 内容: `m/<col>`, `m_any_day`, `m_rate_day`, `gap_ffill/<col>`, `run_na/<col>`, `run_obs/<col>`, 群集約 `m_cnt/m_rate/<grp>`
	- ブランチ: `feat/miss-core-su1`
	- 提出可否: `ΔMSR ≥ +0.5σ` かつ 分散過度↑なし
	- 備考: 先頭NaN保護と `clip≤60` を明示テスト
- **SU2_履歴率**
	- 内容: `roll_m_rate_W/<col>` `W∈{5,10,20,60}`, `exp_m_rate/<col>`, `valid_share_W/<col>`, `stable_window/<col>`
	- ブランチ: `feat/miss-hist-su2`（設定スイープは `sweep/miss-hist-win`）
	- 提出可否: SU1適用を前提にOOFで `+0.5σ`
	- 備考: burn-in=20未満はNaN据置
- **SU3_遷移・再出現**
	- 内容: `na_to_obs/<col>`, `obs_to_na/<col>`, `reappear_gap/<col>`, `pos_since_reappear/<col>`, `rank_miss_prob_day/<grp>`
	- ブランチ: `feat/miss-transitions-su3`
	- 提出可否: `ΔMSR` が僅少でも予測分散が安定なら提出候補
	- 備考: すべて過去参照のみ
- **SU4_代入影響トレース**
	- 内容: `imp_used/<col>`, `imp_delta/<col>`, `imp_absdelta/<col>`, 代入種別 One-hot（`ffill/bfill/missforest/holiday_bridge`）, 交差 `holiday_bridge * m/<col>`
	- ブランチ: `feat/miss-imptrace-su4`
	- 提出可否: `ΔMSR ≥ +0.5σ` または外れ提出でLB↑
	- 備考: `imp_delta` は ±p99 winsorize
- **SU5_共欠損・交差**
	- 内容: `co_miss_cnt/<a>_<b>`, `co_miss_rolrate_W/<a>_<b>`、群内上位Kペアのみ（`K=10` 目安）
	- ブランチ: `feat/miss-comiss-k10-su5`
	- 提出可否: OOFで `+0.5σ` かつPSI悪化なし
	- 備考: ペア選定は学習期の相関 `|ρ|` 上位→fold内固定
- **SU6_圧縮（必要時のみ）**
	- 内容: 群内欠損フラグ行列PCA `miss_pca1..k`（`k≤3`, fold内fit）
	- ブランチ: `feat/miss-pca-k2-su6`
	- 提出可否: 特徴爆発時のみ適用し `ΔMSR` 非劣化を確認
	- 備考: コンポーネントは学習折のみにfit

### 提出運用ルール
- 1日最大2提出。各SUでOOFスイープ→最良構成のみ提出。
- LB劣化閾値: ベース比 `−0.002` 以下で即リバート。
- 横ばいは保留1回まで。改善なければ落とす。

### フラグ設計例（`configs/preprocess.yaml`）
```
features:
	miss_core: true      # SU1
	miss_hist: false     # SU2
	miss_transitions: false  # SU3
	miss_imptrace: false     # SU4
	miss_comiss: false       # SU5
	miss_pca: false          # SU6
params:
	miss_hist:
		windows: [5,10,20,60]
		burn_in: 20
	miss_core:
		max_gap_clip: 60
	miss_imptrace:
		winsor_p: 0.99
	miss_comiss:
		top_k_pairs: 10
```

### PRテンプレ最小項目
- 目的とリークリスク確認チェック
- 追加特徴一覧と命名規約
- OOF表: MSR/Sharpe/分散/PSI
- 提出判断: 可 or 否（根拠）
- 再現条件: seed, fold境界, 有効フラグ

### アーティファクト命名
- `results/ablation/miss/SU{n}_yyyy-mm-dd.csv`
- `artifacts/features/SU{n}/feature_list.json`

この単位で進めれば、因果の切り出しと提出コストの最適化を両立できる。
