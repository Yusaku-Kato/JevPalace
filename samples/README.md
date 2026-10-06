# サンプル集

判定 API（`POST /v1/decide`）を Python から使う例です。
どれも、先に `start-judge.bat` で API を起動してから、リポジトリのルートで実行します。

```bat
judge\.venv\Scripts\python.exe samples\prefectures_winners_or_losers.py
```

| ファイル | 内容 | このサンプルで分かること |
|---|---|---|
| [pm_good_or_evil.py](pm_good_or_evil.py) | 日本の歴代総理大臣 66 人を「善人か悪人か」判定 | 一番シンプルな使い方（1 件ずつ順番に呼ぶ） |
| [world_leaders_good_or_evil.py](world_leaders_good_or_evil.py) | 現在の各国の指導者（約 360 人）を「善人か悪人か」判定 | 外部データ（Wikidata）を取得してキャッシュし、判定にかける |
| [billionaires_genius_or_not.py](billionaires_genius_or_not.py) | 世界長者番付トップ 500 を「天才か凡才か」判定 | `ThreadPoolExecutor` による並列リクエスト |
| [prefectures_winners_or_losers.py](prefectures_winners_or_losers.py) | 47 都道府県の住民を「勝ち組か負け組か」判定 | 並列処理と、1 件ごとの処理時間・往復時間の計測 |
| [benchmark_models.py](benchmark_models.py) | 複数のモデルで上のサンプルを実行し、速度と正答率（16 問）を比較 | `POST /v1/backends/switch` によるモデル切替（例: `benchmark_models.py qwen3-1.7b lfm2.5-1.2b`） |

## 共通のオプション

| オプション | 効果 |
|---|---|
| `--explain` | 判定の理由文も表示します（遅くなります）。 |
| `--reasoning low` | 思考してから判定させます（`low` / `medium` / `high`）。 |
| `--csv out.csv` | 結果を CSV に保存します。 |
| `--workers N` | 同時リクエスト数を指定します（並列版のみ）。既定は API の並列スロット数です。 |
| `--limit N` | 先頭の N 件だけ判定します（件数の多いサンプルのみ）。 |

## 使っている API の機能

```python
r = httpx.post("http://127.0.0.1:8700/v1/decide", json={
    "input": "東京都に住む人",
    "schema": {"verdict": {"type": "choice", "options": {"勝ち組": "…", "負け組": "…"}}},
    "instruction": "入力の都道府県に住む人が、一般的に見て勝ち組か負け組かを判定してください。",
})
d = r.json()
d["verdict"]                                # "勝ち組"
d["fields"]["verdict"]["probabilities"]     # {"勝ち組": 0.997, "負け組": 0.003}
d["meta"]["latency_ms"]                     # 判定にかかった時間
```

- 選択肢に説明を付ける（`options` を辞書にする）と、モデルが判定基準を理解しやすくなります。
- 並列で投げる場合は、スレッドごとに `httpx.Client` を作ります（`billionaires_genius_or_not.py` を参照）。
- このプロジェクトの既定モデル（Bonsai 2）は、並列にしても速くなるのは約 1.4 倍までです。Qwen 系のモデルなら、並列化の効果がもっと大きくなります。

## データの取得元

- **世界の指導者**: [Wikidata](https://www.wikidata.org/) の SPARQL API。初回取得には `--contact`（メールアドレスか URL）が必要です。これは Wikimedia の User-Agent ポリシーの要件です。
- **長者番付**: Forbes のリアルタイム長者番付の非公式 JSON エンドポイント。仕様が予告なく変わる可能性があります。
- 取得したデータは `samples/*_cache.json` に保存されます。git の管理対象外です。

## 注意

これらは API の使い方を示すための例です。
実在の人物や地域についての「善人/悪人」「天才/凡才」「勝ち組/負け組」といった判定は、モデルの主観や世間のステレオタイプを映したものにすぎません。
事実に基づく評価ではないので、結果をそのような評価として扱わないでください。
