# JevPalace Decision API

[Jev](https://jevai.net/) 風の判定 API を、ローカル LLM（既定: Ternary-Bonsai-2-27B）で動かすものです。
入力と、あらかじめ型を決めた `schema` を受け取り、文章は生成せず、型どおりの値と確率・確信度だけを返します。

```
POST /v1/decide
{
  "input": "先月分が二重に請求されています。返金してください。",
  "schema": {"route": ["billing", "technical", "sales"], "score": "number"}
}
→ {"route": "billing", "score": 0.9962, "confidence": 0.997, ...}      (約 0.6 秒)
```

- **型の安全性**: JSON スキーマで出力を制約しているため、選択肢にない値や型違いは出ません。
- **確率**: モデルの自己申告ではなく、トークンの logprobs から計算しています（例: 「billing」を選ぶ確率が 99.7%）。
- **画像対応のオン/オフ**: オフの時は画像用モジュール（mmproj）を読み込まず、空いた VRAM を並列処理に回します。
- **モデルの差し替え**: `config.yaml` の `backends` に追加するだけで、GGUF でも OpenAI 互換 API でも使えます。

## 起動

```bat
start-judge.bat              rem テキスト専用で起動（高速、既定）
start-judge.bat --vision     rem 画像対応で起動
start-judge.bat --backend X  rem config.yaml の別バックエンドで起動
```

- Web UI: http://127.0.0.1:8700 （schema を書いて試す、画像対応の切替、モデルの切替ができます）
- Swagger: http://127.0.0.1:8700/docs
- llama-server（ポート 8090）は API が自動で起動・停止します。API を強制終了しても残りません。

## schema の書き方

| 書き方 | 種類 | 返る値 | `fields.<名前>` の中身 |
|---|---|---|---|
| `["a", "b", "c"]` | Choice（選択） | 選んだ選択肢 | `probabilities`: 各選択肢の確率、`confidence` = 選んだ選択肢の確率 |
| `{"type": "choice", "options": {"a": "説明", ...}}` | Choice（選択肢に説明を付ける） | 同上 | 同上 |
| `"boolean"` | Binary（二値） | `true` / `false` | `probability`: true の確率、`confidence` = max(p, 1-p) |
| `"number"` | Score（0〜1） | 0〜1 の小数 | 内部では 0〜10 の 11 段階で答えさせ、その期待値を 0〜1 に変換します。`distribution` に各段階の確率、`confidence` = 選んだ段階 ±1 に入る確率 |
| `{"type": "score", "min": 1, "max": 5}` | Score（任意の段階） | 整数 | 上と同じ形式に加えて `expected`（期待値）を返します |
| `"string"` | 抽出 | 文字列（該当なしは `""`） | `confidence` = その文字列を生成した確率 |

どの型も `{"type": ..., "description": "このフィールドで何を判定するか"}` の形で説明を付けられます。
説明を付けると精度が上がります。

## レスポンス

```json
{
  "route": "billing",
  "score": 0.9962,
  "confidence": 0.997,
  "abstain": false,
  "fields": {
    "route": {"value": "billing", "probabilities": {"billing": 0.997, "sales": 0.002, "technical": 0.001}, "confidence": 0.997},
    "score": {"value": 0.9962, "distribution": {"0": 0.0, "...": 0.0, "10": 0.962}, "confidence": 1.0}
  },
  "meta": {"model": "...", "latency_ms": 631, "confidence_source": "logprobs", "...": "..."}
}
```

- トップレベルの値は Jev と同じ形です（各フィールドの値と `confidence`）。
- トップレベルの `confidence` は、全フィールドの確信度の最小値（一番自信のないフィールドの値）です。
- `min_confidence` を指定すると、`confidence` がそれ未満の時に `abstain: true` を返します（スマートな if 文として使う場合の「保留」）。
- `confidence`, `fields`, `abstain`, `reason`, `meta`, `thinking` はフィールド名に使えません。

## リクエストのフィールド

| フィールド | 説明 |
|---|---|
| `input` | 判定対象。文字列でも、状態を表す JSON オブジェクトでも渡せます。 |
| `schema` | 出力フィールドの型（上の表を参照）。 |
| `instruction` | 判定の観点や質問（任意）。例: 「入力は敬語で書かれているか」 |
| `preset` | `judges/*.yaml` を読み込みます（schema・instruction・例）。リクエスト側の指定で上書きできます。 |
| `examples` | few-shot 例。形式は `[{"input": ..., "output": {...}}]` です。 |
| `images` | 画像（data URL / base64 / http(s) URL）。画像モードの時だけ使えます。オフの時に送ると 400 を返します。 |
| `explain` | `true` にすると理由文も返します（拡張機能。既定はオフ、遅くなります）。 |
| `reasoning` | `off`（既定）/ `low` / `medium` / `high`。思考させると精度が上がりますが、遅くなります。 |
| `min_confidence` | abstain の閾値。 |

## その他のエンドポイント

| エンドポイント | 説明 |
|---|---|
| `POST /v1/systemone` | `/v1/decide` の別名です。 |
| `POST /v1/decide/batch` | `{"defaults": {...共通値}, "items": [{...}, ...]}` の形で送ると並列に判定します。 |
| `POST /v1/decide/upload` | multipart で画像ファイルを送ります（`preset`, `input`, `files`, `options`=残りの項目の JSON）。 |
| `POST /v1/vision` | `{"enabled": true/false}` で画像対応を切り替えます。llama-server を再起動するため約 5 秒かかります。 |
| `GET /v1/backends`, `POST /v1/backends/switch` | `{"name": "..."}` でモデルを切り替えます。失敗した場合は元のモデルに戻ります。 |
| `GET /v1/presets`, `POST /v1/presets/reload` | プリセットの一覧と再読み込み。 |
| `GET /v1/status` | 現在の状態を返します。 |

```python
import httpx
r = httpx.post("http://127.0.0.1:8700/v1/decide", timeout=60, json={
    "input": {"order_id": "A-1029", "status": "delayed", "days_late": 6, "customer_message": "まだ届きません。キャンセルしたい"},
    "schema": {"action": ["wait", "refund", "cancel_and_refund", "escalate"], "anger": {"type": "score", "min": 1, "max": 5}},
})
d = r.json()
if d["action"] == "cancel_and_refund" and d["confidence"] > 0.9:
    ...
```

## プリセット

`judges/<id>.yaml` に、`name`, `description`, `instruction`, `schema`, `examples` を書きます。
任意で `reasoning`, `min_confidence`, `accepts_images` も指定できます。
同梱しているのは `routing`, `spam`, `sentiment`, `moderation`, `yes_no`, `ad_image`（画像用）です。

## モデルを差し替える

`config.yaml` の `backends` に追加し、`active_backend` を書き換えるか、UI や `--backend` で切り替えます。

登録済みのバックエンド:

| 名前 | モデル | 速度（1件、並列なし） | 備考 |
|---|---|---|---|
| `bonsai2`（既定） | Ternary-Bonsai-2-27B（2bit） | 約 0.35 秒 | 画像対応。精度重視 |
| `qwen3.5-9b` | Qwen3.5-9B（Q6_K、unsloth 版） | 約 0.22 秒 | 画像対応。速度と精度のバランス型。思考オンにすると長く考える（`low` でも 10 秒以上） |
| `qwen3-1.7b` | Qwen3-1.7B（Q8_0） | 約 0.08 秒 | テキストのみ。約 4〜5 倍速いが、判定ミスが目立つ |

`start-judge.bat --backend qwen3-1.7b` で起動するか、UI 右上のプルダウンで切り替えます。

- **別の GGUF**: `kind: llamacpp` で `model_path`（画像対応なら `mmproj_path` も）を指定します。思考モードの無いモデルは `reasoning_style: none` にします。
- **OpenAI 互換 API**（vLLM、Ollama、LM Studio、クラウドなど）: `kind: openai` で `base_url`, `model`, `api_key_env` を指定します。
  logprobs が取れない API では確率と confidence が `null` になります（`confidence_source: none`）。

## 速度の目安（RTX 4080、Bonsai 2 27B、思考オフ、理由文なし）

| | テキスト専用 | 画像対応 |
|---|---|---|
| 1 件（フィールド 1〜2 個） | 約 0.2〜0.7 秒 | 約 0.3〜0.8 秒 |
| 1 件（フィールド 7 個） | 約 1.4 秒 | — |
| 画像 1 枚 | — | 約 0.9〜1.2 秒 |
| 並列スロット / コンテキスト | 8 / 32K | 4 / 16K |

フィールドが増えるほど遅くなります。理由文（`explain`）や思考（`reasoning`）を付けると、さらに 1〜3 秒かかります。

速度に関わる実装上の工夫:

- 思考オフの時は、空白なしの JSON だけを許す GBNF 文法で出力させています。出力トークン数がほぼ半分になります（llama.cpp 系のバックエンドだけ。`compact_grammar: false` で無効にできます）。
- llama-server の RAM プロンプトキャッシュは無効にしています（`--cache-ram 0`）。このモデルは 1 エントリが約 610MB あり、並列処理の妨げになるためです。
- 注意: このモデルの 2bit 形式（PQ2_0）は、同時に複数件処理してもあまり速くなりません（1 件ずつ約 55 トークン/秒、8 件同時でも合計約 60 トークン/秒）。並列化による短縮はおよそ 1.4 倍が上限です。

## ファイル構成

```
judge/
  app/main.py        FastAPI (エンドポイント)
  app/engine.py      schema の解釈、プロンプト・JSON スキーマ生成、確率計算
  app/scoring.py     logprobs → 選択肢ごとの確率
  app/backends.py    バックエンド抽象 (llama-server の管理 / OpenAI 互換)
  app/static/        Web UI
  judges/*.yaml      プリセット
  config.yaml        設定
  tests/             test_scoring.py (モデル不要), smoke.py (起動中の API で動作確認)
```
