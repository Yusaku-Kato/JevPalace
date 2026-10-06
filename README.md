# JevPalace

[Jev](https://jevai.net/) のような「判定専用 API」を、手元の GPU とローカル LLM で動かすプロジェクトです。

入力と、出力の型（`schema`）を渡すと、文章は生成せず、型どおりの値と確信度だけを返します。

```
POST /v1/decide
{
  "input": "先月分が二重に請求されています。返金してください。",
  "schema": {"route": ["billing", "technical", "sales"], "score": "number"}
}
→ {"route": "billing", "score": 0.93, "confidence": 0.90, ...}
```

- **型の安全性**: 出力は JSON スキーマ / GBNF 文法で制約されるので、選択肢にない値や型違いは返りません。
- **確信度**: モデルの自己申告ではなく、トークンの logprobs から各選択肢の確率を計算しています。
- **画像対応のオン/オフ**: オフの時は画像用モジュールを読み込まないので、軽くて速く動きます。
- **モデルの差し替え**: 設定ファイルに追加するだけで、別の GGUF モデルや OpenAI 互換 API に切り替えられます。
- **Web UI**: schema を書いて、その場で判定を試せます。

API の詳細、schema の書き方、プリセットの作り方は [judge/README.md](judge/README.md) を参照してください。

## 動作環境

- Windows 10/11（x64）
- NVIDIA GPU。VRAM 16GB を推奨（RTX 4080 で動作確認）。小さいモデルなら 8GB でも動きます。
- Python 3.11 以降

## セットアップ

### 1. リポジトリを取得する

```bat
git clone https://github.com/<user>/JevPalace.git
cd JevPalace
```

### 2. llama.cpp（PrismML フォーク）を入れる

Bonsai 2 の 2bit 形式（PQ2_0）は、通常の llama.cpp では動きません。
[PrismML-Eng/llama.cpp のリリース](https://github.com/PrismML-Eng/llama.cpp/releases/latest)から
Windows・CUDA 12 版をダウンロードし、中身を `llama\` フォルダに展開してください（`llama\llama-server.exe` ができれば OK）。

動作確認したビルドは `b10754-2459f68b5` です。Bonsai を使わないなら、通常の [llama.cpp](https://github.com/ggml-org/llama.cpp/releases) でも動きます。

### 3. モデルをダウンロードする

`models\` フォルダに置きます。使うモデルの分だけで構いません。

| バックエンド名 | ファイル | 入手先 |
|---|---|---|
| `bonsai2`（既定） | `Ternary-Bonsai-2-27B-Abliterated-PQ2_0.gguf`（7.2GB） | [Hikari07jp/Ternary-Bonsai-2-27B-Abliterated-GGUF](https://huggingface.co/Hikari07jp/Ternary-Bonsai-2-27B-Abliterated-GGUF) |
| 〃（画像用） | `Ternary-Bonsai-2-27B-mmproj-Q8_0.gguf`（0.6GB） | [prism-ml/Ternary-Bonsai-2-27B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-27B-gguf) |
| `qwen3.5-9b` | `Qwen3.5-9B-Q6_K.gguf`（7.0GB） | [unsloth/Qwen3.5-9B-GGUF](https://huggingface.co/unsloth/Qwen3.5-9B-GGUF) |
| 〃（画像用） | `mmproj-F16.gguf` を `Qwen3.5-9B-mmproj-F16.gguf` という名前で保存（0.9GB） | 同上 |
| `qwen3-1.7b` | `Qwen3-1.7B-Q8_0.gguf`（1.8GB） | [Qwen/Qwen3-1.7B-GGUF](https://huggingface.co/Qwen/Qwen3-1.7B-GGUF) |

例:

```bat
curl -L -o models\Qwen3-1.7B-Q8_0.gguf https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf
```

Bonsai 2 を使わない場合は、`judge\config.yaml` の `active_backend` を使うモデルの名前に変えてください。

### 4. 起動する

```bat
start-judge.bat
```

初回は `judge\.venv` に Python の仮想環境を作り、依存パッケージを入れます。
起動するとブラウザで Web UI（http://127.0.0.1:8700）が開きます。

```bat
start-judge.bat --vision                 rem 画像対応モードで起動
start-judge.bat --backend qwen3.5-9b     rem 別のモデルで起動
```

API ドキュメント（Swagger UI）は http://127.0.0.1:8700/docs で見られます。

## モデルの比較（RTX 4080、思考オフ）

| | Bonsai 2（27B） | Qwen3.5-9B | Qwen3-1.7B |
|---|---|---|---|
| 1 件あたり | 約 0.37 秒 | 約 0.22 秒 | 約 0.08 秒 |
| 47 件を 8 並列で処理 | 12.6 秒 | 7.6 秒 | 2.4 秒 |
| 画像 | ✅ | ✅ | — |
| 判定精度（手元のテスト） | 高い | 高い | ミスが目立つ |

## ディレクトリ構成

```
JevPalace/
├─ judge/              判定 API 本体 (FastAPI)
│   ├─ app/            API・判定ロジック・Web UI
│   ├─ judges/         プリセット (YAML)
│   ├─ tests/          テスト
│   ├─ config.yaml     バックエンド (モデル) の設定
│   └─ README.md       API の詳細
├─ tmp/                API を使った一時的な実験スクリプト
├─ llama/              llama.cpp のバイナリ (各自ダウンロード、git 管理外)
├─ models/             GGUF モデル (各自ダウンロード、git 管理外)
└─ start-judge.bat     起動スクリプト
```

## テスト

```bat
judge\.venv\Scripts\python.exe judge\tests\test_scoring.py   rem モデル不要の単体テスト
judge\.venv\Scripts\python.exe judge\tests\smoke.py          rem 起動中の API に対する動作確認
```

## 注意

- 判定結果はモデルの出力であり、正しさは保証されません。確信度も、モデルの偏りをそのまま反映します。
- 既定の Bonsai 2 は拒否応答を取り除いた（abliterated）版です。用途に応じてモデルを選んでください。
- `tmp/` のスクリプトは、実在の人物や地域についてモデルの主観的な判定を出力する実験です。結果は事実に基づく評価ではありません。
