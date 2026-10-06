"""サンプル: 複数のモデル (バックエンド) で samples のスクリプトを実行し、速度と簡単な正答率を比べる。

usage (先に start-judge.bat で API を起動しておく):
    judge\\.venv\\Scripts\\python.exe samples\\benchmark_models.py qwen3-1.7b lfm2.5-1.2b ...

- バックエンドは POST /v1/backends/switch で順番に切り替える (終わったら元のバックエンドに戻す)。
- 速度: 各サンプルを子プロセスで実行し、全体の所要時間を測る (Python の起動時間 約 0.3 秒を含む)。
- 正答率: 答えがはっきりしている判定 16 問。モデルの大まかな判定力の目安。
"""
import subprocess
import sys
import time
from pathlib import Path

import httpx

API = "http://127.0.0.1:8700"
ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable

# (名前, サンプルの引数, 件数)
WORKLOADS = [
    ("都道府県 47件 並列", ["samples/prefectures_winners_or_losers.py"], 47),
    ("都道府県 47件 1件ずつ", ["samples/prefectures_winners_or_losers.py", "--workers", "1"], 47),
    ("総理大臣 66人 1件ずつ", ["samples/pm_good_or_evil.py"], 66),
    ("長者番付 100人 並列", ["samples/billionaires_genius_or_not.py", "--limit", "100"], 100),
]

# (説明, リクエスト, 正解のフィールドと値)
ACCURACY = [
    ("請求の振り分け", {"preset": "routing", "input": "先月分が二重に請求されています。返金してください。"}, ("route", "billing")),
    ("障害の振り分け", {"preset": "routing", "input": "ログインすると500エラーが出て仕事が止まっています"}, ("route", "technical")),
    ("見積もりの振り分け", {"preset": "routing", "input": "100名規模で導入を検討しています。見積もりをお願いできますか"}, ("route", "sales")),
    ("パスワードの振り分け", {"preset": "routing", "input": "パスワードを忘れてしまい、再設定のメールも届きません"}, ("route", "account")),
    ("当選詐欺", {"preset": "spam", "input": "【当選】Amazonギフト券5万円分が当たりました。24時間以内にこちらのURLから受け取り"}, ("spam", True)),
    ("業務連絡", {"preset": "spam", "input": "明日の会議は10時からに変更になりました。資料は共有フォルダにあります。"}, ("spam", False)),
    ("フィッシング", {"preset": "spam", "input": "【重要】お客様のアカウントがロックされました。以下のリンクからログインして本人確認してください"}, ("spam", True)),
    ("否定的", {"preset": "sentiment", "input": "サポートの対応が遅すぎて二度と使いたくない"}, ("sentiment", "negative")),
    ("肯定的", {"preset": "sentiment", "input": "スタッフの方がとても親切で、また絶対に来ます！"}, ("sentiment", "positive")),
    ("中立", {"preset": "sentiment", "input": "本日の営業時間は10時から18時までです。"}, ("sentiment", "neutral")),
    ("敬語である", {"preset": "yes_no", "instruction": "入力は敬語(丁寧語)で書かれているか", "input": "資料を送付いたしますのでご確認ください"}, ("answer", True)),
    ("敬語でない", {"preset": "yes_no", "instruction": "入力は敬語(丁寧語)で書かれているか", "input": "資料送っといたから見といて"}, ("answer", False)),
    ("数字を含む", {"preset": "yes_no", "instruction": "入力に電話番号が含まれているか", "input": "ご連絡は 03-1234-5678 までお願いします"}, ("answer", True)),
    ("脅迫", {"preset": "moderation", "input": "おいお前、住所は東京都新宿区1-2-3だろ。晒してやるから覚悟しとけ"}, ("harassment", True)),
    ("無害", {"preset": "moderation", "input": "週末は家族で公園に行ってピクニックをしました"}, ("harassment", False)),
    ("名前の抽出", {"input": "お世話になっております。株式会社ABCの田中です。", "schema": {"person": "string"}}, ("person", "田中")),
]


def run_workloads() -> list[tuple[str, float, int]]:
    out = []
    for name, args, n in WORKLOADS:
        t = time.perf_counter()
        p = subprocess.run([PY, *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8",
                           env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"})
        dt = time.perf_counter() - t
        if p.returncode != 0:
            print(f"    {name}: 失敗\n{p.stdout[-500:]}{p.stderr[-500:]}")
            out.append((name, float("nan"), n))
            continue
        print(f"    {name:<22} {dt:6.2f}秒  ({dt * 1000 / n:5.0f}ms/件)", flush=True)
        out.append((name, dt, n))
    return out


def run_accuracy(c: httpx.Client) -> tuple[int, list[str]]:
    ok, wrong = 0, []
    for desc, body, (field, want) in ACCURACY:
        d = c.post("/v1/decide", json=body).json()
        got = d.get(field, d.get("error"))
        hit = (want in got) if isinstance(want, str) and field == "person" and isinstance(got, str) else got == want
        ok += hit
        if not hit:
            wrong.append(f"{desc}({field}={got!r})")
    return ok, wrong


def main() -> None:
    backends = sys.argv[1:]
    c = httpx.Client(base_url=API, timeout=600)
    try:
        st = c.get("/v1/status").json()
    except httpx.HTTPError:
        sys.exit("API に接続できません。先に start-judge.bat を起動してください。")
    original = st["backend"]["name"]
    backends = backends or [original]
    summary = []
    try:
        for b in backends:
            print(f"\n=== {b}", flush=True)
            r = c.post("/v1/backends/switch", json={"name": b, "vision": False})
            if r.status_code != 200:
                print(f"    切替失敗: {r.text[:300]}")
                continue
            info = r.json()["backend"]
            vram = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader"],
                                  capture_output=True, text=True).stdout.strip()
            print(f"    model={info['model']} load={info.get('load_seconds')}s VRAM={vram}", flush=True)
            c.post("/v1/decide", json=ACCURACY[0][1])  # ウォームアップ
            times = run_workloads()
            acc, wrong = run_accuracy(c)
            print(f"    正答 {acc}/{len(ACCURACY)}" + (f"  誤答: {', '.join(wrong)}" if wrong else ""), flush=True)
            summary.append((b, info.get("load_seconds"), vram, times, acc))
    finally:
        if backends != [original]:
            c.post("/v1/backends/switch", json={"name": original})

    print("\n| バックエンド | 読込 | VRAM | " + " | ".join(n for n, _, _ in WORKLOADS) + f" | 正答 (/{len(ACCURACY)}) |")
    print("|---" * (len(WORKLOADS) + 4) + "|")
    for b, load, vram, times, acc in summary:
        cells = " | ".join(f"{dt:.1f}秒 ({dt * 1000 / n:.0f}ms/件)" for _, dt, n in times)
        print(f"| {b} | {load}s | {vram} | {cells} | {acc} |")


if __name__ == "__main__":
    main()
