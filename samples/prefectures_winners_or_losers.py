"""サンプル: 47 都道府県について、そこに住む人が「勝ち組か負け組か」を並列で判定する。

usage (先に start-judge.bat で API を起動しておく):
    judge\\.venv\\Scripts\\python.exe samples\\prefectures_winners_or_losers.py [--workers 8] [--explain] [--csv out.csv]

各行の時間:
    処理 = API 内で判定にかかった時間 (meta.latency_ms)
    往復 = リクエストを送ってから返るまで (並列時は順番待ちを含む)

※ 結果はローカル LLM (既定: Bonsai 2) の主観・ステレオタイプの反映であり、事実的な評価ではない。
"""
import argparse
import csv
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

API = "http://127.0.0.1:8700"

PREFECTURES = [
    "北海道", "青森県", "岩手県", "宮城県", "秋田県", "山形県", "福島県",
    "茨城県", "栃木県", "群馬県", "埼玉県", "千葉県", "東京都", "神奈川県",
    "新潟県", "富山県", "石川県", "福井県", "山梨県", "長野県", "岐阜県", "静岡県", "愛知県",
    "三重県", "滋賀県", "京都府", "大阪府", "兵庫県", "奈良県", "和歌山県",
    "鳥取県", "島根県", "岡山県", "広島県", "山口県",
    "徳島県", "香川県", "愛媛県", "高知県",
    "福岡県", "佐賀県", "長崎県", "熊本県", "大分県", "宮崎県", "鹿児島県", "沖縄県",
]

SCHEMA = {"verdict": {"type": "choice", "options": {"勝ち組": "恵まれた暮らし・成功している", "負け組": "恵まれない暮らし・成功していない"}}}
INSTRUCTION = "入力の都道府県に住む人が、一般的に見て勝ち組か負け組かを判定してください。"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, help="同時リクエスト数 (既定: API の並列スロット数)")
    ap.add_argument("--explain", action="store_true", help="理由文も出す (遅くなる)")
    ap.add_argument("--reasoning", default="off", choices=["off", "low", "medium", "high"])
    ap.add_argument("--csv", help="結果を CSV に保存")
    a = ap.parse_args()

    try:
        st = httpx.get(API + "/v1/status", timeout=10).json()
    except httpx.HTTPError:
        sys.exit("API に接続できません。先に start-judge.bat を起動してください。")
    workers = a.workers or st["backend"]["parallel"]
    print(f"{len(PREFECTURES)} 都道府県 / 並列 {workers}  model: {st['backend']['model']}  (結果はモデルの主観です)\n", flush=True)

    local = threading.local()

    def judge(item: tuple[int, str]) -> dict:
        no, pref = item
        if not hasattr(local, "client"):  # httpx.Client はスレッドごとに持つ
            local.client = httpx.Client(base_url=API, timeout=600)
        body = {"input": f"{pref}に住む人", "schema": SCHEMA, "instruction": INSTRUCTION,
                "explain": a.explain, "reasoning": a.reasoning}
        t = time.perf_counter()
        try:
            r = local.client.post("/v1/decide", json=body)
            rt_ms = round((time.perf_counter() - t) * 1000)
            d = r.json()
            if r.status_code != 200:
                return {"no": no, "pref": pref, "error": d.get("error", r.text), "rt_ms": rt_ms}
        except httpx.HTTPError as e:
            return {"no": no, "pref": pref, "error": str(e), "rt_ms": round((time.perf_counter() - t) * 1000)}
        probs = d["fields"]["verdict"]["probabilities"] or {}
        return {"no": no, "pref": pref, "verdict": d["verdict"], "p_win": probs.get("勝ち組"),
                "confidence": d["confidence"], "latency_ms": d["meta"]["latency_ms"], "rt_ms": rt_ms,
                "reason": d.get("reason", "")}

    t0 = time.perf_counter()
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(judge, enumerate(PREFECTURES, 1)):  # 表示は都道府県コード順 (処理自体は並列)
            rows.append(res)
            head = f"{res['no']:2d}. {res['pref']:<5}"
            if "error" in res:
                print(f"{head}\tエラー: {res['error']}  (往復 {res['rt_ms']}ms)", flush=True)
                continue
            line = (f"{head}\t{res['verdict']}\t(勝ち組 {res['p_win'] or 0:6.1%})"
                    f"\t処理 {res['latency_ms']:4d}ms / 往復 {res['rt_ms']:4d}ms")
            if a.explain:
                line += f"\n      {res['reason']}"
            print(line, flush=True)
    total = time.perf_counter() - t0

    ok = [r for r in rows if "error" not in r]
    win = sum(r["verdict"] == "勝ち組" for r in ok)
    print(f"\n勝ち組 {win} / 負け組 {len(ok) - win} (計 {len(ok)}, エラー {len(rows) - len(ok)})")
    if ok:
        lat = sorted(r["latency_ms"] for r in ok)
        print(f"処理時間: 平均 {sum(lat) / len(lat):.0f}ms / 中央値 {lat[len(lat) // 2]}ms / 最小 {lat[0]}ms / 最大 {lat[-1]}ms")
    print(f"全体: {total:.2f}秒 (1件あたり実効 {total * 1000 / max(len(rows), 1):.0f}ms)")
    if a.csv and ok:
        fields = ["no", "pref", "verdict", "p_win", "confidence", "latency_ms", "rt_ms", "reason"]
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(ok)
        print(f"saved: {a.csv}")


if __name__ == "__main__":
    main()
