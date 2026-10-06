"""一時スクリプト: 世界長者番付 (Forbes リアルタイム) トップ 500 を「天才か凡才か」並列で判定する。

usage (先に start-judge.bat で API を起動しておく):
    judge\\.venv\\Scripts\\python.exe tmp\\billionaires_genius_or_not.py [--workers 8] [--limit 20]
        [--explain] [--reasoning low] [--csv out.csv] [--refresh]

- 長者番付は Forbes の非公式 JSON エンドポイントから取得し、tmp\\billionaires_cache.json に保存する
  (2 回目以降はキャッシュを使う。--refresh で取り直し)。
- 並列数の既定は API の並列スロット数 (画像オフ時 8)。それ以上にしても速くはならない。

※ 結果はローカル LLM (既定: Bonsai 2) の主観的な出力であり、事実的な評価ではない。
"""
import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

API = "http://127.0.0.1:8700"
FORBES = "https://www.forbes.com/forbesapi/person/rtb/0/position/true.json"
CACHE = Path(__file__).with_name("billionaires_cache.json")

SCHEMA = {"verdict": {"type": "choice", "options": {"天才": "並外れた才能・知性の持ち主", "凡才": "才能・知性は平凡"}}}
INSTRUCTION = ("入力の人物（世界の富豪）が、本人の能力・業績を総合的に見て天才か凡才かを判定してください。"
               "資産額の大きさそのものではなく、本人の才能で判断します。")


def fetch_billionaires(refresh: bool) -> list[dict]:
    if CACHE.exists() and not refresh:
        print(f"キャッシュを使用: {CACHE.name} (取り直すなら --refresh)", flush=True)
        return json.loads(CACHE.read_text(encoding="utf-8"))
    print("Forbes から長者番付を取得中…", flush=True)
    r = httpx.get(FORBES, params={"fields": "rank,personName,finalWorth,source,countryOfCitizenship", "limit": 500},
                  headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=60, follow_redirects=True)
    r.raise_for_status()
    people = [{"rank": p.get("rank"), "name": p.get("personName", ""), "worth_billion_usd": round((p.get("finalWorth") or 0) / 1000, 1),
               "source": p.get("source", ""), "country": p.get("countryOfCitizenship", "")}
              for p in r.json()["personList"]["personsLists"]]
    people.sort(key=lambda p: p["rank"] or 10**9)
    CACHE.write_text(json.dumps(people, ensure_ascii=False, indent=1), encoding="utf-8")
    return people


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, help="同時リクエスト数 (既定: API の並列スロット数)")
    ap.add_argument("--limit", type=int, help="上位 N 人だけ判定 (お試し用)")
    ap.add_argument("--explain", action="store_true", help="理由文も出す (遅くなる)")
    ap.add_argument("--reasoning", default="off", choices=["off", "low", "medium", "high"])
    ap.add_argument("--csv", help="結果を CSV に保存")
    ap.add_argument("--refresh", action="store_true", help="キャッシュを使わず Forbes から取り直す")
    a = ap.parse_args()

    try:
        st = httpx.get(API + "/v1/status", timeout=10).json()
    except httpx.HTTPError:
        sys.exit("API に接続できません。先に start-judge.bat を起動してください。")
    try:
        people = fetch_billionaires(a.refresh)
    except (httpx.HTTPError, KeyError, ValueError) as e:
        sys.exit(f"長者番付の取得に失敗しました: {e}")
    if a.limit:
        people = people[:a.limit]
    workers = a.workers or st["backend"]["parallel"]
    print(f"{len(people)} 人 / 並列 {workers}  model: {st['backend']['model']}  (結果はモデルの主観です)\n", flush=True)

    local = threading.local()

    def judge(p: dict) -> dict:
        if not hasattr(local, "client"):  # httpx.Client はスレッドごとに持つ
            local.client = httpx.Client(base_url=API, timeout=600)
        body = {"input": f"{p['name']}（{p['source']}、{p['country']}）", "schema": SCHEMA, "instruction": INSTRUCTION,
                "explain": a.explain, "reasoning": a.reasoning}
        try:
            r = local.client.post("/v1/decide", json=body)
            d = r.json()
            if r.status_code != 200:
                return {**p, "error": d.get("error", r.text)}
        except httpx.HTTPError as e:
            return {**p, "error": str(e)}
        probs = d["fields"]["verdict"]["probabilities"] or {}
        return {**p, "verdict": d["verdict"], "p_genius": probs.get("天才"), "confidence": d["confidence"],
                "reason": d.get("reason", "")}

    t0 = time.time()
    rows = []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(judge, people):  # map は順位順に返す (処理自体は並列)
            rows.append(res)
            head = f"{res['rank']:3d}. {res['name']} ({res['source']}, ${res['worth_billion_usd']}B)"
            if "error" in res:
                print(f"{head}\tエラー: {res['error']}", flush=True)
                continue
            line = f"{head}\t{res['verdict']}\t(天才 {res['p_genius'] or 0:6.1%})"
            if a.explain:
                line += f"\n       {res['reason']}"
            print(line, flush=True)

    ok = [r for r in rows if "error" not in r]
    genius = sum(r["verdict"] == "天才" for r in ok)
    dt = time.time() - t0
    print(f"\n天才 {genius} / 凡才 {len(ok) - genius} (計 {len(ok)} 人, エラー {len(rows) - len(ok)})"
          f"  {dt:.1f}秒 ({dt / max(len(rows), 1):.2f}秒/人)")
    if a.csv and ok:
        fields = ["rank", "name", "worth_billion_usd", "source", "country", "verdict", "p_genius", "confidence", "reason"]
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            w.writerows(ok)
        print(f"saved: {a.csv}")


if __name__ == "__main__":
    main()
