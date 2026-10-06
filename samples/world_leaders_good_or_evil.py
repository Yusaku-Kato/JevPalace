"""サンプル: 現在の各国の指導者を一人ずつ「善人か悪人か」判定する。

指導者の一覧は実行時に Wikidata から取得する (主権国家ごとの国家元首 P35 / 政府の長 P6)。
ハードコードしないのは、指導者が頻繁に入れ替わるため。

usage (先に start-judge.bat で API を起動しておく):
    judge\\.venv\\Scripts\\python.exe samples\\world_leaders_good_or_evil.py --contact you@example.com
        [--role both|head_of_state|head_of_government] [--explain] [--reasoning low] [--csv out.csv] [--limit 10]

--contact は Wikidata の User-Agent ポリシーで必要 (無いと 403)。初回だけ必要で、取得結果は
samples\\leaders_cache.json に保存され、2 回目以降はそれを使う (--refresh で取り直し)。

※ 結果はローカル LLM (既定: Bonsai 2) の主観的な出力であり、事実的な評価ではない。
※ Wikidata の内容は最新とは限らない (就任直後の指導者が反映されていないことがある)。
"""
import argparse
import csv
import json
import os
import sys
from pathlib import Path

import httpx

API = "http://127.0.0.1:8700"
WIKIDATA = "https://query.wikidata.org/sparql"
CACHE = Path(__file__).with_name("leaders_cache.json")

# 主権国家 (Q3624078) のうち、歴史上の国家 (Q3024240) を除いたものの元首と政府の長
QUERY = """
SELECT ?country ?countryLabel ?role ?person ?personLabel WHERE {
  ?country wdt:P31 wd:Q3624078 .
  FILTER NOT EXISTS { ?country wdt:P31 wd:Q3024240 }
  FILTER NOT EXISTS { ?country wdt:P576 ?dissolved }
  { ?country wdt:P35 ?person . BIND("head_of_state" AS ?role) }
  UNION
  { ?country wdt:P6 ?person . BIND("head_of_government" AS ?role) }
  SERVICE wikibase:label { bd:serviceParam wikibase:language "ja,en". }
}
"""
ROLE_JA = {"head_of_state": "国家元首", "head_of_government": "政府の長"}

SCHEMA = {"verdict": {"type": "choice", "options": {"善人": "総合的に見て善い人物", "悪人": "総合的に見て悪い人物"}}}
INSTRUCTION = "入力の人物（現在の国家指導者）が、これまでの行いを総合的に見て善人か悪人かを判定してください。"


def fetch_bindings(contact: str | None, refresh: bool) -> list[dict]:
    """Wikidata から取得 (結果は CACHE に保存し、2 回目以降は再利用)。"""
    if CACHE.exists() and not refresh:
        print(f"キャッシュを使用: {CACHE.name} (取り直すなら --refresh)", flush=True)
        return json.loads(CACHE.read_text(encoding="utf-8"))
    if not contact:
        sys.exit("Wikidata はアクセス時に連絡先を求めます (User-Agent ポリシー: https://w.wiki/4wJS)。\n"
                 "--contact にメールアドレスか URL を指定するか、環境変数 WIKIDATA_CONTACT を設定してください。\n"
                 "例: --contact you@example.com")
    print("Wikidata から指導者一覧を取得中…", flush=True)
    r = httpx.get(WIKIDATA, params={"query": QUERY, "format": "json"}, timeout=120,
                  headers={"User-Agent": f"JevPalace-sample/0.1 ({contact}) httpx/{httpx.__version__}",
                           "Accept": "application/sparql-results+json"})
    r.raise_for_status()
    bindings = r.json()["results"]["bindings"]
    CACHE.write_text(json.dumps(bindings, ensure_ascii=False), encoding="utf-8")
    return bindings


def fetch_leaders(role: str, contact: str | None, refresh: bool) -> list[dict]:
    leaders: dict[tuple[str, str], dict] = {}
    for b in fetch_bindings(contact, refresh):
        name = b["personLabel"]["value"]
        if name.startswith("Q") and name[1:].isdigit():  # ラベルの無い項目は飛ばす
            continue
        key = (b["country"]["value"], b["person"]["value"])
        roles = leaders.setdefault(key, {"country": b["countryLabel"]["value"], "name": name, "roles": set()})["roles"]
        roles.add(b["role"]["value"])
    out = []
    for v in leaders.values():
        if role != "both" and role not in v["roles"]:
            continue
        v["role"] = "・".join(ROLE_JA[x] for x in sorted(v["roles"], reverse=True))
        out.append(v)
    return sorted(out, key=lambda x: (x["country"], x["role"], x["name"]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--role", default="both", choices=["both", "head_of_state", "head_of_government"],
                    help="対象: 両方 / 国家元首のみ / 政府の長のみ")
    ap.add_argument("--explain", action="store_true", help="理由文も出す (遅くなる)")
    ap.add_argument("--reasoning", default="off", choices=["off", "low", "medium", "high"])
    ap.add_argument("--csv", help="結果を CSV に保存")
    ap.add_argument("--limit", type=int, help="先頭 N 人だけ判定 (お試し用)")
    ap.add_argument("--contact", default=os.environ.get("WIKIDATA_CONTACT"),
                    help="Wikidata に送る連絡先 (メールアドレスか URL)。初回の取得時のみ必要")
    ap.add_argument("--refresh", action="store_true", help="キャッシュを使わず Wikidata から取り直す")
    a = ap.parse_args()

    c = httpx.Client(base_url=API, timeout=300)
    try:
        st = c.get("/v1/status").json()
    except httpx.HTTPError:
        sys.exit("API に接続できません。先に start-judge.bat を起動してください。")

    try:
        leaders = fetch_leaders(a.role, a.contact, a.refresh)
    except httpx.HTTPError as e:
        sys.exit(f"Wikidata の取得に失敗しました: {e}")
    if a.limit:
        leaders = leaders[:a.limit]
    print(f"{len(leaders)} 人  model: {st['backend']['model']}  (結果はモデルの主観です)\n")

    rows = []
    for i, p in enumerate(leaders, 1):
        r = c.post("/v1/decide", json={"input": f"{p['name']}（{p['country']}の{p['role']}）", "schema": SCHEMA,
                                       "instruction": INSTRUCTION, "explain": a.explain, "reasoning": a.reasoning})
        if r.status_code != 200:
            print(f"{i:3d}. {p['country']} {p['name']}: エラー {r.json()}")
            continue
        d = r.json()
        p_good = (d["fields"]["verdict"]["probabilities"] or {}).get("善人", 0)
        line = f"{i:3d}. {p['country']}\t{p['role']}\t{p['name']}\t{d['verdict']}\t(善人 {p_good:6.1%})"
        if a.explain:
            line += f"\n       {d.get('reason', '')}"
        print(line, flush=True)
        rows.append({"no": i, "country": p["country"], "role": p["role"], "name": p["name"],
                     "verdict": d["verdict"], "p_good": p_good, "confidence": d["confidence"],
                     "reason": d.get("reason", "")})

    if not rows:
        return
    good = sum(r["verdict"] == "善人" for r in rows)
    print(f"\n善人 {good} / 悪人 {len(rows) - good} (計 {len(rows)} 人)")
    if a.csv:
        with open(a.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"saved: {a.csv}")


if __name__ == "__main__":
    main()
