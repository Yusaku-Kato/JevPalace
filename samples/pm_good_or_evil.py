"""サンプル: 日本の歴代内閣総理大臣を一人ずつ「善人か悪人か」判定する。

usage (先に start-judge.bat で API を起動しておく):
    judge\\.venv\\Scripts\\python.exe samples\\pm_good_or_evil.py [--explain] [--reasoning low] [--csv out.csv]

※ 結果はローカル LLM (既定: Bonsai 2) の主観的な出力であり、歴史的・事実的な評価ではない。
"""
import argparse
import csv
import sys

import httpx

API = "http://127.0.0.1:8700"

# 初めて就任した順 (再任は省略)。2026年10月時点の手元の知識に基づく。
PRIME_MINISTERS = [
    "伊藤博文", "黒田清隆", "山県有朋", "松方正義", "大隈重信", "桂太郎", "西園寺公望", "山本権兵衛",
    "寺内正毅", "原敬", "高橋是清", "加藤友三郎", "清浦奎吾", "加藤高明", "若槻礼次郎", "田中義一",
    "浜口雄幸", "犬養毅", "斎藤実", "岡田啓介", "広田弘毅", "林銑十郎", "近衛文麿", "平沼騏一郎",
    "阿部信行", "米内光政", "東条英機", "小磯国昭", "鈴木貫太郎", "東久邇宮稔彦王", "幣原喜重郎",
    "吉田茂", "片山哲", "芦田均", "鳩山一郎", "石橋湛山", "岸信介", "池田勇人", "佐藤栄作",
    "田中角栄", "三木武夫", "福田赳夫", "大平正芳", "鈴木善幸", "中曽根康弘", "竹下登", "宇野宗佑",
    "海部俊樹", "宮沢喜一", "細川護熙", "羽田孜", "村山富市", "橋本龍太郎", "小渕恵三", "森喜朗",
    "小泉純一郎", "安倍晋三", "福田康夫", "麻生太郎", "鳩山由紀夫", "菅直人", "野田佳彦", "菅義偉",
    "岸田文雄", "石破茂", "高市早苗",
]

SCHEMA = {"verdict": {"type": "choice", "options": {"善人": "総合的に見て善い人物", "悪人": "総合的に見て悪い人物"}}}
INSTRUCTION = "入力の人物（日本の内閣総理大臣経験者）が、生涯の行いを総合的に見て善人か悪人かを判定してください。"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--explain", action="store_true", help="理由文も出す (遅くなる)")
    ap.add_argument("--reasoning", default="off", choices=["off", "low", "medium", "high"])
    ap.add_argument("--csv", help="結果を CSV に保存")
    a = ap.parse_args()

    c = httpx.Client(base_url=API, timeout=300)
    try:
        st = c.get("/v1/status").json()
    except httpx.HTTPError:
        sys.exit("API に接続できません。先に start-judge.bat を起動してください。")
    print(f"model: {st['backend']['model']}  (結果はモデルの主観です)\n")

    rows = []
    for i, name in enumerate(PRIME_MINISTERS, 1):
        r = c.post("/v1/decide", json={"input": name, "schema": SCHEMA, "instruction": INSTRUCTION,
                                       "explain": a.explain, "reasoning": a.reasoning})
        if r.status_code != 200:
            print(f"{i:2d}. {name}: エラー {r.json()}")
            continue
        d = r.json()
        p_good = d["fields"]["verdict"]["probabilities"] or {}
        line = f"{i:2d}. {name:<8}\t{d['verdict']}\t(善人 {p_good.get('善人', 0):6.1%})\t{d['meta']['latency_ms']}ms"
        if a.explain:
            line += f"\n      {d.get('reason', '')}"
        print(line, flush=True)
        rows.append({"no": i, "name": name, "verdict": d["verdict"], "p_good": p_good.get("善人"),
                     "confidence": d["confidence"], "reason": d.get("reason", "")})

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
