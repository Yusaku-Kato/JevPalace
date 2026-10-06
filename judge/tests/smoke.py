"""起動中の Decision API に対する動作確認。 usage: .venv\\Scripts\\python.exe tests\\smoke.py [--vision]"""
import base64
import io
import json
import sys
import time

import httpx
from PIL import Image, ImageDraw

c = httpx.Client(base_url="http://127.0.0.1:8700", timeout=600)


def show(title, body):
    r = c.post("/v1/decide", json=body)
    d = r.json()
    if r.status_code != 200:
        print(f"[{title}] HTTP {r.status_code}: {d}")
        return d
    compact = {k: v for k, v in d.items() if k not in ("fields", "meta", "abstain")}
    print(f"[{title}] {d['meta']['latency_ms']}ms {json.dumps(compact, ensure_ascii=False)}")
    return d


def img_data_url():
    im = Image.new("RGB", (800, 600), "white")
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, 800, 120), fill="#d6336c")
    d.text((40, 40), "BIG SALE  50% OFF  -  BUY NOW", fill="white")
    d.ellipse((300, 200, 500, 400), fill="orange")
    d.text((330, 450), "$19.99 only today!", fill="black")
    buf = io.BytesIO()
    im.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


if "--vision" not in sys.argv:
    # JEV のトップページの例そのまま
    show("jev-example", {"input": "先月分が二重に請求されています。返金してください。",
                         "schema": {"route": ["billing", "technical", "sales"], "score": "number"}})
    show("preset routing", {"preset": "routing", "input": "ログインしようとすると500エラーが出て仕事が止まっています"})
    show("preset spam", {"preset": "spam", "input": "おめでとうございます！Amazonギフト券5万円分が当選しました。24時間以内にこちらのURLから"})
    show("sentiment+explain", {"preset": "sentiment", "input": "サポートの対応が遅すぎて二度と使いたくない", "explain": True})
    show("smart if", {"preset": "yes_no", "instruction": "入力は敬語(丁寧語)で書かれているか", "input": "資料を送付いたしますのでご確認ください"})
    show("moderation", {"preset": "moderation", "input": "おいお前、住所は東京都新宿区1-2-3だろ。晒してやるから覚悟しとけ"})
    show("json state + scale + extract", {
        "input": {"order_id": "A-1029", "status": "delayed", "days_late": 6, "customer_message": "まだ届きません。佐藤です。キャンセルしたい"},
        "schema": {"action": ["wait", "refund", "cancel_and_refund", "escalate"],
                   "anger": {"type": "score", "min": 1, "max": 5},
                   "customer_name": "string"}})
    show("min_confidence", {"input": "まあ悪くはないけど値段を考えると微妙かな", "schema": {"sentiment": ["positive", "neutral", "negative"]},
                            "min_confidence": 0.9})
    show("reasoning=low", {"input": "まあ悪くはないけど値段を考えると微妙かな", "schema": {"sentiment": ["positive", "neutral", "negative"]},
                           "reasoning": "low"})
    show("image while vision off", {"preset": "ad_image", "images": [img_data_url()]})
    show("bad schema", {"input": "x", "schema": {"confidence": "boolean"}})
    t = time.time()
    r = c.post("/v1/decide/batch", json={"defaults": {"preset": "sentiment"},
                                         "items": [{"input": s} for s in ["最高!", "最悪", "今日は火曜日", "まあまあ", "感動した", "がっかり", "普通", "また行きたい"]]})
    print("[batch x8]", [x.get("sentiment") for x in r.json()["results"]], f"{time.time() - t:.2f}s")
else:
    show("ad_image", {"preset": "ad_image", "images": [img_data_url()]})
    show("ad_image+text", {"preset": "ad_image", "input": "友達とのランチ写真", "images": [img_data_url()]})
print(json.dumps(c.get("/v1/status").json()["backend"]["name"], ensure_ascii=False))
