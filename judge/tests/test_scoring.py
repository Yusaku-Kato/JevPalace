"""モデル不要の単体テスト: python tests\\test_scoring.py"""
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import engine, scoring  # noqa: E402


def tok(text, p=0.99, top=None):
    top = top or [(text, p)]
    return {"token": text, "logprob": math.log(p), "top_logprobs": [{"token": t, "logprob": math.log(q)} for t, q in top]}


def toks(*items):
    return scoring.parse_logprobs(list(items))


def test_choice_distribution_basic():
    t = toks(tok('{"r": "'), tok("spam", .8, [("spam", .8), ("not", .15), ("hello", .05)]), tok('"}'))
    d = scoring.choice_distribution(t, len('{"r": "'), ["spam", "not_spam"])
    assert abs(d["spam"] - .8 / .95) < 1e-6 and abs(d["not_spam"] - .15 / .95) < 1e-6


def test_shared_prefix_multi_token():
    # "1" と "10" (score) のように先頭トークンを共有 → 次のトークンで分岐
    t = toks(tok('{"s": "'), tok("1", .9, [("1", .9), ("5", .1)]), tok("0", .6, [("0", .6), ('"}', .4)]), tok('"}'))
    d = scoring.choice_distribution(t, len('{"s": "'), ["1", "5", "10"])
    assert abs(d["5"] - .1) < 1e-6 and abs(d["10"] - .54) < 1e-6 and abs(d["1"] - .36) < 1e-6


def test_head_inside_token():
    t = toks(tok('{"r":'), tok(' "spam', .7, [(' "spam', .7), (' "not', .3), ("xx", .2)]), tok('"}'))
    d = scoring.choice_distribution(t, len('{"r": "'), ["spam", "not_spam"])
    assert abs(d["spam"] - .7) < 1e-6


def test_score_fields_all_kinds_with_thinking_prefix():
    fields = [engine.parse_field("route", ["billing", "tech"]), engine.parse_field("urgent", "boolean"),
              engine.parse_field("score", "number"), engine.parse_field("name", "string")]
    content = '{"route": "billing", "urgent": true, "score": "9", "name": "山田"}'
    pre = '<think>\n"route": "tech"\n</think>\n\n'
    t = toks(tok(pre), tok('{"route": "'), tok("billing", .9, [("billing", .9), ("tech", .1)]),
             tok('", "urgent": '), tok("true", .8, [("true", .8), ("false", .2)]),
             tok(', "score": "'), tok("9", .6, [("9", .6), ("8", .3), ("1", .1)]), tok('", "name": "'),
             tok("山田", .5), tok('"}'))
    d = engine._score_fields(fields, json.loads(content), content, t)
    assert d["route"]["value"] == "billing" and abs(d["route"]["confidence"] - .9) < 1e-4
    assert d["urgent"]["value"] is True and abs(d["urgent"]["probability"] - .8) < 1e-4
    # "1" の後に閉じ引用符/0 が来るかは不明だが、このトークン列では "1" は分岐先が無いので "1" と "10" で均等割り
    assert abs(d["score"]["value"] - (9 * .6 + 8 * .3 + 1 * .05 + 10 * .05) / 10) < 1e-3
    assert abs(d["score"]["confidence"] - .95) < 1e-4  # 9±1 (8,9,10)
    assert d["name"]["value"] == "山田" and abs(d["name"]["confidence"] - .5) < 1e-4


def test_compact_grammar():
    fields = [engine.parse_field("route", ["a", "b"]), engine.parse_field("s", "number"),
              engine.parse_field("ok", "boolean"), engine.parse_field("n", "string")]
    g = engine.build_gbnf(fields, explain=True)
    assert g.startswith('root ::= "{" f0 f1 f2 f3 reason "}"')
    assert 'f0 ::= "\\"route\\": " ("\\"a\\"" | "\\"b\\"")' in g
    assert '"\\"10\\""' in g and 'f2 ::= ", \\"ok\\": " ("true" | "false")' in g


def test_parse_field_errors():
    for name, raw in [("confidence", "boolean"), ("x", ["a"]), ("x", "weird"), ("x", {"type": "score", "min": 0, "max": 100})]:
        try:
            engine.parse_field(name, raw)
        except engine.DecisionError:
            continue
        raise AssertionError((name, raw))


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)

