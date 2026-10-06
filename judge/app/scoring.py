"""logprobs から各ラベルの確率を推定する。

モデル出力 (JSON) の中でラベル文字列が始まる位置を特定し、そこから先のトークンを
1 つずつ辿りながら「各ラベルと矛盾しない候補トークン」の確率を集計する。
ラベルが複数トークンにまたがる場合や、先頭トークンを共有する場合 (例: "spam" / "spam_like") も扱える。
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass


@dataclass
class Token:
    text: str
    logprob: float
    top: list[tuple[str, float]]  # (token, logprob)


def parse_logprobs(content: list[dict] | None) -> list[Token]:
    out: list[Token] = []
    for t in content or []:
        top = [(x.get("token", ""), x.get("logprob", -1e9)) for x in t.get("top_logprobs") or []]
        if not top:
            top = [(t.get("token", ""), t.get("logprob", 0.0))]
        out.append(Token(t.get("token", ""), t.get("logprob", 0.0), top))
    return out


def _locate(tokens: list[Token], char_offset: int) -> tuple[int, int] | None:
    """全トークン連結文字列中の char_offset を含む (token index, token 内オフセット)。"""
    pos = 0
    for i, t in enumerate(tokens):
        if pos <= char_offset < pos + len(t.text):
            return i, char_offset - pos
        pos += len(t.text)
    return None


def _content_start(tokens: list[Token], content: str) -> int | None:
    """logprobs は思考部分も含むので、本文 (content) が始まる位置を探す。"""
    full = "".join(t.text for t in tokens)
    idx = full.rfind(content)
    if idx >= 0:
        return idx
    stripped = content.strip()
    idx = full.rfind(stripped)
    return idx if idx >= 0 else None


def choice_distribution(tokens: list[Token], start: int, choices: list[str], terminator: str = '"') -> dict[str, float] | None:
    """tokens 連結文字列の start 文字目から choices のどれかが生成されたとき、各 choice の確率を返す。

    choice の後ろに terminator (JSON 文字列なら閉じ引用符) が続くことを前提に、接頭辞の衝突を解決する。
    """
    loc = _locate(tokens, start)
    if loc is None:
        return None
    idx, off = loc
    targets = {c: c + terminator for c in choices}
    scores = {c: 0.0 for c in choices}
    alive = list(choices)
    prefix = ""
    path_p = 1.0

    def consistent(full: str, c: str) -> bool:
        t = targets[c]
        return t.startswith(full) or full.startswith(t)

    while idx < len(tokens):
        tok = tokens[idx]
        head = tok.text[:off]  # 最初のトークンだけ、ラベル開始前の部分 (例: ' "') を含みうる
        chosen = tok.text[off:]
        cands: dict[str, float] = {}
        for text, lp in tok.top:
            if not text.startswith(head):
                continue
            part = text[len(head):]
            if not part:
                continue
            cands[part] = cands.get(part, 0.0) + math.exp(lp)
        if chosen not in cands:
            cands[chosen] = math.exp(tok.logprob)

        groups: dict[str, list[str]] = {}
        for part in cands:
            full = prefix + part
            groups[part] = [c for c in alive if consistent(full, c)]
        total = sum(p for part, p in cands.items() if groups[part])
        if total <= 0:
            return None
        for part, p in cands.items():
            g = groups[part]
            if part == chosen or not g:
                continue
            for c in g:  # 曖昧な場合は均等割り (稀)
                scores[c] += path_p * (p / total) / len(g)
        g = groups[chosen]
        if not g:
            return None
        path_p *= cands[chosen] / total
        prefix += chosen
        alive = g
        if len(alive) == 1 or any(prefix.startswith(targets[c]) for c in alive):
            done = [c for c in alive if prefix.startswith(targets[c])] or alive
            for c in done:
                scores[c] += path_p / len(done)
            break
        idx += 1
        off = 0
    else:
        return None
    s = sum(scores.values())
    return {c: v / s for c, v in scores.items()} if s > 0 else None


def span_probability(tokens: list[Token], start: int, end: int) -> float | None:
    """[start, end) の文字列を生成した確率 (重なるトークンの確率の積)。"""
    pos, lp, hit = 0, 0.0, False
    for t in tokens:
        nxt = pos + len(t.text)
        if nxt > start and pos < max(end, start + 1):
            lp += t.logprob
            hit = True
        if pos >= max(end, start + 1):
            break
        pos = nxt
    return math.exp(lp) if hit else None
