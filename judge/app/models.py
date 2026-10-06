from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class DecideRequest(BaseModel):
    """JEV 互換の判定リクエスト。

    schema の各フィールドは次のどれか:
      ["a", "b", ...]                         Choice  (選択肢から 1 つ + 各選択肢の確率)
      "boolean"                               Binary  (true/false + true の確率)
      "number" / "score"                      Score   (0〜1 のスコア。内部では 0〜10 の 11 段階)
      "string"                                抽出    (入力から短い文字列を取り出す)
      {"type": "choice", "options": [...] or {"name": "説明"}, "description": "..."}
      {"type": "score", "min": 1, "max": 5, "description": "..."}
      {"type": "boolean" | "string", "description": "..."}
    """

    input: Any = Field(None, description="判定対象 (文字列、または状態を表す JSON オブジェクト)")
    schema_: dict[str, Any] | None = Field(None, alias="schema", description="出力フィールドの型定義")
    preset: str | None = Field(None, description="judges/*.yaml のプリセット ID (schema / instruction を読み込む)")
    instruction: str | None = Field(None, description="判定の観点・質問 (任意)")
    images: list[str] | None = Field(None, description="画像 (data URL / base64 / http(s) URL)。画像モード時のみ")
    examples: list[dict] | None = Field(None, description='few-shot 例 [{"input": "...", "output": {...}}]')
    explain: bool | None = Field(None, description="理由文も返す (拡張。遅くなる)")
    reasoning: Literal["off", "low", "medium", "high"] | None = Field(None, description="思考モード (拡張。off が最速)")
    min_confidence: float | None = Field(None, ge=0, le=1, description="confidence がこれ未満なら abstain=true")
    return_reasoning: bool = False
    max_tokens: int | None = None

    model_config = {"populate_by_name": True}


class BatchRequest(BaseModel):
    items: list[DecideRequest]
    defaults: DecideRequest | None = Field(None, description="各 item で未指定の項目に適用される共通値")


class VisionRequest(BaseModel):
    enabled: bool


class BackendSwitch(BaseModel):
    name: str
    vision: bool | None = None
