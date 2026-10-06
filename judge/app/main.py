"""JevPalace Decision API (FastAPI)."""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import uvicorn
import yaml
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from . import engine
from .backends import BackendError, OpenAICompatBackend, make_backend
from .models import BackendSwitch, BatchRequest, DecideRequest, VisionRequest

APP_DIR = Path(__file__).resolve().parent
JUDGE_DIR = APP_DIR.parent
CONFIG_PATH = Path(os.environ.get("JEV_CONFIG", JUDGE_DIR / "config.yaml"))


class State:
    def __init__(self) -> None:
        self.cfg: dict = {}
        self.backend: OpenAICompatBackend | None = None
        self.presets: dict[str, dict] = {}
        self.sem = asyncio.Semaphore(1)
        self.http = httpx.AsyncClient()
        self.switch_lock = asyncio.Lock()
        self.started = time.time()
        self.count = 0
        self.log_path: Path | None = None

    @property
    def defaults(self) -> dict:
        return self.cfg.get("defaults") or {}

    def resize_sem(self) -> None:
        self.sem = asyncio.Semaphore(max(1, self.backend.parallel if self.backend else 1))


S = State()


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}


async def activate(name: str, vision: bool) -> None:
    backends = S.cfg.get("backends") or {}
    if name not in backends:
        raise HTTPException(404, f"backend '{name}' は config.yaml にありません")
    old = S.backend
    if old is not None:
        await old.stop()
    S.backend = make_backend(name, backends[name])
    try:
        await S.backend.start(vision)
    except BackendError:
        if old is not None and old.name != name:  # 失敗したら元のバックエンドに戻す
            S.backend = make_backend(old.name, backends.get(old.name, old.cfg))
            await S.backend.start(old.vision_enabled)
            S.resize_sem()
        raise
    S.resize_sem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    S.cfg = load_config()
    S.presets = engine.load_presets()
    if S.cfg.get("log_decisions"):
        S.log_path = JUDGE_DIR / "logs" / "decisions.jsonl"
        S.log_path.parent.mkdir(parents=True, exist_ok=True)
    vision = os.environ.get("JEV_VISION")
    vision_on = (vision.lower() in ("1", "true", "on")) if vision else bool(S.cfg.get("vision", False))
    name = os.environ.get("JEV_BACKEND") or S.cfg["active_backend"]
    # 起動はバックグラウンドで (UI はすぐ開ける。判定リクエストは準備完了まで待つ)
    task = asyncio.create_task(activate(name, vision_on))
    task.add_done_callback(lambda t: t.exception() and print(f"[jev] backend start failed: {t.exception()}"))
    yield
    if S.backend:
        await S.backend.stop()
    await S.http.aclose()


app = FastAPI(title="JevPalace Decision API", version="1.0.0", lifespan=lifespan)


def auth(request: Request) -> None:
    key = (S.cfg.get("server") or {}).get("api_key")
    if not key:
        return
    got = request.headers.get("x-api-key") or request.headers.get("authorization", "").removeprefix("Bearer ").strip()
    if got != key:
        raise HTTPException(401, "invalid api key")


@app.exception_handler(engine.DecisionError)
async def _bad_request(_: Request, e: engine.DecisionError):
    return JSONResponse({"error": str(e)}, status_code=400)


@app.exception_handler(BackendError)
async def _backend_error(_: Request, e: BackendError):
    return JSONResponse({"error": str(e)}, status_code=502)


def _backend() -> OpenAICompatBackend:
    if S.backend is None:
        raise HTTPException(503, "バックエンド起動中です")
    return S.backend


async def _run(req: DecideRequest) -> dict:
    backend = _backend()
    spec = engine.resolve(req, S.presets)
    async with S.sem:
        res = await engine.decide(req, spec, backend, S.defaults, S.http)
    S.count += 1
    if S.log_path:
        rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "input": req.input,
               **{k: v for k, v in res.items() if k != "thinking"}}
        with open(S.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return res


# --------------------------------------------------------------------------- routes
@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(APP_DIR / "static" / "index.html")


@app.get("/v1/status", dependencies=[Depends(auth)])
async def status() -> dict:
    b = S.backend
    return {
        "backend": b.info() if b else None,
        "health": await b.health() if b else {"ok": False},
        "vision": b.vision_enabled if b else False,
        "backends": list((S.cfg.get("backends") or {}).keys()),
        "presets": list(S.presets.keys()),
        "defaults": S.defaults,
        "requests_served": S.count,
        "uptime_s": round(time.time() - S.started),
    }


@app.get("/v1/presets", dependencies=[Depends(auth)])
async def presets() -> list[dict]:
    return [{"id": k, "name": v.get("name", k), "description": v.get("description", ""),
             "instruction": v.get("instruction", ""), "schema": v.get("schema", {}),
             "accepts_images": v.get("accepts_images", False)} for k, v in S.presets.items()]


@app.get("/v1/presets/{preset_id}", dependencies=[Depends(auth)])
async def preset_detail(preset_id: str) -> dict:
    if preset_id not in S.presets:
        raise HTTPException(404, "not found")
    return S.presets[preset_id]


@app.post("/v1/presets/reload", dependencies=[Depends(auth)])
async def reload_presets() -> dict:
    S.presets = engine.load_presets()
    return {"presets": list(S.presets.keys())}


@app.post("/v1/decide", dependencies=[Depends(auth)])
async def decide(req: DecideRequest) -> dict:
    """JEV 互換: {"input": ..., "schema": {...}} → {各フィールド..., "confidence": ...}"""
    return await _run(req)


@app.post("/v1/systemone", dependencies=[Depends(auth)], include_in_schema=False)
async def systemone(req: DecideRequest) -> dict:
    return await _run(req)


@app.post("/v1/decide/batch", dependencies=[Depends(auth)])
async def decide_batch(req: BatchRequest) -> dict:
    t0 = time.perf_counter()
    base = req.defaults.model_dump(exclude_unset=True, by_alias=True) if req.defaults else {}

    async def one(item: DecideRequest) -> dict:
        merged = DecideRequest(**{**base, **item.model_dump(exclude_unset=True, by_alias=True)})
        try:
            return await _run(merged)
        except (engine.DecisionError, BackendError, HTTPException) as e:
            return {"error": getattr(e, "detail", None) or str(e)}

    results = await asyncio.gather(*(one(i) for i in req.items))
    return {"results": results, "latency_ms": round((time.perf_counter() - t0) * 1000)}


@app.post("/v1/decide/upload", dependencies=[Depends(auth)])
async def decide_upload(preset: str | None = Form(None), input: str | None = Form(None),
                        options: str | None = Form(None, description="DecideRequest の残りの項目を JSON で"),
                        files: list[UploadFile] = File(default_factory=list)) -> dict:
    extra = json.loads(options) if options else {}
    imgs = []
    for f in files:
        raw = await f.read()
        if raw:
            imgs.append(f"data:{f.content_type or 'image/png'};base64,{base64.b64encode(raw).decode()}")
    req = DecideRequest(**{**extra, "preset": preset or extra.get("preset"), "input": input or extra.get("input"),
                           "images": imgs or extra.get("images")})
    return await _run(req)


@app.post("/v1/vision", dependencies=[Depends(auth)])
async def set_vision(req: VisionRequest) -> dict:
    b = _backend()
    async with S.switch_lock:
        await b.set_vision(req.enabled)
        S.resize_sem()
    return {"vision": b.vision_enabled, "backend": b.info()}


@app.get("/v1/backends", dependencies=[Depends(auth)])
async def backends() -> dict:
    cfg = S.cfg.get("backends") or {}
    return {"active": S.backend.name if S.backend else None,
            "available": {k: {"kind": v.get("kind"), "model": v.get("model") or v.get("model_path"),
                              "vision": bool(v.get("mmproj_path") or v.get("vision"))} for k, v in cfg.items()}}


@app.post("/v1/backends/switch", dependencies=[Depends(auth)])
async def switch_backend(req: BackendSwitch) -> dict:
    async with S.switch_lock:
        S.cfg = load_config()
        vision = req.vision if req.vision is not None else (S.backend.vision_enabled if S.backend else False)
        await activate(req.name, vision)
    return {"backend": S.backend.info()}


def cli() -> None:
    ap = argparse.ArgumentParser(description="JevPalace Decision API")
    ap.add_argument("--host")
    ap.add_argument("--port", type=int)
    ap.add_argument("--backend", help="config.yaml の backends のキー")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--vision", action="store_true", help="画像モードで起動")
    g.add_argument("--no-vision", action="store_true", help="テキスト専用 (高速) で起動")
    a = ap.parse_args()
    if a.backend:
        os.environ["JEV_BACKEND"] = a.backend
    if a.vision:
        os.environ["JEV_VISION"] = "1"
    if a.no_vision:
        os.environ["JEV_VISION"] = "0"
    server = (load_config().get("server") or {})
    uvicorn.run(app, host=a.host or server.get("host", "127.0.0.1"), port=a.port or int(server.get("port", 8700)),
                log_level="info")


if __name__ == "__main__":
    cli()
