"""LLM バックエンドの抽象化。

- OpenAICompatBackend: OpenAI 互換の /v1/chat/completions を叩くだけ (外部の llama-server / vLLM / Ollama / LM Studio / OpenAI 等)
- LlamaCppBackend: 上記 + このフォルダ内の llama-server.exe を起動・停止・再起動して管理する。
  画像オフ時は --mmproj を付けずに起動するので、VRAM が空き・起動も推論も軽くなる。

モデルを替えるときは config.yaml の backends にエントリを足して active を切り替えるだけ。
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]  # D:\JevPalace


def _abs(p: str | None) -> str | None:
    if not p:
        return None
    path = Path(p)
    return str(path if path.is_absolute() else ROOT / path)


@dataclass
class ChatResult:
    content: str
    reasoning: str
    logprobs: list[dict] | None
    usage: dict[str, Any]
    timings: dict[str, Any] | None
    model: str


class BackendError(RuntimeError):
    pass


class OpenAICompatBackend:
    kind = "openai"

    def __init__(self, name: str, cfg: dict):
        self.name = name
        self.cfg = cfg
        self.base_url = cfg.get("base_url", "http://127.0.0.1:8080/v1").rstrip("/")
        key = cfg.get("api_key") or (os.environ.get(cfg["api_key_env"]) if cfg.get("api_key_env") else None)
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.model = cfg.get("model", "local")
        self.supports_logprobs = bool(cfg.get("logprobs", True))
        self.json_mode = cfg.get("json_mode", "json_schema")  # json_schema | json_object | none
        self.reasoning_style = cfg.get("reasoning_style", "qwen")  # qwen | openai | none
        self.timeout = float(cfg.get("timeout", 300))
        # llama.cpp 系は GBNF の "grammar" を受け付ける。思考オフ時に空白なしの出力を強制して速くする
        self.compact_grammar = bool(cfg.get("compact_grammar", self.kind == "llamacpp"))
        self.extra_body = cfg.get("extra_body") or {}
        self._vision_cfg = bool(cfg.get("vision", False))
        self.vision_enabled = self._vision_cfg
        self.client = httpx.AsyncClient(timeout=self.timeout)

    # --- capability / lifecycle -------------------------------------------------
    @property
    def vision_available(self) -> bool:
        return self._vision_cfg

    @property
    def parallel(self) -> int:
        return int(self.cfg.get("parallel", 4))

    async def start(self, vision: bool) -> None:
        self.vision_enabled = vision and self.vision_available
        h = await self.health()
        if not h["ok"]:
            raise BackendError(f"{self.base_url} に接続できません: {h.get('error', 'HTTP error')}")

    async def set_vision(self, enabled: bool) -> None:
        if enabled and not self.vision_available:
            raise BackendError(f"backend '{self.name}' は画像入力に対応していません (vision: false)")
        self.vision_enabled = enabled

    async def stop(self) -> None:
        await self.client.aclose()

    async def health(self) -> dict:
        try:
            r = await self.client.get(self.base_url + "/models", headers=self.headers, timeout=5)
            return {"ok": r.status_code == 200}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "error": str(e)}

    def info(self) -> dict:
        return {
            "name": self.name,
            "kind": self.kind,
            "model": self.model,
            "base_url": self.base_url,
            "vision_available": self.vision_available,
            "vision_enabled": self.vision_enabled,
            "logprobs": self.supports_logprobs,
            "parallel": self.parallel,
        }

    # --- request ---------------------------------------------------------------
    def _reasoning_params(self, reasoning: str) -> dict:
        if self.reasoning_style == "qwen":
            if reasoning == "off":
                return {"chat_template_kwargs": {"enable_thinking": False}}
            effort = {"low": "low", "medium": "medium", "high": "xhigh"}[reasoning]
            return {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": effort}}
        if self.reasoning_style == "openai":
            return {} if reasoning == "off" else {"reasoning_effort": reasoning}
        return {}

    async def wait_ready(self) -> None:
        return None

    async def chat(self, messages: list[dict], schema: dict | None, reasoning: str, max_tokens: int,
                   temperature: float = 0.0, top_logprobs: int = 10, grammar: str | None = None) -> ChatResult:
        await self.wait_ready()
        body: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            **self._reasoning_params(reasoning),
            **self.extra_body,
        }
        if grammar is not None:
            body["grammar"] = grammar
        elif schema is not None and self.json_mode == "json_schema":
            body["response_format"] = {"type": "json_schema", "json_schema": {"name": "decision", "strict": True, "schema": schema}}
        elif schema is not None and self.json_mode == "json_object":
            body["response_format"] = {"type": "json_object"}
        if self.supports_logprobs:
            body["logprobs"] = True
            body["top_logprobs"] = top_logprobs
        try:
            r = await self.client.post(self.base_url + "/chat/completions", json=body, headers=self.headers)
        except httpx.HTTPError as e:
            raise BackendError(f"バックエンドに接続できません: {e}") from e
        if r.status_code != 200:
            raise BackendError(f"backend HTTP {r.status_code}: {r.text[:500]}")
        data = r.json()
        ch = data["choices"][0]
        msg = ch.get("message") or {}
        return ChatResult(
            content=msg.get("content") or "",
            reasoning=msg.get("reasoning_content") or msg.get("reasoning") or "",
            logprobs=(ch.get("logprobs") or {}).get("content"),
            usage=data.get("usage") or {},
            timings=data.get("timings"),
            model=data.get("model") or self.model,
        )


def _kill_with_parent(proc: subprocess.Popen) -> None:
    """Windows: Job Object に入れて、API プロセスが強制終了されても llama-server が残らないようにする。"""
    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.OpenProcess.restype = wintypes.HANDLE

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD)]

    class IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_uint64) for n in ("R", "W", "O", "RT", "WT", "OT")]

    class EXT(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("Io", IO), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    global _JOB
    if "_JOB" not in globals():
        job = k32.CreateJobObjectW(None, None)
        info = EXT()
        info.Basic.LimitFlags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))
        _JOB = job  # ハンドルは API プロセス終了時に OS が閉じる → ジョブ内プロセスも終了
    h = k32.OpenProcess(0x0100 | 0x0001, False, proc.pid)  # PROCESS_SET_QUOTA | PROCESS_TERMINATE
    if h:
        k32.AssignProcessToJobObject(_JOB, h)
        k32.CloseHandle(h)


class LlamaCppBackend(OpenAICompatBackend):
    """このフォルダの llama-server を子プロセスとして管理する。"""

    kind = "llamacpp"

    def __init__(self, name: str, cfg: dict):
        cfg = dict(cfg)
        self.host = cfg.get("host", "127.0.0.1")
        self.port = int(cfg.get("port", 8090))
        cfg.setdefault("base_url", f"http://{self.host}:{self.port}/v1")
        cfg.setdefault("model", Path(cfg["model_path"]).stem)
        super().__init__(name, cfg)
        self.exe = _abs(cfg.get("server_exe", "llama/llama-server.exe"))
        self.model_path = _abs(cfg["model_path"])
        self.mmproj_path = _abs(cfg.get("mmproj_path"))
        self.proc: subprocess.Popen | None = None
        self.log_path = ROOT / "judge" / "logs" / f"llama-server-{name}.log"
        self._lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self.started_at: float | None = None
        self.load_seconds: float | None = None
        self.last_error: str | None = None

    @property
    def vision_available(self) -> bool:
        return bool(self.mmproj_path and Path(self.mmproj_path).exists())

    def _profile(self, vision: bool) -> dict:
        """画像オン/オフで別の起動パラメータを使えるようにする (オフ時は並列数・ctx を増やせる)。"""
        prof = dict(self.cfg.get("common_args") or {})
        prof.update(self.cfg.get("vision_profile" if vision else "text_profile") or {})
        return prof

    @property
    def parallel(self) -> int:
        return int(self._profile(self.vision_enabled).get("parallel", 4))

    def _cmd(self, vision: bool) -> list[str]:
        p = self._profile(vision)
        cmd = [self.exe, "-m", self.model_path,
               "--host", self.host, "--port", str(self.port),
               "-ngl", str(p.get("n_gpu_layers", 999)),
               "-c", str(p.get("ctx_size", 16384)),
               "-np", str(p.get("parallel", 4)),
               "--jinja", "--no-webui"]
        if p.get("flash_attn", True):
            cmd += ["-fa", "on"]
        if vision and self.mmproj_path:
            cmd += ["--mmproj", self.mmproj_path]
            if p.get("image_max_tokens"):
                cmd += ["--image-max-tokens", str(p["image_max_tokens"])]
        else:
            cmd += ["--no-mmproj"]
        for a in p.get("extra_args") or []:
            cmd.append(str(a))
        return cmd

    async def start(self, vision: bool) -> None:
        async with self._lock:
            await self._start_locked(vision and self.vision_available)

    async def _start_locked(self, vision: bool) -> None:
        self._ready.clear()
        self._kill()
        self.vision_enabled = vision
        if not Path(self.model_path).exists():
            raise BackendError(f"モデルが見つかりません: {self.model_path}")
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        t0 = time.time()
        self.proc = subprocess.Popen(self._cmd(vision), stdout=log, stderr=subprocess.STDOUT,
                                     cwd=str(ROOT), creationflags=flags)
        try:
            _kill_with_parent(self.proc)
        except Exception as e:  # noqa: BLE001
            print(f"[jev] job object setup failed: {e}")
        self.started_at = t0
        deadline = t0 + float(self.cfg.get("startup_timeout", 300))
        while time.time() < deadline:
            if self.proc.poll() is not None:
                self.last_error = f"llama-server が終了しました (code {self.proc.returncode})。ログ: {self.log_path}"
                raise BackendError(self.last_error)
            try:
                r = await self.client.get(f"http://{self.host}:{self.port}/health", timeout=2)
                if r.status_code == 200:
                    self.load_seconds = round(time.time() - t0, 2)
                    self.last_error = None
                    self._ready.set()
                    return
            except httpx.HTTPError:
                pass
            await asyncio.sleep(0.5)
        self._kill()
        self.last_error = "llama-server の起動がタイムアウトしました"
        raise BackendError(self.last_error)

    async def set_vision(self, enabled: bool) -> None:
        if enabled and not self.vision_available:
            raise BackendError(f"mmproj が見つからないため画像モードにできません: {self.mmproj_path}")
        async with self._lock:
            if enabled == self.vision_enabled and self.proc and self.proc.poll() is None:
                return
            await self._start_locked(enabled)

    async def wait_ready(self) -> None:
        if not self._ready.is_set():
            try:
                await asyncio.wait_for(self._ready.wait(), timeout=float(self.cfg.get("startup_timeout", 300)))
            except asyncio.TimeoutError as e:
                raise BackendError(self.last_error or "バックエンドが起動していません") from e

    def _kill(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    async def stop(self) -> None:
        self._ready.clear()
        self._kill()
        await super().stop()

    async def health(self) -> dict:
        alive = bool(self.proc and self.proc.poll() is None)
        return {"ok": alive and self._ready.is_set(), "process_alive": alive, "error": self.last_error}

    def info(self) -> dict:
        d = super().info()
        d.update({"model_path": self.model_path, "mmproj_path": self.mmproj_path,
                  "load_seconds": self.load_seconds, "pid": self.proc.pid if self.proc else None,
                  "profile": self._profile(self.vision_enabled)})
        return d


KINDS = {"llamacpp": LlamaCppBackend, "openai": OpenAICompatBackend}


def make_backend(name: str, cfg: dict) -> OpenAICompatBackend:
    kind = cfg.get("kind", "openai")
    if kind not in KINDS:
        raise ValueError(f"unknown backend kind: {kind}")
    return KINDS[kind](name, cfg)
