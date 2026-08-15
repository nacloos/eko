"""Tinker-backed model conversation for Eko's host."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Callable

import eko as core

CALL_TIMEOUT = 300
MAX_TOKENS = 8192
DEFAULT_MODEL = "thinkingmachines/Inkling-Small"
SESSION_VERSION = 3
EFFORT = {"low": .2, "medium": .7, "high": .9, "xhigh": .99, "max": .99}


def session_file(session_id: str) -> Path:
    root = Path(os.environ.get(
        "XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "eko" / "sessions" / f"{session_id}.json"


def _message(message: core.Message) -> dict:
    if any(not isinstance(part, core.Text) for part in message.content):
        raise ValueError("the Tinker model provider does not support image input")
    return {"role": message.role,
            "content": "".join(part.text for part in message.content)}


class Tinker:
    """A durable conversation sampled through Tinker's native Cookbook API."""

    def __init__(self, cwd: Path, model: str | None = None,
                 effort: str | None = None, session_id: str | None = None,
                 resume: bool = False, client: Any = None,
                 renderer: Any = None) -> None:
        self.cwd = cwd
        self.model = model or DEFAULT_MODEL
        self.effort = effort
        try:
            self.session_id = (str(uuid.UUID(session_id)) if session_id
                               else str(uuid.uuid4()))
        except ValueError as error:
            raise ValueError(f"invalid session ID: {session_id}") from error
        self.state_file = session_file(self.session_id)
        self.client = client
        self.renderer = renderer
        self.base_model: str | None = None
        self.service = None
        self.system: str | None = None
        self.messages: list[dict] = []
        self.interrupted = threading.Event()
        self.context_used = 0
        if resume:
            self._load(model)

    def _load(self, requested_model: str | None) -> None:
        try:
            state = json.loads(self.state_file.read_text())
        except FileNotFoundError as error:
            raise ValueError(f"session not found: {self.session_id}") from error
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"could not read session: {self.session_id}") from error
        if (not isinstance(state, dict)
                or state.get("version") != SESSION_VERSION
                or state.get("session_id") != self.session_id
                or not isinstance(state.get("model"), str)
                or not isinstance(state.get("base_model"), str)
                or not isinstance(state.get("system"), str)
                or not isinstance(state.get("messages"), list)
                or not isinstance(state.get("cwd"), str)):
            raise ValueError(f"invalid session: {self.session_id}")
        if Path(state["cwd"]) != self.cwd:
            raise ValueError(f"session belongs to {state['cwd']}, not {self.cwd}")
        if requested_model is not None and requested_model != state["model"]:
            raise ValueError(f"session uses {state['model']}, not {requested_model}")
        self.model = state["model"]
        self.base_model = state["base_model"]
        self.system = state["system"]
        self.effort = self.effort if self.effort is not None else state.get("effort")
        self.messages = state["messages"]

    def _save(self, messages: list[dict]) -> None:
        state = {
            "version": SESSION_VERSION, "session_id": self.session_id,
            "model": self.model, "base_model": self.base_model,
            "effort": self.effort, "cwd": str(self.cwd),
            "system": self.system, "messages": messages,
        }
        self.state_file.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        temporary = self.state_file.with_name(
            f".{self.session_id}.{uuid.uuid4()}.tmp")
        try:
            descriptor = os.open(
                temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w") as file:
                json.dump(state, file, separators=(",", ":"))
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, self.state_file)
        finally:
            temporary.unlink(missing_ok=True)

    def _connect(self) -> tuple[Any, Any]:
        if self.client is None:
            import tinker

            self.service = tinker.ServiceClient()
            target = ({"model_path": self.model}
                      if self.model.startswith("tinker://")
                      else {"base_model": self.model})
            self.client = self.service.create_sampling_client(**target)
        if self.base_model is None:
            self.base_model = self.client.get_base_model()
        if self.renderer is None:
            from tinker_cookbook import model_info, renderers
            from tinker_cookbook.tokenizer_utils import get_tokenizer

            name = model_info.get_recommended_renderer_name(self.base_model)
            self.renderer = renderers.get_renderer(
                name, get_tokenizer(self.base_model))
        return self.client, self.renderer

    @staticmethod
    def _text(message: dict) -> str:
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        return "\n".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type") == "text")

    def complete(self, system: str, message: core.Message,
                 on_text: Callable[[str], None]) -> core.Message:
        if message.role != "user":
            raise ValueError("model input must be a user message")
        if self.system is None:
            self.system = system
            self.messages = [{"role": "system", "content": system}]
        elif self.system != system:
            raise ValueError("session system context does not match this invocation")
        self.interrupted.clear()
        user = _message(message)
        client, renderer = self._connect()
        options = ({"effort": EFFORT[self.effort]}
                   if self.effort is not None else {})
        prompt = renderer.build_generation_prompt(
            [*self.messages, user], **options)

        import tinker

        future = client.sample(
            prompt, num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=MAX_TOKENS, stop=renderer.get_stop_sequences()))
        deadline = time.monotonic() + CALL_TIMEOUT
        while True:
            if self.interrupted.is_set():
                raise InterruptedError
            try:
                remaining = max(0, deadline - time.monotonic())
                response = future.result(timeout=min(.1, remaining))
                break
            except FutureTimeout:
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        f"model response timed out after {CALL_TIMEOUT}s")

        parsed, _termination = renderer.parse_response(response.sequences[0].tokens)
        assistant = {"role": "assistant", "content": parsed.get("content", "")}
        answer = self._text(assistant)
        on_text(answer)
        messages = [*self.messages, user, assistant]
        self._save(messages)
        self.messages = messages
        self.context_used = len(prompt.to_ints())
        return core.Message("assistant", (core.Text(answer),))

    def close(self) -> None:
        self.interrupt()

    def interrupt(self) -> None:
        self.interrupted.set()


def ensure_auth() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit(
            "TINKER_API_KEY is not set. Create a key at "
            "https://tinker-console.thinkingmachines.ai/keys")
