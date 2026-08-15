"""Model providers for Eko's host."""

from __future__ import annotations

import base64
import json
import select
import shutil
import signal
import subprocess
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


def ensure_tinker_auth() -> None:
    if not os.environ.get("TINKER_API_KEY"):
        raise SystemExit(
            "TINKER_API_KEY is not set. Create a key at "
            "https://tinker-console.thinkingmachines.ai/keys")


# ── Claude model connection ───────────────────────────────────────────────────

CALL_TIMEOUT = 300


def _claude_content(message: core.Message) -> list[dict]:
    """Serialize provider-neutral content as Claude blocks."""
    blocks: list[dict] = []
    for part in message.content:
        if isinstance(part, core.Text):
            blocks.append({"type": "text", "text": part.text})
        else:
            if part.name:
                blocks.append({"type": "text", "text": f"Image: {part.name}"})
            blocks.append({"type": "image", "source": {
                "type": "base64", "media_type": part.media_type,
                "data": base64.b64encode(part.data).decode(),
            }})
    return blocks


class Claude:
    """A persistent, tool-free connection to the LLM through the Claude CLI.

    Stream JSON lets several Eko turns share one model conversation. ``--safe-mode``
    prevents machine-specific instructions, hooks, plugins, and skills from changing
    the model's context, while ``--tools ''`` leaves generated Python as its only action.
    """

    def __init__(self, cwd: Path, model: str = "claude-opus-5",
                 effort: str = "high", session_id: str | None = None,
                 resume: bool = False) -> None:
        self.cwd = cwd
        self.model = model
        self.effort = effort
        self.session_id = session_id or str(uuid.uuid4())
        uuid.UUID(self.session_id)
        self.proc: subprocess.Popen[bytes] | None = None
        self.started = resume
        self.interrupted = threading.Event()
        self.context_used = 0

    def _repair_session(self) -> bool:
        """Repair this session's empty assistant text blocks."""
        config = Path(os.environ.get("CLAUDE_CONFIG_DIR", Path.home() / ".claude"))
        projects = config / "projects"
        if not projects.is_dir():
            return False
        for path in projects.glob(f"*/{self.session_id}.jsonl"):
            lines = path.read_text().splitlines(keepends=True)
            changed = False
            for index, line in enumerate(lines):
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if record.get("type") != "assistant":
                    continue
                content = record.get("message", {}).get("content", [])
                repaired = False
                for block in content:
                    if block.get("type") == "text" and block.get("text") == "":
                        block["text"] = " "
                        repaired = changed = True
                if repaired:
                    ending = "\n" if line.endswith("\n") else ""
                    lines[index] = json.dumps(record, separators=(",", ":")) + ending
            if changed:
                temporary = path.with_suffix(".jsonl.tmp")
                temporary.write_text("".join(lines))
                os.replace(temporary, path)
                return True
        return False

    def _start(self, system: str) -> None:
        session = (["--resume", self.session_id] if self.started else
                   ["--session-id", self.session_id])
        command = [
            "claude", "-p", "--verbose", "--safe-mode", "--tools", "",
            "--model", self.model, "--effort", self.effort,
            *session,
            "--input-format", "stream-json", "--output-format", "stream-json",
            "--include-partial-messages",
            "--system-prompt", system,
        ]
        self.proc = subprocess.Popen(
            command, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
        self.started = True

    def _terminate(self, signum: int, grace: float = 2) -> None:
        """Signal the CLI process group and ensure it is collected."""
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signum)
            except ProcessLookupError:
                pass
            try:
                self.proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.proc.wait()
        self.proc = None

    def complete(self, system: str, message: core.Message,
                 on_text: Callable[[str], None],
                 deadline: float | None = None,
                 retry_delay: float = .2) -> core.Message:
        """Complete a history using the CLI's internally persisted conversation."""
        if message.role != "user":
            raise ValueError("model input must be a user message")
        self.interrupted.clear()
        deadline = deadline or time.monotonic() + CALL_TIMEOUT
        resuming = self.started
        if self.proc is None or self.proc.poll() is not None:
            self._start(system)
        proc = self.proc
        assert proc is not None and proc.stdin and proc.stdout
        event = {"type": "user", "message": {
            "role": "user", "content": _claude_content(message)}}
        proc.stdin.write((json.dumps(event) + "\n").encode())
        proc.stdin.flush()

        parts: list[str] = []
        complete = ""
        while time.monotonic() < deadline:
            ready, _, _ = select.select(
                [proc.stdout], [], [], max(0, deadline - time.monotonic()))
            if not ready:
                break
            line = proc.stdout.readline()
            if not line:
                if self.interrupted.is_set():
                    raise InterruptedError
                break
            if not line.startswith(b"{"):
                continue
            data = json.loads(line)
            if data.get("type") == "stream_event":
                event = data.get("event", {})
                delta = event.get("delta", {})
                if (event.get("type") == "content_block_delta"
                        and delta.get("type") == "text_delta"):
                    text = delta.get("text", "")
                    parts.append(text)
                    on_text(text)
            elif data.get("type") == "assistant":
                complete = "".join(
                    block["text"] for block in data["message"].get("content", [])
                    if block.get("type") == "text")
            elif data.get("type") == "result":
                if data.get("is_error"):
                    detail = data.get("result") or data.get("error")
                    if resuming:
                        self._terminate(signal.SIGTERM)
                        proc.stdin.close()
                        proc.stdout.close()
                        if ("text content blocks must be non-empty" in str(detail)
                                and self._repair_session()):
                            return self.complete(
                                system, message, on_text, deadline, retry_delay)
                        remaining = deadline - time.monotonic()
                        if remaining > 0 and not parts and not complete:
                            delay = min(retry_delay, remaining)
                            if self.interrupted.wait(delay):
                                raise InterruptedError
                            if time.monotonic() < deadline:
                                return self.complete(
                                    system, message, on_text, deadline,
                                    min(retry_delay * 2, 5))
                        raise RuntimeError(
                            "Model session could not resume; context was not "
                            f"reset. {detail or ''}".rstrip())
                    raise RuntimeError(detail or "Model call failed")
                usage = data.get("usage") or {}
                self.context_used = (int(usage["prompt_tokens"])
                                     if usage.get("prompt_tokens") is not None else
                                     sum(int(usage.get(name) or 0) for name in (
                                         "input_tokens", "cache_read_input_tokens",
                                         "cache_creation_input_tokens")))
                return core.Message(
                    "assistant", (core.Text(complete or "".join(parts)),))
        raise RuntimeError("Model produced no result")

    def close(self) -> None:
        """Give the CLI a brief chance to flush its session, then stop it."""
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                assert proc.stdin
                proc.stdin.close()
                proc.stdin = None
                proc.wait(timeout=3)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self._terminate(signal.SIGTERM)
                return
        if proc.stdout:
            proc.stdout.close()
        if self.proc is proc:
            self.proc = None

    def interrupt(self) -> None:
        self.interrupted.set()
        self._terminate(signal.SIGKILL)


# ── Claude authentication ─────────────────────────────────────────────────────

def auth_status() -> bool:
    """Return whether the Claude CLI can access the model."""
    if shutil.which("claude") is None:
        raise SystemExit("Claude Code is not installed or is not on PATH.")
    status = subprocess.run(
        ["claude", "auth", "status", "--json"], text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    if status.returncode:
        return False
    try:
        data = json.loads(status.stdout)
        return bool(data.get("loggedIn"))
    except json.JSONDecodeError:
        return False


def ensure_claude_auth() -> None:
    """Let the official CLI own sign-in; Eko never reads or stores credentials."""
    if auth_status():
        return
    print("Claude Code is not signed in.")
    answer = input("Press Enter to sign in, or q to exit: ").strip().lower()
    if answer == "q":
        raise SystemExit(0)
    subprocess.run(["claude", "auth", "login", "--claudeai"], check=False)
    if not auth_status():
        raise SystemExit("Claude sign-in did not complete.")
