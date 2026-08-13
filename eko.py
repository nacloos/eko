# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "prompt-toolkit>=3.0,<4",
#     "rich>=13,<15",
# ]
# ///

"""Eko is a coding agent with almost no harness: an LLM, Python, and a persistent folder.

    uv run eko.py
    uv run eko.py --cwd ~/projects/my-project "Find and fix a bug"
    uv run eko.py --feral "Keep improving this project"

The whole agent is essentially:

    message = None
    while True:
        message = message or user()
        if message is None:
            break
        if (response := llm(message)).is_done():
            message = None
        else:
            message = run(response.python(), cwd)

Python is the only tool built into the harness. Through it, the model can inspect
the folder, use the shell, and write its own tools.

The loop only needs access to an LLM. This version uses Tinker's native Cookbook
renderer and sampling client, with generated Python as the model's only action.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import queue
import re
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import tokenize
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Protocol

from prompt_toolkit import Application
from prompt_toolkit.application import run_in_terminal
from prompt_toolkit.document import Document
from prompt_toolkit.formatted_text import ANSI
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import HSplit, Layout, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import TextArea
from rich.console import Console, Group
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text as RichText


NAME = "Eko"
SYSTEM = """You are {name}.
You are in {folder}.

Write a fenced ```python-run block to act. Stop your response after the block.
After your response ends, it runs in that folder. Its combined output returns in a
[python exit=N] section, where N is the process exit status. Fenced ```python and
```py blocks are displayed but not run.

All incoming information is sent to you as user-role messages. A message may contain
multiple sections, each beginning with a harness-written provenance header.
[terminal] is text entered by a terminal user. [python exit=N] is output from your
executed Python, where N is its exit status. [process-PID] is text or images sent by
a local process. [harness] is operational guidance. Never predict the contents of
these sections yourself.

Background processes can send later text or image inputs through EKO_SESSION, a
Unix stream socket using one JSON object per line. Send
{{"type":"input","content":[{{"type":"text","text":"done"}}]}}. Images use either
a workspace-relative "path", or base64 "data" with "media_type". Send
{{"type":"interrupt"}} to interrupt current work.{mode}
"""

NUDGE = "Write a fenced ```python-run block, or <done/> if the prompt is resolved."
FERAL_NUDGE = "Write a fenced ```python-run block."
NORMAL_MODE = (" If no action is needed, answer directly. When the prompt is "
               "fully resolved, end with <done/> and no Python block.")
FERAL_MODE = ("\n\nFeral mode: the user is gone; inputs come from Python or other "
              "processes.")
MAX_INPUT_TEXT = 20_000
TIMEOUT = 600
MAX_MESSAGE = 16 * 1024 * 1024
MAX_IMAGE = 5 * 1024 * 1024
MAX_IMAGES = 20


# ── Core agent ────────────────────────────────────────────────────────────────

@dataclass
class Result:
    """Completed execution of one model-written Python program."""

    output: str
    returncode: int
    elapsed: float


@dataclass(frozen=True)
class Event:
    """An observable core change: state, delta, response, result, or error."""

    type: str
    value: object = None


@dataclass(frozen=True)
class Text:
    """One ordered piece of text in an input."""

    text: str


@dataclass(frozen=True)
class Image:
    """One validated image in an input."""

    media_type: str
    data: bytes
    name: str | None = None


Content = Text | Image


@dataclass(frozen=True)
class Input:
    """An atomic multimodal input with harness-attested provenance."""

    source: str
    content: tuple[Content, ...]
    returncode: int | None = None


TERMINAL = "terminal"
PYTHON = "python"
HARNESS = "harness"


class Model(Protocol):
    """The model capability Eko needs, independent of any provider."""

    def ask(self, inputs: tuple[Input, ...], on_text: Callable[[str], None]) -> str: ...
    def interrupt(self) -> None: ...
    def close(self) -> None: ...


class Eko:
    """A running model conversation with an inbox and Python executor.

    The small public surface is ``start``, ``send``, ``interrupt``, ``stop``, and
    ``wait``. Core activity is observable through immutable ``Event`` values.
    Python subprocesses inherit ``EKO_SESSION`` and can send JSON lines back to
    this agent without access to its model credentials or internal state.
    """

    def __init__(self, cwd: Path, model: Model, feral: bool = False,
                 executor: Callable[[str, threading.Event], Result] | None = None,
                 socket_path: Path | None = None,
                 observer: Callable[[Event], None] | None = None) -> None:
        self.cwd = cwd
        self.feral = feral
        self.executor = executor
        self.observer = observer or (lambda _event: None)
        self.model = model
        self.inbox: queue.Queue[Input | None] = queue.Queue()
        self.stopping = threading.Event()
        self.interrupted = threading.Event()
        self.state = "idle"
        self.socket_path = (socket_path or
                             Path("/tmp") / f"eko-{uuid.uuid4().hex}.sock").resolve()
        self.listener: socket.socket | None = None
        self.listener_thread: threading.Thread | None = None
        self.thread: threading.Thread | None = None

    def start(self, prompt: str | None = None) -> None:
        """Start the socket listener and agent loop, optionally with initial text."""
        if self.thread is not None or self.stopping.is_set():
            raise RuntimeError("Eko can only be started once")
        self.socket_path.parent.mkdir(parents=True, exist_ok=True)
        self.socket_path.unlink(missing_ok=True)
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.socket_path))
        os.chmod(self.socket_path, 0o600)
        self.listener.listen()
        self.listener.settimeout(.2)
        self.listener_thread = threading.Thread(target=self._listen, daemon=True)
        self.thread = threading.Thread(target=self._run, daemon=True)
        if prompt:
            self.send(prompt)
        self.listener_thread.start()
        self.thread.start()

    def send(self, incoming: Input | str) -> None:
        """Put attributed input—or convenient terminal text—into the inbox."""
        if isinstance(incoming, str):
            incoming = Input(TERMINAL, (Text(incoming),))
        self.inbox.put(incoming)

    def interrupt(self) -> None:
        """Cancel the active model call or Python process, if any."""
        if self.state == "idle":
            return
        self.interrupted.set()
        if self.state == "thinking":
            self.model.interrupt()

    def stop(self) -> None:
        """Stop accepting work and release the model, listener, and socket path."""
        if self.stopping.is_set():
            return
        self.stopping.set()
        self.interrupted.set()
        self.inbox.put(None)
        self.model.interrupt()
        if self.listener is not None:
            self.listener.close()
        self.socket_path.unlink(missing_ok=True)

    def wait(self, timeout: float | None = None) -> None:
        """Wait for the agent thread to finish."""
        if self.thread is not None:
            self.thread.join(timeout)

    def status(self) -> str:
        pending = len(self.pending())
        return self.state + (f" · {pending} pending" if pending else "")

    def pending(self) -> list[str]:
        with self.inbox.mutex:
            events = [event for event in self.inbox.queue if event is not None]
        return [next((part.text for part in event.content
                      if isinstance(part, Text)), "[image]") for event in events]

    def _emit(self, event: Event) -> None:
        """Publish a core event to the optional presentation or embedding."""
        self.observer(event)

    def _set_state(self, state: str) -> None:
        self.state = state
        self._emit(Event("state", state))

    def _drain(self) -> tuple[Input, ...]:
        """Drain queued inputs so one model turn can batch them without losing origin."""
        pending: list[Input] = []
        while True:
            try:
                message = self.inbox.get_nowait()
            except queue.Empty:
                break
            if message is None:
                self.inbox.put(None)
                break
            pending.append(message)
        return tuple(pending)

    def _receive(self) -> tuple[Input, ...] | None:
        """Wait for one input, then batch everything already queued behind it."""
        self._set_state("idle")
        self.interrupted.clear()
        first = self.inbox.get()
        if first is None:
            return None
        return (first, *self._drain())

    def _execute(self, code: str) -> Result:
        self._set_state("running Python")
        if self.executor is not None:
            return self.executor(code, self.interrupted)
        env = os.environ.copy()
        env["EKO_SESSION"] = str(self.socket_path)
        return _run_python(code, self.cwd, self.interrupted, env=env)

    def _run(self) -> None:
        """Alternate attributed inputs, model responses, and Python execution."""
        inputs = None
        try:
            while True:
                if inputs is None:
                    inputs = self._receive()
                    if inputs is None:
                        return
                try:
                    self._set_state("thinking")
                    response = self.model.ask(
                        tuple(_limit_input(incoming) for incoming in inputs),
                        lambda text: self._emit(Event("delta", text)))
                    predicted = any(
                        kind == "prose" and re.search(
                            r"(?m)^\[(?:terminal|python(?: exit=-?\d+)?|"
                            r"process-\d+|harness)\]\s*$", text)
                        for kind, text, _closed in response_segments(response))
                    code = _python(response)
                    self._emit(Event("response", (response, code)))

                    if code is None and "<done/>" in response and not self.feral:
                        inputs = None
                    elif code is None:
                        inputs = (Input(HARNESS, (Text(
                            FERAL_NUDGE if self.feral else NUDGE),)),)
                    else:
                        result = self._execute(code)
                        self._emit(Event("result", result))
                        if self.interrupted.is_set():
                            inputs = None
                            continue
                        output = result.output or "(no output)"
                        inputs = (Input(
                            PYTHON, (Text(output),), result.returncode),)
                    if inputs is not None:
                        inputs += self._drain()
                        if predicted:
                            inputs += (Input(HARNESS, (Text(
                                "Warning: do not predict the contents of attributed "
                                "sections; wait for them to arrive."
                            ),)),)
                except InterruptedError:
                    self._emit(Event("error", "Interrupted"))
                    inputs = None
                except Exception as error:
                    if self.stopping.is_set():
                        return
                    self._emit(Event("error", str(error)))
                    inputs = None
        finally:
            self.model.close()

    # External processes use the same inbox through a small JSON-lines socket.
    def _listen(self) -> None:
        assert self.listener is not None
        while not self.stopping.is_set():
            try:
                connection, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._serve, args=(connection,), daemon=True).start()

    def _serve(self, connection: socket.socket) -> None:
        """Receive JSON and derive provenance from kernel-attested peer credentials."""
        with connection:
            try:
                credentials = connection.getsockopt(
                    socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
                pid, _uid, _gid = struct.unpack("3i", credentials)
                source = f"process-{pid}"
                reader = connection.makefile("rb")
                while line := reader.readline(MAX_MESSAGE + 1):
                    if self.stopping.is_set():
                        return
                    if len(line) > MAX_MESSAGE:
                        raise ValueError("session input is too large")
                    message = json.loads(line)
                    kind = message.get("type")
                    if kind == "input":
                        self.send(_parse_input(message, source, self.cwd))
                    elif kind == "interrupt":
                        self.interrupt()
                    else:
                        raise ValueError("unsupported session event type")
            except (ConnectionError, OSError):
                return
            except Exception as error:
                self._emit(Event("error", f"Agent input rejected: {error}"))

# ── Input and Python details ──────────────────────────────────────────────────

IMAGE_TYPES = {
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"\xff\xd8\xff": "image/jpeg",
    b"GIF87a": "image/gif",
    b"GIF89a": "image/gif",
}


def _image_type(data: bytes) -> str | None:
    kind = next((value for signature, value in IMAGE_TYPES.items()
                 if data.startswith(signature)), None)
    if kind is None and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        kind = "image/webp"
    return kind


def _parse_input(message: dict, source: str, cwd: Path) -> Input:
    """Validate untrusted socket JSON and normalize it into an ``Input``.

    Images may be inline base64 or workspace-relative paths. The caller supplies
    the source derived from the Unix connection; JSON cannot declare provenance.
    """
    if message.get("type") != "input" or not isinstance(message.get("content"), list):
        raise ValueError("expected an input event containing content")
    root = cwd.resolve()
    content: list[Content] = []
    images = 0
    for raw in message["content"]:
        if not isinstance(raw, dict):
            raise ValueError("content parts must be objects")
        if raw.get("type") == "text" and isinstance(raw.get("text"), str):
            content.append(Text(raw["text"]))
        elif raw.get("type") == "image":
            images += 1
            if images > MAX_IMAGES:
                raise ValueError(f"cannot submit more than {MAX_IMAGES} images")
            name = raw.get("name") if isinstance(raw.get("name"), str) else None
            if isinstance(raw.get("path"), str):
                relative = Path(raw["path"])
                if relative.is_absolute():
                    raise ValueError("image path must be relative")
                path = (root / relative).resolve(strict=True)
                if path != root and root not in path.parents:
                    raise ValueError("image path leaves the workspace")
                data = path.read_bytes()
                name = name or raw["path"]
                media_type = _image_type(data)
            elif isinstance(raw.get("data"), str):
                data = base64.b64decode(raw["data"], validate=True)
                media_type = raw.get("media_type")
                if media_type != _image_type(data):
                    raise ValueError("image media_type does not match its data")
            else:
                raise ValueError("image requires path or base64 data")
            if not data or len(data) > MAX_IMAGE or media_type is None:
                raise ValueError(f"unsupported image or size outside 1..{MAX_IMAGE}")
            content.append(Image(media_type, data, name))
        else:
            raise ValueError("unsupported content part")
    if not content:
        raise ValueError("input content cannot be empty")
    return Input(source, tuple(content))


OPEN_FENCE = re.compile(r"^[ \t]{0,3}(`{3,})[ \t]*python-run[ \t]*$")


def _opening_fence(line: str) -> int:
    """Return the backtick count for an executable Python fence."""
    match = OPEN_FENCE.fullmatch(line.rstrip("\r\n"))
    return len(match.group(1)) if match else 0


def _closing_fence(line: str, length: int) -> bool:
    """Whether this complete line closes a fence of ``length`` backticks."""
    return bool(re.fullmatch(
        rf"[ \t]{{0,3}}`{{{length},}}[ \t]*", line.rstrip("\r\n")))


def response_segments(text: str) -> list[tuple[str, str, bool]]:
    """Split prose and executable Python fences using Markdown's line rules."""
    segments: list[tuple[str, str, bool]] = []
    parts: list[str] = []
    fence = 0
    for line in text.splitlines(keepends=True):
        if not fence:
            if length := _opening_fence(line):
                if parts:
                    segments.append(("prose", "".join(parts), True))
                    parts = []
                fence = length
            else:
                parts.append(line)
        elif _closing_fence(line, fence):
            segments.append(("python", "".join(parts), True))
            parts = []
            fence = 0
        else:
            parts.append(line)
    if parts or fence:
        segments.append(("python" if fence else "prose", "".join(parts), not fence))
    return segments


def _python(text: str) -> str | None:
    blocks = [content for kind, content, closed in response_segments(text)
              if kind == "python" and closed]
    return "\n".join(blocks) if blocks else None


def _clip(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n\n… {omitted:,} characters omitted …\n\n{text[-half:]}"


def _limit_input(incoming: Input, limit: int = MAX_INPUT_TEXT) -> Input:
    """Limit all text in one input while preserving its multimodal order."""
    total = sum(len(part.text) for part in incoming.content
                if isinstance(part, Text))
    if total <= limit:
        return incoming

    head = limit // 2
    tail_start = total - (limit - head)
    position = 0
    marker = Text(f"\n\n… {total - limit:,} characters omitted …\n\n")
    content: list[Content] = []
    marked = False
    for part in incoming.content:
        if isinstance(part, Image):
            content.append(part)
            continue
        start, end = position, position + len(part.text)
        if start < head:
            kept = part.text[:max(0, min(end, head) - start)]
            if kept:
                content.append(Text(kept))
        if end > head and not marked:
            content.append(marker)
            marked = True
        if end > tail_start:
            kept = part.text[max(0, tail_start - start):]
            if kept:
                content.append(Text(kept))
        position = end
    return Input(incoming.source, tuple(content), incoming.returncode)


def _run_python(code: str, cwd: Path, interrupted: threading.Event, *,
               env: dict[str, str] | None = None,
               sandbox: bool = False) -> Result:
    """Run one model-written Python block in the persistent working folder."""
    python = cwd / ".venv/bin/python"
    executable = str(python if python.exists() else Path(sys.executable))
    command = [executable, "-u", "-c", code]
    if sandbox:
        bwrap = shutil.which("bwrap")
        if bwrap is None:
            raise RuntimeError("--sandbox requires Bubblewrap (bwrap)")
        session = Path((env or {})["EKO_SESSION"])
        sandbox_python = ("/workspace/.venv/bin/python" if python.exists()
                          else "/usr/bin/python3")
        command = [
            bwrap, "--die-with-parent", "--clearenv",
            "--unshare-user", "--unshare-ipc", "--unshare-uts",
            "--unshare-cgroup", "--unshare-net",
            "--ro-bind", "/usr", "/usr",
            "--symlink", "usr/bin", "/bin",
            "--symlink", "usr/lib", "/lib",
            "--symlink", "usr/lib64", "/lib64",
            "--symlink", "usr/sbin", "/sbin",
            "--bind", str(cwd), "/workspace",
            "--dev", "/dev", "--tmpfs", "/tmp",
            "--dir", "/run", "--ro-bind", str(session), "/run/eko.sock",
            "--setenv", "HOME", "/workspace", "--setenv", "TMPDIR", "/tmp",
            "--setenv", "PATH", "/usr/bin:/bin",
            "--setenv", "EKO_SESSION", "/run/eko.sock",
            "--chdir", "/workspace", sandbox_python, "-u", "-c", code,
        ]
    started = time.monotonic()
    proc = subprocess.Popen(
        command, cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        errors="replace", start_new_session=True, env=env)
    deadline = started + TIMEOUT
    while True:
        try:
            output, _ = proc.communicate(timeout=.1)
            break
        except subprocess.TimeoutExpired:
            if interrupted.is_set() or time.monotonic() >= deadline:
                os.killpg(proc.pid, signal.SIGKILL)
                output, _ = proc.communicate()
                reason = ("Interrupted" if interrupted.is_set()
                          else f"TIMEOUT after {TIMEOUT}s")
                output += f"\n{reason}"
                break
    return Result(output, proc.returncode, time.monotonic() - started)


# ── Claude model connection ───────────────────────────────────────────────────

CALL_TIMEOUT = 300


def _claude_content(events: tuple[Input, ...]) -> list[dict]:
    """Serialize attributed inputs as Claude text and image blocks."""
    blocks: list[dict] = []
    for index, event in enumerate(events):
        header = event.source
        if event.returncode is not None:
            header += f" exit={event.returncode}"
        prefix = "" if index == 0 else "\n\n"
        blocks.append({"type": "text", "text": f"{prefix}[{header}]\n"})
        for part in event.content:
            if isinstance(part, Text):
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

    def __init__(self, cwd: Path, model: str = "claude-opus-5", effort: str = "high",
                 feral: bool = False, name: str = NAME,
                 folder: str | Path | None = None) -> None:
        self.cwd = cwd
        self.folder = folder if folder is not None else cwd
        self.model = model
        self.effort = effort
        self.feral = feral
        self.name = name
        self.session_id = str(uuid.uuid4())
        self.proc: subprocess.Popen[bytes] | None = None
        self.started = False
        self.interrupted = threading.Event()

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

    def _start(self) -> None:
        session = (["--resume", self.session_id] if self.started else
                   ["--session-id", self.session_id])
        command = [
            "claude", "-p", "--verbose", "--safe-mode", "--tools", "",
            "--model", self.model, "--effort", self.effort,
            *session,
            "--input-format", "stream-json", "--output-format", "stream-json",
            "--include-partial-messages",
            "--system-prompt", SYSTEM.format(
                name=self.name, folder=self.folder,
                mode=FERAL_MODE if self.feral else NORMAL_MODE),
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

    def ask(self, events: tuple[Input, ...], on_text: Callable[[str], None],
            deadline: float | None = None, retry_delay: float = .2) -> str:
        """Send one message, forwarding text deltas while collecting the response."""
        self.interrupted.clear()
        deadline = deadline or time.monotonic() + CALL_TIMEOUT
        resuming = self.started
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        proc = self.proc
        assert proc is not None and proc.stdin and proc.stdout
        event = {"type": "user", "message": {
            "role": "user", "content": _claude_content(events)}}
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
                            return self.ask(events, on_text, deadline, retry_delay)
                        remaining = deadline - time.monotonic()
                        if remaining > 0 and not parts and not complete:
                            delay = min(retry_delay, remaining)
                            if self.interrupted.wait(delay):
                                raise InterruptedError
                            if time.monotonic() < deadline:
                                return self.ask(
                                    events, on_text, deadline,
                                    min(retry_delay * 2, 5))
                        raise RuntimeError(
                            "Model session could not resume; context was not "
                            f"reset. {detail or ''}".rstrip())
                    raise RuntimeError(detail or "Model call failed")
                return complete or "".join(parts)
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


def ensure_auth() -> None:
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


# ── Terminal rendering ────────────────────────────────────────────────────────

MAX_DISPLAY_OUTPUT = 4_000
MAX_DISPLAY_LINES = 5
GOLD = "#d7af5f"
GOLD_ACTIVE = "#e5bd68"


def response_renderable(text: str):
    """Render executable Python as panels and leave Markdown intact."""
    text = text.replace("<done/>", "")
    items = []
    for kind, content, _closed in response_segments(text):
        if kind == "prose" and content.strip():
            items.append(Markdown(content.strip()))
        elif kind == "python":
            items.append(Panel(
                Syntax(content.rstrip("\n") or " ", "python",
                       theme="ansi_dark", word_wrap=True),
                title=f"[bold {GOLD}]python[/bold {GOLD}]", title_align="left",
                border_style=GOLD, padding=(0, 1)))
    return Group(*items)


def display_output(text: str, width: int = 120) -> str:
    """Make a compact terminal preview while preserving model-facing output."""
    lines = text.splitlines()
    if len(lines) > MAX_DISPLAY_LINES:
        retained = MAX_DISPLAY_LINES - 1
        head = retained // 2
        tail = retained - head
        omitted = len(lines) - retained
        lines = lines[:head] + [f"… +{omitted} lines"] + lines[-tail:]
    lines = [line if len(line) <= width else line[:width - 1] + "…"
             for line in lines]
    return _clip("\n".join(lines), MAX_DISPLAY_OUTPUT)


class NativeStream:
    """Append stable model-output lines without redrawing terminal history."""

    def __init__(self, ui: UI) -> None:
        self.ui = ui
        self.text = ""
        self.buffer = ""
        self.pending_output: list[str] = []
        self.last_flush = time.monotonic()
        self.code = False
        self.fence_length = 0
        self.box_open = False
        self.prose = ""
        self.code_lines: list[str] = []
        self.code_emitted = 0

    def feed(self, delta: str) -> None:
        self.text += delta
        self.buffer += delta.replace("\x1b", "")
        self._drain()
        if time.monotonic() - self.last_flush >= .1:
            self._flush()

    def _queue(self, text: str) -> None:
        self.pending_output.append(text)

    def _flush(self) -> None:
        if self.pending_output:
            self.ui._append("".join(self.pending_output))
            self.pending_output.clear()
        self.last_flush = time.monotonic()

    def _drain(self, final: bool = False) -> None:
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            self._line(line + "\n")
        if final and self.buffer:
            line, self.buffer = self.buffer, ""
            self._line(line)
        if final:
            if self.code:
                self._commit_code(final=True)
                self._close_box()
                self.code = False
            else:
                self._prose("", final=True)

    def _line(self, line: str) -> None:
        if self.code:
            if _closing_fence(line, self.fence_length):
                self._commit_code(final=True)
                self._close_box()
                self.code = False
                self.fence_length = 0
            else:
                self.code_lines.append(line.rstrip("\r\n"))
                self._commit_code()
            return
        if length := _opening_fence(line):
            self._prose("", final=True)
            self.code = True
            self.fence_length = length
            self._open_box()
        else:
            self._prose(line)

    def _prose(self, text: str, final: bool = False) -> None:
        text = self._visible_prose(text).replace("<done/>", "")
        self.prose += text
        # A blank line closes a Markdown block. Keeping only the unfinished
        # block mutable prevents later tokens from restyling terminal history.
        while "\n\n" in self.prose:
            block, self.prose = self.prose.split("\n\n", 1)
            self._render_prose(block)
        if final and self.prose:
            self._render_prose(self.prose)
            self.prose = ""

    def _visible_prose(self, text: str) -> str:
        return text

    def _render_prose(self, text: str) -> None:
        if text.strip():
            self._queue(self.ui._render(Markdown(text.strip())) + "\n")

    def _width(self) -> int:
        return max(20, shutil.get_terminal_size().columns - 2)

    def _open_box(self) -> None:
        self.code_lines = []
        self.code_emitted = 0
        width = self._width()
        title = "─ python "
        self._queue(self.ui._render(RichText(
            "╭" + title + "─" * max(0, width - len(title) - 2) + "╮",
            style=f"bold {GOLD}")))
        self.box_open = True

    def _stable_code_lines(self) -> int:
        """Return the prefix whose Python highlighting cannot change later."""
        source = "\n".join(self.code_lines) + "\n"
        stable = len(self.code_lines)
        try:
            list(tokenize.generate_tokens(io.StringIO(source).readline))
        except (tokenize.TokenError, IndentationError, SyntaxError) as error:
            location = error.args[1] if len(error.args) > 1 else None
            if isinstance(location, tuple):
                stable = min(stable, max(0, location[0] - 1))
        return stable

    def _commit_code(self, final: bool = False) -> None:
        end = len(self.code_lines) if final else self._stable_code_lines()
        if end <= self.code_emitted:
            return
        source = "\n".join(self.code_lines) + "\n"
        highlighted = Syntax("", "python", theme="ansi_dark").highlight(source)
        lines = highlighted.split("\n")
        for line in lines[self.code_emitted:end]:
            self._code_line(line)
        self.code_emitted = end

    def _code_line(self, line: RichText) -> None:
        width = self._width()
        content = line[:max(1, width - 4)]
        padding = " " * max(0, width - len(content.plain) - 4)
        row = RichText()
        row.append("│ ", style=GOLD)
        row.append_text(content)
        row.append(padding)
        row.append(" │", style=GOLD)
        self._queue(self.ui._render(row))

    def _close_box(self) -> None:
        if self.box_open:
            width = self._width()
            self._queue(self.ui._render(RichText(
                "╰" + "─" * (width - 2) + "╯", style=GOLD)))
            self.box_open = False

    def finish(self) -> None:
        self._drain(final=True)
        self._close_box()
        self._flush()


class UI:
    """Commit completed messages above a small live composer."""

    def __init__(self) -> None:
        self.live = ""
        self.streamed_response = ""
        self.lock = threading.Lock()
        self.activity_stop: threading.Event | None = None
        self.activity_thread: threading.Thread | None = None
        self.status: Callable[[], str] = lambda: "idle"
        self.pending: Callable[[], list[str]] = lambda: []
        self.on_submit: Callable[[str], None] = lambda text: None
        self.on_interrupt: Callable[[], None] = lambda: None
        self.on_exit: Callable[[], None] = lambda: None
        self.on_start: Callable[[], None] = lambda: None
        self.keys = KeyBindings()

        self.transcript_window = Window(
            FormattedTextControl(self._transcript), height=1,
            wrap_lines=False, always_hide_cursor=True)
        self.composer = TextArea(
            height=1,
            prompt=[(f"{GOLD} bold", "› ")],
            multiline=True, wrap_lines=True)
        self.status_window = Window(FormattedTextControl(self._status), height=1)

        @self.keys.add("enter")
        def send(event) -> None:
            self._submit()

        @self.keys.add("c-c")
        def cancel(event) -> None:
            if self.composer.text:
                self.composer.buffer.set_document(Document("", 0), bypass_readonly=True)
            else:
                self.on_interrupt()

        @self.keys.add("c-d")
        def exit_app(event) -> None:
            if not self.composer.text:
                self.on_exit()

        @self.keys.add("escape")
        def interrupt(event) -> None:
            self.on_interrupt()

        root = HSplit([
            self.transcript_window,
            Window(height=1, char="─", style="class:rule"),
            self.composer,
            self.status_window,
        ])
        self.app: Application[None] = Application(
            layout=Layout(root, focused_element=self.composer),
            key_bindings=self.keys, full_screen=False, mouse_support=False,
            min_redraw_interval=.04,
            style=Style.from_dict({
                "rule": "#444444", "status": "#888888",
                "pending": "#888888", "pending.label": f"{GOLD} bold",
            }))
        self.app.ttimeoutlen = .1
        self.app.timeoutlen = .1

    def connect(self, agent: Eko) -> None:
        """Observe an Eko and route terminal controls back to it."""
        stream = None

        def render(event: Event) -> None:
            nonlocal stream
            if event.type == "state":
                state = str(event.value)
                if state == "thinking":
                    self._start_activity("thinking")
                elif state == "running Python":
                    self._start_activity("running")
                else:
                    self._stop_activity()
            elif event.type == "delta":
                if stream is None:
                    stream = NativeStream(self)
                stream.feed(str(event.value))
            elif event.type == "response":
                self._stop_activity()
                if stream is not None:
                    stream.finish()
                    stream = None
                response, code = event.value
                self.streamed_response = response
                self.assistant(response, code)
            elif event.type == "result":
                self._stop_activity()
                self.result(event.value)
            elif event.type == "error":
                self._stop_activity()
                if stream is not None:
                    stream.finish()
                    stream = None
                self.stopped(str(event.value))

        agent.observer = render
        self.status = agent.status
        self.pending = agent.pending
        self.on_submit = agent.send
        self.on_interrupt = agent.interrupt

    def _start_activity(self, label: str) -> None:
        """Show activity until output arrives, without involving the agent core."""
        self._stop_activity()
        stopped = self.activity_stop = threading.Event()
        frames = ("", ".", "..", "...", "..", ".")

        def animate() -> None:
            frame = 0
            while not stopped.is_set():
                rendered = self._render(RichText(
                    f"{label}{frames[frame % len(frames)]}",
                    style=f"dim {GOLD_ACTIVE}"))
                with self.lock:
                    if stopped.is_set():
                        return
                    self.live = rendered
                self.app.invalidate()
                frame += 1
                stopped.wait(.12)

        self.activity_thread = threading.Thread(target=animate, daemon=True)
        self.activity_thread.start()

    def _stop_activity(self) -> None:
        if self.activity_stop is not None:
            self.activity_stop.set()
            self.activity_stop = None
        thread, self.activity_thread = self.activity_thread, None
        if thread is not None and thread is not threading.current_thread():
            thread.join(.2)
        with self.lock:
            self.live = ""
        self.app.invalidate()

    def _submit(self) -> None:
        text = self.composer.text.strip()
        if not text:
            return
        self.composer.buffer.set_document(Document("", 0), bypass_readonly=True)
        if text == "/exit":
            self.on_exit()
            return
        self.user(text)
        self.on_submit(text)

    def _transcript(self):
        with self.lock:
            return ANSI(self.live)

    def _pending(self):
        lines = self.pending()
        if not lines:
            return []
        return [("class:pending.label", "queued  "),
                ("class:pending", "\n        ".join(lines[-3:]))]

    def _status(self):
        state = self.status()
        active = state.startswith(("thinking", "running"))
        hint = ("esc interrupt · enter send" if active else
                "enter send · ctrl+d exit")
        return [("class:status", f"  {hint}")]

    def _render(self, renderable) -> str:
        stream = io.StringIO()
        width = max(40, shutil.get_terminal_size().columns - 2)
        Console(file=stream, force_terminal=True, color_system="truecolor",
                highlight=False, width=width).print(renderable)
        return stream.getvalue()

    def _append(self, text: str) -> int:
        def write() -> None:
            sys.stdout.write(text)
            sys.stdout.flush()

        if not self.app.is_running or self.app.loop is None:
            write()
        elif threading.current_thread() is self.app._loop_thread:
            run_in_terminal(write)
        else:
            done = threading.Event()

            def schedule() -> None:
                task = run_in_terminal(write)
                task.add_done_callback(lambda _: done.set())

            self.app.loop.call_soon_threadsafe(schedule)
            done.wait(2)
        return 0

    def header(self, cwd: Path, model: str, name: str = NAME) -> None:
        header = Group(RichText(name, style=f"bold {GOLD}"), RichText.assemble(
            (str(cwd), "dim"), ("  ·  ", "dim"), (model, "dim"),
            ("  ·  signed in", "dim")), RichText(""))
        # Place the initial composer near the bottom. These are real terminal
        # lines, so subsequent turns naturally replace them and enter native
        # scrollback instead of making a full-screen spacer jump around.
        padding = max(
            0, shutil.get_terminal_size().lines - 9)
        self._append(self._render(header) + "\n" * padding)

    def user(self, text: str) -> None:
        rendered = self._render(RichText.assemble(("› ", f"bold {GOLD}"), text))
        self._append(rendered + "\n")

    def assistant(self, text: str, code: str | None) -> None:
        if text == self.streamed_response:
            self.streamed_response = ""
            return
        rendered = self._render(response_renderable(text))
        if rendered.strip():
            self._append(rendered)

    def result(self, result: Result) -> None:
        ok = result.returncode == 0
        style = "dim" if ok else "red"
        label = f"Exit {result.returncode} · {result.elapsed:.1f}s"
        width = max(40, shutil.get_terminal_size().columns - 4)
        output = display_output(result.output, width)
        body = RichText(output.rstrip() or "(no output)", style="dim")
        self._append(self._render(Group(RichText(label, style=style), body)) + "\n")

    def stopped(self, message: str = "Stopped") -> None:
        rendered = self._render(RichText(f"! {message}", style="#ff875f")) + "\n"
        self._append(rendered)

    def run(self) -> None:
        self.app.pre_run_callables.append(self.on_start)
        self.app.run()

    def exit(self) -> None:
        if self.app.is_running:
            self.app.exit()


def print_event(event: Event) -> None:
    """Render core events as plain output for headless operation."""
    if event.type == "delta":
        print(event.value, end="", flush=True)
    elif event.type == "response":
        print(flush=True)
    elif event.type == "result":
        result = event.value
        assert isinstance(result, Result)
        if result.output:
            print(result.output, end="" if result.output.endswith("\n") else "\n",
                  flush=True)
    elif event.type == "error":
        print(f"! {event.value}", file=sys.stderr, flush=True)


def run(cwd: Path, prompt: str | None = None, *, model: str = "claude-opus-5",
        effort: str = "high", feral: bool = False,
        executor: Callable[[str, threading.Event], Result] | None = None,
        name: str = NAME, folder: str | Path | None = None,
        headless: bool = False, socket_path: Path | None = None,
        sandbox: bool = False) -> None:
    """Run Eko with a TUI or as a plain process controlled by its session socket."""
    cwd = cwd.expanduser().resolve()
    if not cwd.is_dir():
        raise ValueError(f"not a directory: {cwd}")
    if sandbox and executor is not None:
        raise ValueError("sandbox and a custom executor are mutually exclusive")
    ensure_auth()
    llm = Claude(cwd, model, effort, feral, name,
                 "/workspace" if sandbox and folder is None else folder)
    agent = Eko(cwd, llm, feral, executor=executor, socket_path=socket_path,
                observer=print_event if headless else None)
    if sandbox:
        env = os.environ.copy()
        env["EKO_SESSION"] = str(agent.socket_path)
        agent.executor = lambda code, interrupted: _run_python(
            code, cwd, interrupted, env=env, sandbox=True)
    if headless:
        print(f"EKO_SESSION={agent.socket_path}", flush=True)
        agent.start(prompt)
        try:
            agent.wait()
        except KeyboardInterrupt:
            agent.interrupt()
            agent.wait()
        finally:
            agent.stop()
        return
    ui = UI()
    ui.header(cwd, model, name)
    if prompt:
        ui.user(prompt)
    ui.connect(agent)

    def exit_app() -> None:
        agent.stop()
        ui.exit()

    ui.on_exit = exit_app
    ui.on_start = lambda: agent.start(prompt)
    try:
        ui.run()
    except KeyboardInterrupt:
        agent.stop()
    finally:
        agent.stop()
    agent.wait(5)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="prompt to run; opens an input if omitted")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="working directory")
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--headless", action="store_true",
                        help="run without the terminal UI")
    parser.add_argument("--sandbox", action="store_true",
                        help="run generated Python in a Bubblewrap sandbox")
    parser.add_argument("--session-socket", type=Path,
                        help="path for the JSON-lines session socket")
    parser.add_argument(
        "--feral", action="store_true",
        help="keep acting without a completion state until interrupted")
    args = parser.parse_args()

    cwd = args.cwd.expanduser().resolve()
    if not cwd.is_dir():
        parser.error(f"not a directory: {cwd}")
    try:
        run(cwd, args.prompt, model=args.model, effort=args.effort, feral=args.feral,
            name=args.name, headless=args.headless, socket_path=args.session_socket,
            sandbox=args.sandbox)
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
