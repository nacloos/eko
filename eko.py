# /// script
# requires-python = ">=3.10"
# dependencies = []
# ///

"""A coding agent with almost no harness: an LLM, Python, and a folder.

``Eko`` is the embeddable agent. This file can also run it as a child process:

    EKO_MODEL=/path/to/model.sock python eko.py --cwd /path/to/project

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

The process interface reads commands from stdin, writes events to stdout, and
uses a model conversation through a JSON-lines Unix socket. It has no provider or
presentation dependencies.
"""
from __future__ import annotations

import base64
import json
import os
import queue
import re
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

NAME = "Eko"
SYSTEM = """You are {name}.
You are in {folder}.

Write a fenced ```python-run block to act. Stop your response after the block.
After your response ends, it runs in that folder. Its combined output returns in a
[python exit=N] section, where N is the process exit status. Fenced ```python and
```py blocks are displayed but not run.

All incoming information is sent to you as user-role messages. A message may contain
multiple sections, each beginning with a harness-written provenance header.
[terminal] is text entered by a controlling user. [python exit=N] is output from your
executed Python, where N is its exit status. [process-PID] is text or images sent by
a local process. [harness] is operational guidance. Never predict the contents of
these sections yourself.

Background processes can send later text or image inputs through EKO_SESSION, a
Unix stream socket using one JSON object per line. Send
{{"type":"input","content":[{{"type":"text","text":"done"}}]}}. Images use either
a workspace-relative "path", or base64 "data" with "media_type". Send
{{"type":"interrupt"}} to interrupt current work.

EKO_AGENT points to this agent's own executable.{mode}
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
ACTIVE_CHILDREN: set[int] = set()
CHILDREN_LOCK = threading.Lock()


# ── Core agent ────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
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


@dataclass(frozen=True)
class Message:
    """One provider-neutral conversation message."""

    role: str
    content: tuple[Content, ...]


TERMINAL = "terminal"
PYTHON = "python"
HARNESS = "harness"


class Eko:
    """A running model conversation with an inbox and Python executor.

    The small public surface is ``start``, ``send``, ``interrupt``, ``stop``, and
    ``wait``. Core activity is observable through immutable ``Event`` values.
    Python subprocesses inherit ``EKO_SESSION`` and can send JSON lines back to
    this agent without access to its model credentials or internal state.
    """

    def __init__(self, cwd: Path, model, feral: bool = False,
                 socket_path: Path | None = None,
                 observer: Callable[[Event], None] | None = None,
                 name: str = NAME) -> None:
        self.cwd = cwd.resolve()
        self.feral = feral
        self.observer = observer or (lambda _event: None)
        self.model = model
        mode = FERAL_MODE if feral else NORMAL_MODE
        self.system = SYSTEM.format(
            name=name, folder=self.cwd, mode=mode)
        self.messages: list[Message] = []
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
        self.model.start(self.system)
        if prompt:
            self.send(prompt)
        self.listener_thread.start()
        self.thread.start()

    def send(self, incoming: Input | str) -> None:
        """Put attributed input—or convenient controlling-user text—into the inbox."""
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
        """Wait for the agent loop and its inbox listener to finish."""
        if self.thread is not None:
            self.thread.join(timeout)
        if (self.listener_thread is not None
                and self.listener_thread is not threading.current_thread()):
            self.listener_thread.join(timeout)

    def status(self) -> str:
        """Describe current and queued work for an observer."""
        queued = len(self.pending())
        return self.state + (f" · {queued} pending" if queued else "")

    def pending(self) -> list[str]:
        """Return short descriptions of inputs waiting behind active work."""
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
        env = os.environ.copy()
        env["EKO_SESSION"] = str(self.socket_path)
        env["EKO_AGENT"] = str(Path(__file__).resolve())
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
                    message = user_message(tuple(
                        limit_input(incoming) for incoming in inputs))
                    reply = self.model.send(
                        message,
                        lambda text: self._emit(Event("delta", text)))
                    if reply.role != "assistant":
                        raise ValueError("model must return an assistant message")
                    response = message_text(reply)
                    self.messages.extend((message, reply))
                    predicted = any(
                        kind == "prose" and re.search(
                            r"(?m)^\[(?:terminal|python(?: exit=-?\d+)?|"
                            r"process-\d+|harness)\]\s*$", text)
                        for kind, text, _closed in response_segments(response))
                    code = executable_python(response)
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
                        self.send(decode_input(message, source, self.cwd))
                    elif kind == "interrupt":
                        self.interrupt()
                    else:
                        raise ValueError("unsupported session event type")
            except (ConnectionError, OSError):
                return
            except Exception as error:
                self._emit(Event("error", f"Agent input rejected: {error}"))

# ── Input decoding ───────────────────────────────────────────────────────────

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


def decode_input(message: dict, source: str, cwd: Path) -> Input:
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


# ── Response and conversation encoding ───────────────────────────────────────

OPEN_FENCE = re.compile(r"^[ \t]{0,3}(`{3,})[ \t]*python-run[ \t]*$")


def opening_fence(line: str) -> int:
    """Return the backtick count for an executable Python fence."""
    match = OPEN_FENCE.fullmatch(line.rstrip("\r\n"))
    return len(match.group(1)) if match else 0


def closing_fence(line: str, length: int) -> bool:
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
            if length := opening_fence(line):
                if parts:
                    segments.append(("prose", "".join(parts), True))
                    parts = []
                fence = length
            else:
                parts.append(line)
        elif closing_fence(line, fence):
            segments.append(("python", "".join(parts), True))
            parts = []
            fence = 0
        else:
            parts.append(line)
    if parts or fence:
        segments.append(("python" if fence else "prose", "".join(parts), not fence))
    return segments


def executable_python(text: str) -> str | None:
    blocks = [content for kind, content, closed in response_segments(text)
              if kind == "python" and closed]
    return "\n".join(blocks) if blocks else None


def limit_input(incoming: Input, limit: int = MAX_INPUT_TEXT) -> Input:
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


def user_message(inputs: tuple[Input, ...]) -> Message:
    """Combine attributed inputs into one provider-neutral user message."""
    content: list[Content] = []
    for index, incoming in enumerate(inputs):
        header = incoming.source
        if incoming.returncode is not None:
            header += f" exit={incoming.returncode}"
        prefix = "" if index == 0 else "\n\n"
        content.append(Text(f"{prefix}[{header}]\n"))
        content.extend(incoming.content)
    return Message("user", tuple(content))


def message_text(message: Message) -> str:
    """Return the text of a message in content order."""
    return "".join(part.text for part in message.content if isinstance(part, Text))


# ── Python execution ─────────────────────────────────────────────────────────

def _run_python(code: str, cwd: Path, interrupted: threading.Event, *,
               env: dict[str, str] | None = None) -> Result:
    """Run one model-written Python block in the persistent working folder."""
    python = cwd / ".venv/bin/python"
    executable = str(python if python.exists() else Path(sys.executable))
    started = time.monotonic()
    with CHILDREN_LOCK:
        proc = subprocess.Popen(
            [executable, "-u", "-c", code], cwd=cwd, stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            errors="replace", start_new_session=True, env=env)
        ACTIVE_CHILDREN.add(proc.pid)
    try:
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
    finally:
        with CHILDREN_LOCK:
            ACTIVE_CHILDREN.discard(proc.pid)
    return Result(output, proc.returncode, time.monotonic() - started)


def _reap_children() -> None:
    """Reap orphaned background processes when Eko is namespace PID 1."""
    children = Path("/proc/1/task/1/children")
    while True:
        try:
            pids = [int(pid) for pid in children.read_text().split()]
        except OSError:
            return
        with CHILDREN_LOCK:
            for pid in pids:
                if pid not in ACTIVE_CHILDREN:
                    try:
                        os.waitpid(pid, os.WNOHANG)
                    except ChildProcessError:
                        pass
        time.sleep(.2)


def _shutdown_children(grace: float = 1) -> None:
    """Terminate and reap every remaining child before namespace PID 1 exits."""
    children = Path("/proc/1/task/1/children")

    def pids() -> list[int]:
        try:
            return [int(pid) for pid in children.read_text().split()]
        except OSError:
            return []

    def reap() -> None:
        while True:
            try:
                if os.waitpid(-1, os.WNOHANG) == (0, 0):
                    return
            except ChildProcessError:
                return

    deadline = time.monotonic() + grace
    while pids() and time.monotonic() < deadline:
        for pid in pids():
            try:
                os.kill(pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
        reap()
        time.sleep(.02)
    deadline = time.monotonic() + grace
    while pids() and time.monotonic() < deadline:
        for pid in pids():
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        reap()
        time.sleep(.02)
    reap()


# ── Model transport ──────────────────────────────────────────────────────────

def encode_message(message: Message) -> dict:
    """Encode a provider-neutral message for the model socket."""
    content = []
    for part in message.content:
        if isinstance(part, Text):
            content.append({"type": "text", "text": part.text})
        else:
            content.append({"type": "image", "media_type": part.media_type,
                            "data": base64.b64encode(part.data).decode(),
                            "name": part.name})
    return {"role": message.role, "content": content}


def decode_message(raw: dict) -> Message:
    """Decode one trusted message from the private model socket."""
    content: list[Content] = []
    for part in raw["content"]:
        if part["type"] == "text":
            content.append(Text(part["text"]))
        else:
            content.append(Image(part["media_type"], base64.b64decode(part["data"]),
                                 part.get("name")))
    return Message(raw["role"], tuple(content))


class Model:
    """One model conversation carried by one Unix socket connection."""

    def __init__(self, endpoint: Path) -> None:
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(str(endpoint))
        self.reader = self.socket.makefile("rb")
        self.write_lock = threading.Lock()

    def _send(self, value: dict) -> None:
        with self.write_lock:
            self.socket.sendall((json.dumps(value, separators=(",", ":")) + "\n").encode())

    def start(self, system: str) -> None:
        self._send({"system": system})

    def send(self, message: Message,
             on_text: Callable[[str], None]) -> Message:
        self._send({"message": encode_message(message)})
        while line := self.reader.readline():
            event = json.loads(line)
            if "delta" in event:
                on_text(event["delta"])
            elif "message" in event:
                return decode_message(event["message"])
            elif "error" in event:
                if event.get("interrupted"):
                    raise InterruptedError
                raise RuntimeError(event["error"])
        raise RuntimeError("model service disconnected")

    def interrupt(self) -> None:
        try:
            self._send({"interrupt": True})
        except OSError:
            pass

    def close(self) -> None:
        try:
            self.reader.close()
        finally:
            self.socket.close()


# ── Process interface ────────────────────────────────────────────────────────


def encode_event(event: Event) -> dict:
    """Encode an observable agent event for a controlling process."""
    if event.type == "response":
        response, code = event.value
        return {"type": "response", "text": response, "python": code}
    if event.type == "result":
        result = event.value
        assert isinstance(result, Result)
        return {"type": "result", "output": result.output,
                "returncode": result.returncode, "elapsed": result.elapsed}
    return {"type": event.type, "value": event.value}


def decode_event(raw: dict) -> Event:
    """Decode an event produced by :func:`encode_event`."""
    kind = raw["type"]
    if kind == "response":
        return Event(kind, (raw["text"], raw.get("python")))
    if kind == "result":
        return Event(kind, Result(
            raw["output"], raw["returncode"], raw["elapsed"]))
    return Event(kind, raw.get("value"))


def serve(cwd: Path, model_socket: Path, *, prompt: str | None = None,
          feral: bool = False, name: str = NAME,
          socket_path: Path | None = None) -> None:
    """Run the agent using JSON-lines stdin, stdout, and model socket."""
    if os.getpid() == 1:
        threading.Thread(target=_reap_children, daemon=True).start()
    write_lock = threading.Lock()

    def write(value: dict) -> None:
        data = json.dumps(value, separators=(",", ":")) + "\n"
        with write_lock:
            sys.stdout.write(data)
            sys.stdout.flush()

    agent = Eko(
        cwd, Model(model_socket), feral, socket_path=socket_path,
        observer=lambda event: write(encode_event(event)), name=name)
    agent.start(prompt)
    write({"type": "ready", "session": str(agent.socket_path)})
    try:
        for line in sys.stdin:
            try:
                message = json.loads(line)
                kind = message.get("type")
                if kind == "input":
                    agent.send(decode_input(message, TERMINAL, cwd))
                elif kind == "interrupt":
                    agent.interrupt()
                elif kind == "stop":
                    break
                else:
                    raise ValueError("unsupported command")
            except Exception as error:
                write({"type": "error", "value": f"Command rejected: {error}"})
    except KeyboardInterrupt:
        agent.interrupt()
    finally:
        agent.stop()
        agent.wait(5)
        if os.getpid() == 1:
            _shutdown_children()


def main() -> None:
    """Run the agent behind its provider-neutral process interface."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?")
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    parser.add_argument("--model-socket", type=Path,
                        default=os.environ.get("EKO_MODEL"))
    parser.add_argument("--session-socket", type=Path)
    parser.add_argument("--name", default=NAME)
    parser.add_argument("--feral", action="store_true")
    args = parser.parse_args()
    cwd = args.cwd.expanduser().resolve()
    if not cwd.is_dir():
        parser.error(f"not a directory: {cwd}")
    if args.model_socket is None:
        parser.error("no model service; set EKO_MODEL")
    serve(cwd, args.model_socket, prompt=args.prompt, feral=args.feral,
          name=args.name, socket_path=args.session_socket)


if __name__ == "__main__":
    main()
