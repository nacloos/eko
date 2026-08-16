# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "prompt-toolkit>=3.0,<4",
#     "rich>=13,<15",
#     "tinker-cookbook>=0.1,<1",
# ]
# ///

"""Host Eko's model connection, sandbox, and terminal presentation."""

from __future__ import annotations

import argparse
import io
import json
import os
import queue
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import eko as core
import models
from models import (Claude, Tinker, _claude_content, ensure_claude_auth,
                    ensure_tinker_auth)
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

# ── Terminal rendering ────────────────────────────────────────────────────────

MAX_DISPLAY_OUTPUT = 4_000
MAX_DISPLAY_LINES = 5
GOLD = "#d7af5f"
GOLD_ACTIVE = "#e5bd68"


@dataclass(frozen=True)
class SandboxMount:
    source: Path
    target: Path
    readonly: bool = False


def parse_mount(value: str) -> SandboxMount:
    """Parse Docker/Podman-style --mount type=bind,... syntax."""
    options: dict[str, str] = {}
    flags: set[str] = set()
    aliases = {"src": "source", "dst": "target", "destination": "target"}
    for item in value.split(","):
        if "=" in item:
            key, option = item.split("=", 1)
            key = aliases.get(key, key)
            if key in options:
                raise argparse.ArgumentTypeError(f"duplicate mount option: {key}")
            options[key] = option
        else:
            flags.add(item)
    unknown = set(options) - {"type", "source", "target", "readonly", "ro"}
    unknown |= flags - {"readonly", "ro"}
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown mount option: {sorted(unknown)[0]}")
    if options.get("type") != "bind":
        raise argparse.ArgumentTypeError("mount type must be bind")
    if not options.get("source") or not options.get("target"):
        raise argparse.ArgumentTypeError("mount requires source and target")
    source = Path(options["source"]).expanduser().resolve()
    if not source.exists():
        raise argparse.ArgumentTypeError(f"mount source does not exist: {source}")
    target = Path(options["target"])
    try:
        relative = target.relative_to("/workspace")
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "mount target must be inside /workspace") from error
    if relative == Path("."):
        raise argparse.ArgumentTypeError("mount target cannot replace /workspace")
    if ".." in relative.parts:
        raise argparse.ArgumentTypeError("mount target cannot escape /workspace")
    boolean = options.get("readonly", options.get("ro"))
    if boolean is not None and boolean.lower() not in {"true", "false"}:
        raise argparse.ArgumentTypeError("readonly must be true or false")
    readonly = bool(flags & {"readonly", "ro"}) or boolean == "true"
    return SandboxMount(source, target, readonly)


def response_renderable(text: str):
    """Render assistant Markdown without its completion marker."""
    return Markdown(text.replace("<done/>", "").strip())


def python_renderable(code: str):
    """Render one completed native Python tool call."""
    return Panel(
        Syntax(code.rstrip("\n") or " ", "python", theme="ansi_dark", word_wrap=True),
        title=f"[bold {GOLD}]python[/bold {GOLD}]", title_align="left",
        border_style=GOLD, padding=(0, 1))


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
    text = "\n".join(lines)
    if len(text) <= MAX_DISPLAY_OUTPUT:
        return text
    half = MAX_DISPLAY_OUTPUT // 2
    omitted = len(text) - MAX_DISPLAY_OUTPUT
    return (f"{text[:half]}\n\n… {omitted:,} characters omitted …\n\n"
            f"{text[-half:]}")


class NativeStream:
    """Append stable Markdown blocks without redrawing terminal history."""

    def __init__(self, ui: UI) -> None:
        self.ui = ui
        self.text = ""
        self.buffer = ""
        self.pending_output: list[str] = []
        self.last_flush = time.monotonic()
        self.prose = ""

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
            self._prose(line + "\n")
        if final and self.buffer:
            line, self.buffer = self.buffer, ""
            self._prose(line)
        if final:
            self._prose("", final=True)

    def _prose(self, text: str, final: bool = False) -> None:
        text = text.replace("<done/>", "")
        self.prose += text
        # A blank line closes a Markdown block. Keeping only the unfinished
        # block mutable prevents later tokens from restyling terminal history.
        while "\n\n" in self.prose:
            block, self.prose = self.prose.split("\n\n", 1)
            self._render_prose(block)
        if final and self.prose:
            self._render_prose(self.prose)
            self.prose = ""

    def _render_prose(self, text: str) -> None:
        if text.strip():
            self._queue(self.ui._render(Markdown(text.strip())) + "\n")

    def finish(self) -> None:
        self._drain(final=True)
        self._flush()


class UI:
    """Commit completed messages above a small live composer."""

    def __init__(self, context: int = 0) -> None:
        self.live = ""
        self.streamed_response = ""
        self.lock = threading.Lock()
        self.activity_stop: threading.Event | None = None
        self.activity_thread: threading.Thread | None = None
        self.status: Callable[[], str] = lambda: "idle"
        self.context_capacity = context
        self.context_used = 0
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
            }))
        self.app.ttimeoutlen = .1
        self.app.timeoutlen = .1

    def connect(self, agent) -> None:
        """Observe an Eko and route terminal controls back to it."""
        stream = None

        def render(event: core.Event) -> None:
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
                if code is not None:
                    self._append(self._render(python_renderable(code)) + "\n")
                elif response:
                    self.assistant(response)
            elif event.type == "result":
                self._stop_activity()
                self.result(event.value)
            elif event.type == "context":
                self.context_used, self.context_capacity = map(int, event.value)
                self.app.invalidate()
            elif event.type == "error":
                self._stop_activity()
                if stream is not None:
                    stream.finish()
                    stream = None
                self.stopped(str(event.value))

        agent.observer = render
        self.status = agent.status
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

    def _status(self):
        state = self.status()
        active = state.startswith(("thinking", "running"))
        hint = ("esc interrupt · enter send" if active else
                "enter send · ctrl+d exit")
        context = (f" · {core.context_status_line(self.context_used, self.context_capacity)}"
                   if self.context_capacity else "")
        return [("class:status", f"  {hint}{context}")]

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

    def header(self, cwd: Path, model: str, name: str = core.NAME) -> None:
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

    def assistant(self, text: str) -> None:
        if text == self.streamed_response:
            self.streamed_response = ""
            return
        rendered = self._render(response_renderable(text))
        if rendered.strip():
            self._append(rendered)

    def result(self, result: core.Result) -> None:
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




# ── Agent process and model service ──────────────────────────────────────────

class AgentProcess:
    """A local proxy for an eko.py process."""

    def __init__(
        self,
        command: list[str],
        *,
        env: dict[str, str] | None = None,
    ) -> None:
        self.proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True, bufsize=1, env=env)
        self.observer: Callable[[core.Event], None] = lambda _event: None
        self.state = "starting"
        self.ready = threading.Event()
        self.session: str | None = None
        self._write_lock = threading.Lock()
        self._diagnostic_lock = threading.Lock()
        self._startup_diagnostics: list[str] = []
        self._requests: dict[str, queue.Queue[dict]] = {}
        self._requests_lock = threading.Lock()
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._errors = threading.Thread(target=self._read_errors, daemon=True)
        self._reader.start()
        self._errors.start()

    def _read(self) -> None:
        assert self.proc.stdout
        try:
            for line in self.proc.stdout:
                raw = json.loads(line)
                request_id = raw.get("request_id")
                if request_id is not None:
                    with self._requests_lock:
                        response = self._requests.get(request_id)
                    if response is not None:
                        response.put(raw)
                        continue
                event = core.decode_event(raw)
                kind = event.type
                if kind == "ready":
                    self.session = raw["session"]
                    self.ready.set()
                else:
                    if kind == "state":
                        self.state = str(event.value)
                    self.observer(event)
        except Exception as error:
            message = f"Agent connection failed: {error}"
            self._record_startup_diagnostic(message)
            self.observer(core.Event("error", message))
        finally:
            self.ready.set()

    def _read_errors(self) -> None:
        assert self.proc.stderr
        for line in self.proc.stderr:
            message = line.rstrip()
            self._record_startup_diagnostic(message)
            self.observer(core.Event("error", message))

    def _record_startup_diagnostic(self, message: str) -> None:
        if self.session is None and message:
            with self._diagnostic_lock:
                self._startup_diagnostics.append(message)

    def startup_error(self) -> str:
        """Return diagnostics emitted before the agent became ready."""
        if self.proc.poll() is not None:
            self._reader.join(.2)
            self._errors.join(.2)
        with self._diagnostic_lock:
            details = "\n".join(self._startup_diagnostics).strip()
        return "agent did not start" + (f":\n{details}" if details else "")

    def _send(self, value: dict) -> None:
        assert self.proc.stdin
        with self._write_lock:
            self.proc.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()

    def send(self, text: str) -> None:
        self._send({"type": "input", "content": [{"type": "text", "text": text}]})

    def _request(self, value: dict, timeout: float = 10) -> dict:
        request_id = uuid.uuid4().hex
        response: queue.Queue[dict] = queue.Queue(maxsize=1)
        with self._requests_lock:
            self._requests[request_id] = response
        try:
            self._send(dict(value, request_id=request_id))
            try:
                reply = response.get(timeout=timeout)
            except queue.Empty as error:
                raise TimeoutError("agent process request timed out") from error
        finally:
            with self._requests_lock:
                self._requests.pop(request_id, None)
        if reply["type"] == "error":
            raise RuntimeError(reply["value"])
        return reply

    def start_process(
        self, argv: list[str], *, cwd: str | None = None,
        stdout: str | None = None,
    ) -> str:
        """Start a managed process inside the agent's sandbox."""
        request: dict[str, Any] = {"type": "process.start", "argv": argv}
        if cwd is not None:
            request["cwd"] = cwd
        if stdout is not None:
            request["stdout"] = stdout
        return str(self._request(request)["process_id"])

    def process_status(self, process_id: str) -> int | None:
        """Return a managed process's exit status, or None while running."""
        reply = self._request({"type": "process.status",
                               "process_id": process_id})
        return reply.get("returncode")

    def signal_process(self, process_id: str, signum: int) -> None:
        """Send a signal to a managed process group."""
        self._request({"type": "process.signal", "process_id": process_id,
                       "signal": signum})

    def interrupt(self) -> None:
        self._send({"type": "interrupt"})

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                self._send({"type": "stop"})
            except (BrokenPipeError, OSError):
                pass
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.terminate()
                self.proc.wait(2)
        for stream in (self.proc.stdin, self.proc.stdout, self.proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        self._reader.join(2)
        self._errors.join(2)

    def status(self) -> str:
        return self.state


def _model_client(connection: socket.socket, cwd: Path,
                  model: str, effort: str | None, session_id: str | None = None,
                  resume: bool = False, tinker_client: Any = None,
                  trajectory_path: Path | None = None,
                  trajectory_dir: Path | None = None,
                  state_root: Path | None = None) -> None:
    """Give one agent connection the conversation requested in its handshake."""
    conversation = None
    lock = threading.Lock()
    active: threading.Thread | None = None
    tool_results: dict[str, queue.Queue[core.Result]] = {}

    def send(value: dict) -> None:
        data = (json.dumps(value, separators=(",", ":")) + "\n").encode()
        with lock:
            try:
                connection.sendall(data)
            except OSError:
                pass

    def complete(message: dict) -> None:
        def python(code: str) -> core.Result:
            call_id = uuid.uuid4().hex
            result: queue.Queue[core.Result] = queue.Queue(maxsize=1)
            tool_results[call_id] = result
            send({"tool_call": {"id": call_id, "code": code}})
            try:
                return result.get()
            finally:
                tool_results.pop(call_id, None)

        try:
            assert conversation is not None
            options = {"max_turns": max_turns}
            if feral:
                options["feral"] = True
            reply = conversation.complete(
                system, core.decode_message(message),
                lambda text: send({"delta": text}), python,
                **options)
            send({"message": core.encode_message(reply),
                  "context_used": conversation.context_used,
                  "limit_reached": getattr(
                      conversation, "limit_reached", False)})
        except InterruptedError:
            send({"error": "Interrupted", "interrupted": True})
        except Exception as error:
            send({"error": str(error)})

    try:
        with connection, connection.makefile("rb") as reader:
            try:
                initial = json.loads(reader.readline())
                if not isinstance(initial, dict):
                    return
                system = initial.get("system")
                if not isinstance(system, str):
                    return
                requested_model = initial.get("model", model)
                requested_effort = initial.get("effort", effort)
                requested_session = initial.get("session_id", session_id)
                requested_resume = initial.get("resume", resume)
                max_turns = initial.get("max_turns", 0)
                feral = initial.get("feral", False)
                if (not isinstance(requested_model, str)
                        or (requested_effort is not None
                            and not isinstance(requested_effort, str))
                        or (requested_session is not None
                            and not isinstance(requested_session, str))
                        or not isinstance(requested_resume, bool)
                        or not isinstance(feral, bool)
                        or not isinstance(max_turns, int)
                        or isinstance(max_turns, bool)
                        or max_turns < 0):
                    return
                if requested_model.startswith("claude-"):
                    conversation = Claude(
                        cwd, requested_model, requested_effort or "high",
                        requested_session, requested_resume)
                else:
                    options = {}
                    if tinker_client is not None:
                        options["client"] = tinker_client
                    if trajectory_path is not None:
                        options["trajectory_path"] = trajectory_path
                    elif trajectory_dir is not None:
                        if requested_session is None:
                            send({"error": (
                                "trajectory_dir requires a session ID")})
                            return
                        try:
                            trace_id = str(uuid.UUID(requested_session))
                        except ValueError:
                            send({"error": (
                                f"invalid session ID: {requested_session}")})
                            return
                        options["trajectory_path"] = (
                            trajectory_dir / f"{trace_id}.jsonl")
                    if state_root is not None:
                        options["state_root"] = state_root
                    conversation = Tinker(
                        cwd, requested_model, requested_effort,
                        requested_session, requested_resume, **options)
                while line := reader.readline():
                    request = json.loads(line)
                    if not isinstance(request, dict):
                        return
                    if request.get("interrupt") is True:
                        conversation.interrupt()
                    elif isinstance(request.get("tool_result"), dict):
                        raw = request["tool_result"]
                        waiting = tool_results.get(raw.get("id"))
                        if waiting is not None:
                            waiting.put(core.Result(
                                str(raw.get("output", "")),
                                int(raw.get("returncode", 1)),
                                float(raw.get("elapsed", 0)),
                                tuple(core.decode_encoded_input(item)
                                      for item in raw.get("inputs", [])),
                            ))
                    elif (isinstance(request.get("message"), dict) and
                          (active is None or not active.is_alive())):
                        active = threading.Thread(
                            target=complete, args=(request["message"],), daemon=True)
                        active.start()
                    else:
                        send({"error": "model generation already active"})
            except (json.JSONDecodeError, UnicodeDecodeError):
                return
    finally:
        if active is not None:
            active.join(1)
        if (conversation is not None and active is not None
                and active.is_alive()):
            conversation.interrupt()
            active.join(2)
        if conversation is not None:
            conversation.close()


class ModelServer:
    def __init__(self, path: Path, cwd: Path, model: str, effort: str,
                 session_id: str | None = None, resume: bool = False,
                 tinker_client: Any = None,
                 trajectory_path: Path | None = None,
                 trajectory_dir: Path | None = None,
                 state_root: Path | None = None) -> None:
        if trajectory_path is not None and trajectory_dir is not None:
            raise ValueError(
                "trajectory_path and trajectory_dir are mutually exclusive")
        self.path, self.cwd, self.model, self.effort = path, cwd, model, effort
        self.primary_session = (session_id, resume)
        self.tinker_client = tinker_client
        self.trajectory_path = trajectory_path
        self.trajectory_dir = trajectory_dir
        self.state_root = state_root
        self.session_lock = threading.Lock()
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(path))
        self.listener.listen()
        self.listener.settimeout(.2)
        self.stopping = threading.Event()
        self.clients: list[threading.Thread] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stopping.is_set():
            try:
                connection, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            with self.session_lock:
                session_id, resume = self.primary_session
                self.primary_session = (None, False)
            options = {}
            if self.tinker_client is not None:
                options["tinker_client"] = self.tinker_client
            if self.trajectory_path is not None:
                options["trajectory_path"] = self.trajectory_path
            if self.trajectory_dir is not None:
                options["trajectory_dir"] = self.trajectory_dir
            if self.state_root is not None:
                options["state_root"] = self.state_root
            thread = threading.Thread(
                target=_model_client,
                args=(connection, self.cwd, self.model, self.effort,
                      session_id, resume), kwargs=options, daemon=True)
            thread.start()
            self.clients.append(thread)

    def close(self) -> None:
        self.stopping.set()
        self.listener.close()
        self.thread.join(2)
        for thread in self.clients:
            thread.join(2)


class WorldRelay:
    """Expose one upstream Unix stream socket at an agent-local path."""

    def __init__(self, path: Path, upstream: Path) -> None:
        self.path = path.resolve()
        self.upstream = upstream.expanduser().resolve()
        if self.path == self.upstream:
            raise ValueError("world relay endpoints must be different")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.unlink(missing_ok=True)
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.path))
        self.path.chmod(0o600)
        self.listener.listen()
        self.listener.settimeout(.2)
        self.stopping = threading.Event()
        self.connections: set[socket.socket] = set()
        self.connection_lock = threading.Lock()
        self.clients: list[threading.Thread] = []
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stopping.is_set():
            try:
                downstream, _ = self.listener.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            thread = threading.Thread(
                target=self._bridge, args=(downstream,), daemon=True
            )
            self.clients.append(thread)
            thread.start()

    def _bridge(self, downstream: socket.socket) -> None:
        upstream = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sockets = (downstream, upstream)
        with self.connection_lock:
            self.connections.update(sockets)
        try:
            upstream.connect(str(self.upstream))
            threads = [
                threading.Thread(
                    target=self._pump, args=(source, destination), daemon=True
                )
                for source, destination in ((downstream, upstream), (upstream, downstream))
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        except OSError:
            pass
        finally:
            with self.connection_lock:
                self.connections.difference_update(sockets)
            for connection in sockets:
                connection.close()

    @staticmethod
    def _pump(source: socket.socket, destination: socket.socket) -> None:
        try:
            while data := source.recv(64 * 1024):
                destination.sendall(data)
        except OSError:
            pass
        finally:
            try:
                destination.shutdown(socket.SHUT_WR)
            except OSError:
                pass

    def close(self) -> None:
        self.stopping.set()
        self.listener.close()
        with self.connection_lock:
            connections = tuple(self.connections)
        for connection in connections:
            connection.close()
        self.thread.join(2)
        for thread in self.clients:
            thread.join(2)
        self.path.unlink(missing_ok=True)


def _mount_parents(path: Path) -> list[str]:
    parents = list(path.parents)[:-1]
    return [item for parent in reversed(parents)
            for item in ("--dir", str(parent))]


def _package_mounts(environment: Path, base: Path) -> tuple[list[str], list[str]]:
    """Map active site-packages that live outside the sandbox environment."""
    mounts: list[str] = []
    paths: list[str] = []
    seen: set[Path] = set()
    external = 0
    for value in sys.path:
        path = Path(value).resolve()
        if path in seen or path.name not in {"site-packages", "dist-packages"}:
            continue
        seen.add(path)
        try:
            relative = path.relative_to(environment)
        except ValueError:
            try:
                path.relative_to(base)
            except ValueError:
                target = Path("/opt/eko-packages") / str(external)
                external += 1
                mounts.extend(("--ro-bind", str(path), str(target)))
                paths.append(str(target))
            else:
                paths.append(str(path))
        else:
            paths.append(str(Path("/opt/eko") / relative))
    return mounts, paths


def _agent_command(cwd: Path, runtime: Path, *, sandbox: bool,
                   feral: bool, name: str, context: int = 0,
                   python_timeout: float = core.PYTHON_TIMEOUT,
                   max_turns: int = 0,
                   clean_workspace: bool = False, model: str | None = None,
                   effort: str | None = None, session_id: str | None = None,
                   resume: bool = False,
                   mounts: tuple[SandboxMount, ...] = ()) -> list[str]:
    source = Path(core.__file__).resolve()
    host_source = Path(__file__).resolve()
    models_source = Path(models.__file__).resolve()
    arguments = ["--cwd", "/workspace" if sandbox else str(cwd),
                 "--model-socket",
                 "/run/eko/model.sock" if sandbox else str(runtime / "model.sock"),
                 "--session-socket",
                 "/run/eko/session.sock" if sandbox else str(runtime / "session.sock"),
                 "--name", name, "--python-timeout", str(python_timeout)]
    if max_turns:
        arguments.extend(("--max-turns", str(max_turns)))
    if model is not None:
        arguments.extend(("--model", model))
    if effort is not None:
        arguments.extend(("--effort", effort))
    if session_id is not None:
        arguments.extend(("--resume" if resume else "--session-id", session_id))
    if context:
        arguments.extend(("--context", str(context)))
    if clean_workspace:
        arguments.append("--clean-workspace")
    if feral:
        arguments.append("--feral")
    if not sandbox:
        if mounts:
            raise ValueError("sandbox mounts require sandbox=True")
        return [sys.executable, str(source), *arguments]
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("--sandbox requires Bubblewrap (bwrap)")
    targets = [mount.target for mount in mounts]
    if len(targets) != len(set(targets)):
        raise ValueError("sandbox mount targets must be unique")
    for first in targets:
        if any(first != second and first in second.parents for second in targets):
            raise ValueError("sandbox mount targets must not overlap")
    mount_arguments = []
    for mount in mounts:
        relative = mount.target.relative_to("/workspace")
        placeholder = cwd / relative
        candidate = cwd
        for part in relative.parts:
            candidate /= part
            if candidate.is_symlink():
                raise ValueError("sandbox mount target cannot contain symbolic links")
        if mount.source.is_dir():
            placeholder.mkdir(parents=True, exist_ok=True)
        else:
            placeholder.parent.mkdir(parents=True, exist_ok=True)
            placeholder.touch(exist_ok=True)
        mount_arguments.extend(("--ro-bind" if mount.readonly else "--bind",
                                str(mount.source), str(mount.target)))
    environment = Path(sys.prefix).resolve()
    interpreter = Path(sys.executable).resolve()
    base = interpreter.parents[1]
    package_mounts, package_paths = _package_mounts(environment, base)
    return [
        bwrap, "--die-with-parent", "--new-session", "--as-pid-1",
        "--clearenv", "--unshare-user", "--unshare-pid", "--unshare-ipc",
        "--unshare-uts", "--unshare-cgroup", "--unshare-net",
        "--ro-bind", "/usr", "/usr",
        "--symlink", "usr/bin", "/bin", "--symlink", "usr/lib", "/lib",
        "--symlink", "usr/lib64", "/lib64", "--symlink", "usr/sbin", "/sbin",
        "--dir", "/etc", "--ro-bind", "/etc/alternatives", "/etc/alternatives",
        *_mount_parents(base), "--ro-bind", str(base), str(base),
        "--ro-bind", str(environment), "/opt/eko", "--dir", "/run",
        "--dir", "/opt/eko-packages", *package_mounts,
        "--ro-bind", str(source), "/run/eko.py",
        "--ro-bind", str(host_source), "/run/eko-host.py",
        "--ro-bind", str(models_source), "/run/models.py",
        "--bind", str(cwd), "/workspace", *mount_arguments,
        "--bind", str(runtime), "/run/eko",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--remount-ro", "/", "--setenv", "HOME", "/workspace",
        "--setenv", "TMPDIR", "/tmp", "--setenv", "LANG", "C.UTF-8",
        "--setenv", "PATH", "/opt/eko/bin:/usr/bin:/bin",
        "--setenv", "PYTHONPATH", os.pathsep.join(["/run", *package_paths]),
        "--setenv", "VIRTUAL_ENV", "/opt/eko",
        "--setenv", "EKO_MODEL", "/run/eko/model.sock",
        "--setenv", "EKO_HOST", "/run/eko-host.py",
        "--setenv", "EKO_WORLD", "/run/eko/world.sock",
        "--chdir", "/workspace", str(interpreter), "/run/eko.py",
        *arguments,
    ]


def print_event(event: core.Event) -> None:
    if event.type == "delta":
        print(event.value, end="", flush=True)
    elif event.type == "response":
        print(flush=True)
    elif event.type == "result":
        result = event.value
        if result.output:
            print(result.output, end="" if result.output.endswith("\n") else "\n",
                  flush=True)
    elif event.type == "error":
        print(f"! {event.value}", file=sys.stderr, flush=True)


class HeadlessObserver:
    """Print agent events and detect idle after an autonomous work cycle."""

    def __init__(self) -> None:
        self.active = False
        self.finished = threading.Event()

    def __call__(self, event: core.Event) -> None:
        print_event(event)
        if event.type != "state":
            return
        if event.value == "idle":
            if self.active:
                self.finished.set()
        else:
            self.active = True


def run(cwd: Path, prompt: str | None, *, model: str, effort: str,
        feral: bool, name: str, headless: bool, sandbox: bool,
        world_socket: Path | None = None, session_id: str | None = None,
        resume: bool = False, context: int = 0,
        clean_workspace: bool = False,
        python_timeout: float = core.PYTHON_TIMEOUT,
        max_turns: int = 0, exit_when_idle: bool = False,
        upstream_model_socket: Path | None = None,
        on_ready: Callable[[AgentProcess], None] | None = None,
        mounts: tuple[SandboxMount, ...] = ()) -> None:
    cwd = cwd.expanduser().resolve()
    if not cwd.is_dir():
        raise ValueError(f"not a directory: {cwd}")
    if upstream_model_socket is None:
        (ensure_claude_auth() if model.startswith("claude-")
         else ensure_tinker_auth())
    upstream_world = world_socket or os.environ.get("EKO_WORLD")
    with tempfile.TemporaryDirectory(prefix="eko-") as directory:
        runtime = Path(directory)
        server = (None if upstream_model_socket is not None else ModelServer(
            runtime / "model.sock", cwd, model, effort, session_id, resume))
        model_relay = (WorldRelay(runtime / "model.sock", upstream_model_socket)
                       if upstream_model_socket is not None else None)
        if server is not None:
            server.start()
        if model_relay is not None:
            model_relay.start()
        relay = (
            WorldRelay(runtime / "world.sock", Path(upstream_world))
            if upstream_world
            else None
        )
        if relay is not None:
            relay.start()
        environment = os.environ.copy()
        environment["EKO_MODEL"] = str(runtime / "model.sock")
        environment["EKO_WORLD"] = str(runtime / "world.sock")
        agent = AgentProcess(
            _agent_command(
                cwd, runtime, sandbox=sandbox, feral=feral, name=name,
                context=context, clean_workspace=clean_workspace,
                python_timeout=python_timeout, max_turns=max_turns,
                model=model, effort=effort, session_id=session_id, resume=resume,
                mounts=mounts,
            ),
            env=environment,
        )
        try:
            if not agent.ready.wait(10) or agent.proc.poll() is not None:
                raise RuntimeError(agent.startup_error())
            if on_ready is not None:
                on_ready(agent)
            if headless:
                observer = HeadlessObserver()
                agent.observer = observer
                print(f"EKO_SESSION={runtime / 'session.sock'}", flush=True)
                if prompt:
                    agent.send(prompt)
                while (agent.proc.poll() is None
                       and not (exit_when_idle and observer.finished.is_set())):
                    time.sleep(.2)
                return
            ui = UI(context=context)
            ui.header(cwd, model, name)
            ui.connect(agent)
            ui.on_exit = lambda: (agent.stop(), ui.exit())
            ui.on_start = lambda: agent.send(prompt) if prompt else None
            if prompt:
                ui.user(prompt)
            ui.run()
        except KeyboardInterrupt:
            signal.signal(signal.SIGINT, signal.SIG_IGN)
        finally:
            try:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            except KeyboardInterrupt:
                signal.signal(signal.SIGINT, signal.SIG_IGN)
            agent.stop()
            if relay is not None:
                relay.close()
            if model_relay is not None:
                model_relay.close()
            if server is not None:
                server.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?")
    parser.add_argument(
        "--cwd", type=Path,
        help="workspace (feral mode defaults to a fresh temporary directory)",
    )
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--effort", default="high")
    parser.add_argument("--name", default=core.NAME)
    parser.add_argument("--context", type=int, default=0)
    parser.add_argument(
        "--python-timeout", type=float, default=core.PYTHON_TIMEOUT,
        help=f"maximum seconds per Python action (default: {core.PYTHON_TIMEOUT:g})",
    )
    parser.add_argument(
        "--max-turns", type=int, default=0,
        help="stop a model response after this many model turns (0: unlimited)",
    )
    # Temporary experiment flag; remove after the workspace-hygiene evaluation.
    parser.add_argument("--clean-workspace", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--exit-when-idle", action="store_true",
        help="in headless mode, exit after the first completed work cycle",
    )
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument(
        "--mount", action="append", type=parse_mount, default=[],
        help="sandbox mount: type=bind,source=PATH,target=/workspace/PATH[,readonly]",
    )
    parser.add_argument(
        "--feral", action="store_true",
        help="start immediately and keep acting autonomously",
    )
    parser.add_argument(
        "--world-socket", type=Path,
        help="connect the agent to an external world socket",
    )
    parser.add_argument(
        "--upstream-model-socket", type=Path,
        help="relay model requests to an existing host-side model service",
    )
    session = parser.add_mutually_exclusive_group()
    session.add_argument(
        "--session-id", help="UUID to use for a new primary conversation"
    )
    session.add_argument(
        "--resume", metavar="UUID", help="resume the primary conversation UUID"
    )
    args = parser.parse_args()
    if args.python_timeout <= 0:
        parser.error("--python-timeout must be greater than zero")
    if args.max_turns < 0:
        parser.error("--max-turns must not be negative")
    if args.exit_when_idle and not args.headless:
        parser.error("--exit-when-idle requires --headless")
    if args.mount and not args.sandbox:
        parser.error("--mount requires --sandbox")

    def launch(cwd: Path) -> None:
        run(cwd, args.prompt, model=args.model, effort=args.effort,
            feral=args.feral, name=args.name, headless=args.headless,
            sandbox=args.sandbox, world_socket=args.world_socket,
            session_id=args.session_id or args.resume,
            resume=args.resume is not None, context=args.context,
            clean_workspace=args.clean_workspace,
            python_timeout=args.python_timeout, max_turns=args.max_turns,
            exit_when_idle=args.exit_when_idle,
            upstream_model_socket=args.upstream_model_socket,
            mounts=tuple(args.mount))

    try:
        if args.feral and args.cwd is None:
            with tempfile.TemporaryDirectory(prefix="eko-workspace-") as directory:
                launch(Path(directory))
        else:
            launch(args.cwd or Path.cwd())
    except (RuntimeError, ValueError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
