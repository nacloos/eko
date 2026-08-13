# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "prompt-toolkit>=3.0,<4",
#     "rich>=13,<15",
# ]
# ///

"""Host Eko's model connection, sandbox, and terminal presentation."""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import select
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tokenize
import uuid
from pathlib import Path
from typing import Callable

import eko as core
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
                 effort: str = "high") -> None:
        self.cwd = cwd
        self.model = model
        self.effort = effort
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
    for kind, content, _closed in core.response_segments(text):
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
    text = "\n".join(lines)
    if len(text) <= MAX_DISPLAY_OUTPUT:
        return text
    half = MAX_DISPLAY_OUTPUT // 2
    omitted = len(text) - MAX_DISPLAY_OUTPUT
    return (f"{text[:half]}\n\n… {omitted:,} characters omitted …\n\n"
            f"{text[-half:]}")


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
            if core.closing_fence(line, self.fence_length):
                self._commit_code(final=True)
                self._close_box()
                self.code = False
                self.fence_length = 0
            else:
                self.code_lines.append(line.rstrip("\r\n"))
                self._commit_code()
            return
        if length := core.opening_fence(line):
            self._prose("", final=True)
            self.code = True
            self.fence_length = length
            self._open_box()
        else:
            self._prose(line)

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
                response, _ = event.value
                self.streamed_response = response
                self.assistant(response)
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
        self._reader = threading.Thread(target=self._read, daemon=True)
        self._errors = threading.Thread(target=self._read_errors, daemon=True)
        self._reader.start()
        self._errors.start()

    def _read(self) -> None:
        assert self.proc.stdout
        try:
            for line in self.proc.stdout:
                raw = json.loads(line)
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
            self.observer(core.Event(
                "error", f"Agent connection failed: {error}"))
        finally:
            self.ready.set()

    def _read_errors(self) -> None:
        assert self.proc.stderr
        for line in self.proc.stderr:
            self.observer(core.Event("error", line.rstrip()))

    def _send(self, value: dict) -> None:
        assert self.proc.stdin
        with self._write_lock:
            self.proc.stdin.write(json.dumps(value, separators=(",", ":")) + "\n")
            self.proc.stdin.flush()

    def send(self, text: str) -> None:
        self._send({"type": "input", "content": [{"type": "text", "text": text}]})

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
                  model: str, effort: str) -> None:
    """Give one agent connection an independent Claude conversation."""
    claude = Claude(cwd, model, effort)
    lock = threading.Lock()
    active: threading.Thread | None = None

    def send(value: dict) -> None:
        data = (json.dumps(value, separators=(",", ":")) + "\n").encode()
        with lock:
            try:
                connection.sendall(data)
            except OSError:
                pass

    def complete(message: dict) -> None:
        try:
            reply = claude.complete(
                system, core.decode_message(message),
                lambda text: send({"delta": text}))
            send({"message": core.encode_message(reply)})
        except InterruptedError:
            send({"error": "Interrupted", "interrupted": True})
        except Exception as error:
            send({"error": str(error)})

    try:
        with connection, connection.makefile("rb") as reader:
            initial = json.loads(reader.readline())
            system = initial.get("system")
            if not isinstance(system, str):
                raise ValueError("first model event must contain system")
            while line := reader.readline():
                request = json.loads(line)
                if request.get("interrupt") is True:
                    claude.interrupt()
                elif (isinstance(request.get("message"), dict) and
                      (active is None or not active.is_alive())):
                    active = threading.Thread(
                        target=complete, args=(request["message"],), daemon=True)
                    active.start()
                else:
                    send({"error": "model generation already active"})
    finally:
        claude.interrupt()
        if active is not None:
            active.join(2)
        claude.close()


class ModelServer:
    def __init__(self, path: Path, cwd: Path, model: str, effort: str) -> None:
        self.path, self.cwd, self.model, self.effort = path, cwd, model, effort
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
            thread = threading.Thread(
                target=_model_client,
                args=(connection, self.cwd, self.model, self.effort), daemon=True)
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


def _agent_command(cwd: Path, runtime: Path, *, sandbox: bool,
                   feral: bool, name: str) -> list[str]:
    source = Path(core.__file__).resolve()
    arguments = ["--cwd", "/workspace" if sandbox else str(cwd),
                 "--model-socket",
                 "/run/eko/model.sock" if sandbox else str(runtime / "model.sock"),
                 "--session-socket",
                 "/run/eko/session.sock" if sandbox else str(runtime / "session.sock"),
                 "--name", name]
    if feral:
        arguments.append("--feral")
    if not sandbox:
        return [sys.executable, str(source), *arguments]
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        raise RuntimeError("--sandbox requires Bubblewrap (bwrap)")
    environment = Path(sys.prefix).resolve()
    base = Path(sys.executable).resolve().parents[1]
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
        "--ro-bind", str(source), "/run/eko.py",
        "--bind", str(cwd), "/workspace", "--bind", str(runtime), "/run/eko",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
        "--remount-ro", "/", "--setenv", "HOME", "/workspace",
        "--setenv", "TMPDIR", "/tmp", "--setenv", "LANG", "C.UTF-8",
        "--setenv", "PATH", "/opt/eko/bin:/usr/bin:/bin",
        "--setenv", "EKO_MODEL", "/run/eko/model.sock",
        "--setenv", "EKO_WORLD", "/run/eko/world.sock",
        "--chdir", "/workspace", "/opt/eko/bin/python", "/run/eko.py",
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


def run(cwd: Path, prompt: str | None, *, model: str, effort: str,
        feral: bool, name: str, headless: bool, sandbox: bool,
        world_socket: Path | None = None) -> None:
    cwd = cwd.expanduser().resolve()
    if not cwd.is_dir():
        raise ValueError(f"not a directory: {cwd}")
    ensure_auth()
    upstream_world = world_socket or os.environ.get("EKO_WORLD")
    with tempfile.TemporaryDirectory(prefix="eko-") as directory:
        runtime = Path(directory)
        server = ModelServer(runtime / "model.sock", cwd, model, effort)
        server.start()
        relay = (
            WorldRelay(runtime / "world.sock", Path(upstream_world))
            if upstream_world
            else None
        )
        if relay is not None:
            relay.start()
        environment = os.environ.copy()
        environment["EKO_WORLD"] = str(runtime / "world.sock")
        agent = AgentProcess(
            _agent_command(
                cwd, runtime, sandbox=sandbox, feral=feral, name=name
            ),
            env=environment,
        )
        try:
            if not agent.ready.wait(10) or agent.proc.poll() is not None:
                raise RuntimeError("agent did not start")
            if headless:
                agent.observer = print_event
                print(f"EKO_SESSION={runtime / 'session.sock'}", flush=True)
                if prompt:
                    agent.send(prompt)
                while agent.proc.poll() is None:
                    time.sleep(.2)
                return
            ui = UI()
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
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--sandbox", action="store_true")
    parser.add_argument(
        "--feral", action="store_true",
        help="start immediately and keep acting autonomously",
    )
    parser.add_argument(
        "--world-socket", type=Path,
        help="connect the agent to an external world socket",
    )
    args = parser.parse_args()

    def launch(cwd: Path) -> None:
        run(cwd, args.prompt, model=args.model, effort=args.effort,
            feral=args.feral, name=args.name, headless=args.headless,
            sandbox=args.sandbox, world_socket=args.world_socket)

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
