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

The whole agent is one loop:

    while prompt := user():
        message = prompt
        while not (response := llm(message)).is_done():
            message = run(response.python(), cwd)

Python is the model's only interface. Through it, the model can inspect the folder,
use the shell, and write its own tools. Everything it writes there persists.

The loop only needs access to an LLM. That could come from an API; this version uses
the Claude Code CLI in print mode, which can access the model through a subscription,
with its tools and system prompt disabled.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import queue
import re
import select
import shutil
import signal
import subprocess
import sys
import threading
import time
import tokenize
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterator, Protocol

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
from rich.text import Text


SYSTEM = """You are Eko, working in {folder}.

Write a fenced ```python block to act. After your response ends, it runs in that
folder and its output arrives in a later user message inside <python_result> tags.
Never write or predict those tags yourself. If no action is needed, answer
directly. When the prompt is fully resolved, end with <done/> and no Python block.
"""

NUDGE = "Write a fenced ```python block, or <done/> if the prompt is resolved."
MAX_OUTPUT = 20_000
TIMEOUT = 600


# ── Core agent ────────────────────────────────────────────────────────────────

@dataclass
class Result:
    output: str
    returncode: int
    elapsed: float


OPEN_FENCE = re.compile(r"^[ \t]{0,3}(`{3,})[ \t]*python[ \t]*$")


def opening_fence(line: str) -> int:
    """Return the backtick count for a Python fence on this complete line."""
    match = OPEN_FENCE.fullmatch(line.rstrip("\r\n"))
    return len(match.group(1)) if match else 0


def closing_fence(line: str, length: int) -> bool:
    """Whether this complete line closes a fence of ``length`` backticks."""
    return bool(re.fullmatch(
        rf"[ \t]{{0,3}}`{{{length},}}[ \t]*", line.rstrip("\r\n")))


def response_segments(text: str) -> list[tuple[str, str, bool]]:
    """Split prose and Python fences using Markdown's line-based fence rules."""
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


def extract_python(text: str) -> str | None:
    blocks = [content for kind, content, closed in response_segments(text)
              if kind == "python" and closed]
    return "\n".join(blocks) if blocks else None


def clipped(text: str, limit: int = MAX_OUTPUT) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    omitted = len(text) - limit
    return f"{text[:half]}\n\n… {omitted:,} characters omitted …\n\n{text[-half:]}"


def run_python(code: str, cwd: Path, interrupted: threading.Event) -> Result:
    """Run one model-written Python block in the persistent working folder."""
    python = cwd / ".venv/bin/python"
    executable = str(python if python.exists() else Path(sys.executable))
    started = time.monotonic()
    proc = subprocess.Popen(
        [executable, "-u", "-c", code], cwd=cwd, stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        errors="replace", start_new_session=True)
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


class AgentIO(Protocol):
    """The few operations the agent loop needs from its surrounding app."""

    interrupted: threading.Event

    def ask(self, message: str) -> str: ...
    def show_response(self, response: str, code: str | None) -> None: ...
    def execute(self, code: str) -> Result: ...
    def show_result(self, result: Result) -> None: ...
    def take_input(self) -> str: ...


def agent(prompt: str, io: AgentIO) -> None:
    """Run the complete model → Python → result loop for one user prompt."""
    message = prompt
    while True:
        response = io.ask(message)
        code = extract_python(response)
        io.show_response(response, code)

        if code is None and "<done/>" in response:
            return
        if code is None:
            message = NUDGE
        else:
            result = io.execute(code)
            io.show_result(result)
            if io.interrupted.is_set():
                return
            output = clipped(result.output) or "(no output)"
            message = f"<python_result>\n{output}\n</python_result>"

        if user_input := io.take_input():
            message += f"\n\n{user_input}"


# ── Terminal rendering ────────────────────────────────────────────────────────

MAX_DISPLAY_OUTPUT = 4_000
MAX_DISPLAY_LINES = 5
GOLD = "#d7af5f"
GOLD_ACTIVE = "#e5bd68"


def visible_response(text: str) -> str:
    """Hide transport-only Python results if the model repeats them."""
    text = re.sub(
        r"<python_result>.*?</python_result>", "", text, flags=re.DOTALL)
    start = text.find("<python_result>")
    if start >= 0:
        text = text[:start]
    else:
        tag = "<python_result>"
        for length in range(1, len(tag)):
            if text.endswith(tag[:length]):
                text = text[:-length]
                break
    return text


def response_renderable(text: str):
    """Render complete or partially streamed Python fences as code panels."""
    text = visible_response(text).replace("<done/>", "")
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
    return clipped("\n".join(lines), MAX_DISPLAY_OUTPUT)


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
        self.in_python_result = False

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
            if closing_fence(line, self.fence_length):
                self._commit_code(final=True)
                self._close_box()
                self.code = False
                self.fence_length = 0
            else:
                self.code_lines.append(line.rstrip("\r\n"))
                self._commit_code()
            return
        if length := opening_fence(line):
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
        """Hide model-predicted result transport, even across stream chunks."""
        visible = []
        while text:
            if self.in_python_result:
                end = text.find("</python_result>")
                if end < 0:
                    break
                text = text[end + len("</python_result>"):]
                self.in_python_result = False
                continue
            start = text.find("<python_result>")
            if start < 0:
                tag = "<python_result>"
                partial = next((length for length in range(len(tag) - 1, 0, -1)
                                if text.endswith(tag[:length])), 0)
                visible.append(text[:-partial] if partial else text)
                break
            visible.append(text[:start])
            text = text[start + len("<python_result>"):]
            self.in_python_result = True
        return "".join(visible)

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
        self._queue(self.ui._render(Text(
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

    def _code_line(self, line: Text) -> None:
        width = self._width()
        content = line[:max(1, width - 4)]
        padding = " " * max(0, width - len(content.plain) - 4)
        row = Text()
        row.append("│ ", style=GOLD)
        row.append_text(content)
        row.append(padding)
        row.append(" │", style=GOLD)
        self._queue(self.ui._render(row))

    def _close_box(self) -> None:
        if self.box_open:
            width = self._width()
            self._queue(self.ui._render(Text(
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

    def header(self, cwd: Path, model: str) -> None:
        header = Group(Text("eko", style=f"bold {GOLD}"), Text.assemble(
            (str(cwd), "dim"), ("  ·  ", "dim"), (model, "dim"),
            ("  ·  signed in", "dim")), Text(""))
        # Place the initial composer near the bottom. These are real terminal
        # lines, so subsequent turns naturally replace them and enter native
        # scrollback instead of making a full-screen spacer jump around.
        padding = max(
            0, shutil.get_terminal_size().lines - 9)
        self._append(self._render(header) + "\n" * padding)

    def user(self, text: str) -> None:
        rendered = self._render(Text.assemble(("› ", f"bold {GOLD}"), text))
        self._append(rendered + "\n")

    @contextmanager
    def streaming(self) -> Iterator[Callable[[str], None]]:
        """Write stable response lines above the fixed composer as they arrive."""
        stream = NativeStream(self)
        stopped = threading.Event()
        frames = ("", ".", "..", "...", "..", ".")

        def animate() -> None:
            frame = 0
            while not stopped.is_set():
                rendered = self._render(Text(
                    f"thinking{frames[frame % len(frames)]}",
                    style=f"dim {GOLD_ACTIVE}"))
                with self.lock:
                    if stopped.is_set():
                        break
                    self.live = rendered
                self.app.invalidate()
                frame += 1
                stopped.wait(.12)

        animator = threading.Thread(target=animate, daemon=True)
        animator.start()

        def write(delta: str) -> None:
            stream.feed(delta)

        try:
            yield write
        finally:
            stopped.set()
            animator.join(timeout=.2)
            stream.finish()
            self.streamed_response = stream.text
            with self.lock:
                self.live = ""
            self.app.invalidate()

    @contextmanager
    def activity(self, label: str) -> Iterator[None]:
        """Animate a short-lived activity in the live area."""
        stopped = threading.Event()
        frames = ("", ".", "..", "...", "..", ".")

        def animate() -> None:
            frame = 0
            while not stopped.is_set():
                rendered = self._render(Text(
                    f"{label}{frames[frame % len(frames)]}",
                    style=f"dim {GOLD_ACTIVE}"))
                with self.lock:
                    self.live = rendered
                self.app.invalidate()
                frame += 1
                stopped.wait(.12)

        animator = threading.Thread(target=animate, daemon=True)
        animator.start()
        try:
            yield
        finally:
            stopped.set()
            animator.join(timeout=.2)
            with self.lock:
                self.live = ""
            self.app.invalidate()

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
        body = Text(output.rstrip() or "(no output)", style="dim")
        self._append(self._render(Group(Text(label, style=style), body)) + "\n")

    def stopped(self, message: str = "Stopped") -> None:
        rendered = self._render(Text(f"! {message}", style="#ff875f")) + "\n"
        self._append(rendered)

    def run(self) -> None:
        self.app.pre_run_callables.append(self.on_start)
        self.app.run()

    def exit(self) -> None:
        if self.app.is_running:
            self.app.exit()


# ── Claude model connection ───────────────────────────────────────────────────

CALL_TIMEOUT = 300


class ClaudeModel:
    """A persistent, tool-free connection to the LLM through the Claude CLI.

    Stream JSON lets several Eko turns share one model conversation. ``--safe-mode``
    prevents machine-specific instructions, hooks, plugins, and skills from changing
    the model's context, while ``--tools ''`` leaves generated Python as its only action.
    """

    def __init__(self, cwd: Path, model: str, effort: str) -> None:
        self.cwd = cwd
        self.model = model
        self.effort = effort
        self.session_id = str(uuid.uuid4())
        self.proc: subprocess.Popen[bytes] | None = None
        self.started = False
        self.interrupted = threading.Event()

    def _start(self) -> None:
        session = (["--resume", self.session_id] if self.started else
                   ["--session-id", self.session_id])
        command = [
            "claude", "-p", "--verbose", "--safe-mode", "--tools", "",
            "--model", self.model, "--effort", self.effort,
            *session,
            "--input-format", "stream-json", "--output-format", "stream-json",
            "--include-partial-messages",
            "--system-prompt", SYSTEM.format(folder=self.cwd),
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

    def ask(self, message: str, on_text: Callable[[str], None],
            deadline: float | None = None, retry_delay: float = .2) -> str:
        """Send one message, forwarding text deltas while collecting the response."""
        self.interrupted.clear()
        deadline = deadline or time.monotonic() + CALL_TIMEOUT
        resuming = self.started
        if self.proc is None or self.proc.poll() is not None:
            self._start()
        proc = self.proc
        assert proc is not None and proc.stdin and proc.stdout
        event = {"type": "user", "message": {"role": "user", "content": [
            {"type": "text", "text": message}]}}
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
                        remaining = deadline - time.monotonic()
                        if remaining > 0 and not parts and not complete:
                            delay = min(retry_delay, remaining)
                            if self.interrupted.wait(delay):
                                raise InterruptedError
                            if time.monotonic() < deadline:
                                return self.ask(
                                    message, on_text, deadline,
                                    min(retry_delay * 2, 5))
                        raise RuntimeError(
                            "Model session could not resume; context was not "
                            f"reset. {detail or ''}".rstrip())
                    raise RuntimeError(detail or "Model call failed")
                return complete or "".join(parts)
        raise RuntimeError("Model produced no result")

    def close(self) -> None:
        """Give the CLI a brief chance to flush its session, then stop it."""
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                assert self.proc.stdin
                self.proc.stdin.close()
                self.proc.stdin = None
                self.proc.wait(timeout=3)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self._terminate(signal.SIGTERM)
                return
        if self.proc.stdout:
            self.proc.stdout.close()
        self.proc = None

    def interrupt(self) -> None:
        self.interrupted.set()
        self._terminate(signal.SIGKILL)


# ── Interactive session and CLI ───────────────────────────────────────────────

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


class Session:
    """Run the model loop while keeping the composer available for new input."""

    def __init__(self, cwd: Path, model: str, effort: str, ui: UI) -> None:
        self.cwd = cwd
        self.ui = ui
        self.llm = ClaudeModel(cwd, model, effort)
        self.followups: queue.Queue[str | None] = queue.Queue()
        self.inputs: queue.Queue[str] = queue.Queue()
        self.stopping = threading.Event()
        self.interrupted = threading.Event()
        self.state = "idle"
        self.thread = threading.Thread(
            target=run_session, args=(self,), daemon=True)

    def start(self, prompt: str | None = None) -> None:
        if prompt:
            self.followups.put(prompt)
        self.thread.start()

    def submit(self, message: str) -> None:
        if self.state == "idle":
            self.followups.put(message)
        else:
            self.inputs.put(message)

    def interrupt(self) -> None:
        if self.state == "idle":
            return
        self.interrupted.set()
        if self.state == "thinking":
            self.llm.interrupt()

    def stop(self) -> None:
        self.stopping.set()
        self.interrupted.set()
        self.followups.put(None)
        self.llm.interrupt()

    def status(self) -> str:
        pending = len(self.pending())
        return self.state + (f" · {pending} pending" if pending else "")

    def pending(self) -> list[str]:
        with self.followups.mutex, self.inputs.mutex:
            return ([message for message in self.inputs.queue]
                    + [message for message in self.followups.queue
                       if message is not None])

    def take_input(self) -> str:
        """Collect user messages submitted while the current task was running."""
        pending: list[str] = []
        while True:
            try:
                message = self.inputs.get_nowait()
            except queue.Empty:
                break
            pending.append(message)
        return "\n".join(pending)

    def next_prompt(self) -> str | None:
        self.state = "idle"
        self.interrupted.clear()
        self.ui.app.invalidate()
        if user_input := self.take_input():
            return user_input
        return self.followups.get()

    # AgentIO: adapt the core loop to the model connection and terminal UI.
    def ask(self, message: str) -> str:
        self.state = "thinking"
        with self.ui.streaming() as write:
            return self.llm.ask(message, write)

    def show_response(self, response: str, code: str | None) -> None:
        self.ui.assistant(response, code)

    def execute(self, code: str) -> Result:
        self.state = "running Python"
        with self.ui.activity("running"):
            return run_python(code, self.cwd, self.interrupted)

    def show_result(self, result: Result) -> None:
        self.ui.result(result)


def run_session(session: Session) -> None:
    """Feed queued prompts to the core agent from a background thread."""
    message = session.next_prompt()
    try:
        while message is not None:
            try:
                agent(message, session)
            except InterruptedError:
                session.ui.stopped("Interrupted")
            except Exception as error:
                if session.stopping.is_set():
                    break
                session.ui.stopped(str(error))
            message = session.next_prompt()
    finally:
        session.llm.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("prompt", nargs="?", help="prompt to run; opens an input if omitted")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="working directory")
    parser.add_argument("--model", default="claude-opus-5")
    parser.add_argument("--effort", default="high")
    args = parser.parse_args()

    cwd = args.cwd.expanduser().resolve()
    if not cwd.is_dir():
        parser.error(f"not a directory: {cwd}")
    ensure_auth()
    ui = UI()
    ui.header(cwd, args.model)
    if args.prompt:
        ui.user(args.prompt)
    session = Session(cwd, args.model, args.effort, ui)
    ui.status = session.status
    ui.pending = session.pending
    ui.on_submit = session.submit
    ui.on_interrupt = session.interrupt

    def exit_app() -> None:
        session.stop()
        ui.exit()

    ui.on_exit = exit_app
    ui.on_start = lambda: session.start(args.prompt)
    try:
        ui.run()
    except KeyboardInterrupt:
        session.stop()
    finally:
        session.stop()
    session.thread.join(timeout=5)


if __name__ == "__main__":
    main()
