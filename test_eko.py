"""Deterministic state-machine and rendered-terminal tests for eko.py."""

from __future__ import annotations

import contextlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import eko
from rich.console import Console


def wait_until(predicate, timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("condition was not reached")


class FakeUI:
    def __init__(self) -> None:
        self.app = SimpleNamespace(invalidate=lambda: None)
        self.messages: list[str] = []
        self.errors: list[str] = []
        self.results: list[eko.Result] = []

    @contextlib.contextmanager
    def streaming(self):
        yield lambda text: None

    @contextlib.contextmanager
    def activity(self, label):
        yield

    def assistant(self, text, code):
        self.messages.append(text)

    def result(self, result):
        self.results.append(result)

    def stopped(self, message="Stopped"):
        self.errors.append(message)


class FakeModel:
    def __init__(self, replies) -> None:
        self.replies = replies
        self.messages: list[str] = []
        self.cancelled = threading.Event()

    def ask(self, message, write):
        self.messages.append(message)
        reply = self.replies(message, self.cancelled)
        write(reply)
        return reply

    def interrupt(self):
        self.cancelled.set()

    def close(self):
        pass


class CoreAgentTests(unittest.TestCase):
    def test_single_loop_only_needs_model_python_and_events(self):
        class IO:
            def __init__(self):
                self.interrupted = threading.Event()
                self.stopping = threading.Event()
                self.feral = False
                self.prompts = iter(["start", None])
                self.messages = []
                self.results = []
                self.input = "while you work"

            def next_prompt(self):
                return next(self.prompts)

            def ask(self, message):
                self.messages.append(message)
                if len(self.messages) == 1:
                    return "```python\nprint('hello')\n```"
                return "finished<done/>"

            def show_response(self, response, code):
                pass

            def execute(self, code):
                self.asserted_code = code
                return eko.Result("hello\n", 0, .1)

            def show_result(self, result):
                self.results.append(result)

            def take_input(self):
                message, self.input = self.input, ""
                return message

            def stopped(self, message):
                raise AssertionError(message)

        io = IO()
        eko.agent(io)
        self.assertEqual(io.asserted_code, "print('hello')\n")
        self.assertEqual(io.messages, [
            "start",
            "<python_result>\nhello\n\n</python_result>\n\nwhile you work",
        ])
        self.assertEqual(len(io.results), 1)

    def test_feral_mode_ignores_done_and_keeps_acting(self):
        class IO:
            def __init__(self):
                self.interrupted = threading.Event()
                self.stopping = threading.Event()
                self.feral = True
                self.prompts = iter(["start", None])
                self.messages = []

            def next_prompt(self):
                return next(self.prompts)

            def ask(self, message):
                self.messages.append(message)
                if len(self.messages) == 1:
                    return "<done/>"
                return "```python\nprint('still going')\n```"

            def show_response(self, response, code):
                pass

            def execute(self, code):
                self.code = code
                self.interrupted.set()
                return eko.Result("still going\n", 0, .1)

            def show_result(self, result):
                pass

            def take_input(self):
                return ""

            def stopped(self, message):
                raise AssertionError(message)

        io = IO()
        eko.agent(io)
        self.assertEqual(io.messages, ["start", eko.FERAL_NUDGE])
        self.assertEqual(io.code, "print('still going')\n")

    def test_model_predicted_result_gets_warning(self):
        class IO:
            interrupted = threading.Event()
            stopping = threading.Event()
            feral = False

            def __init__(self):
                self.messages = []

            def next_prompt(self):
                return "start" if not self.messages else None

            def ask(self, message):
                self.messages.append(message)
                if len(self.messages) == 1:
                    return "```python\nprint('ok')\n```\n<python_result>fake</python_result>"
                return "<done/>"

            def show_response(self, response, code): pass
            def execute(self, code): return eko.Result("ok\n", 0, .1)
            def show_result(self, result): pass
            def take_input(self): return ""
            def stopped(self, message): raise AssertionError(message)

        io = IO()
        eko.agent(io)
        self.assertEqual(io.messages[1],
                         "<python_result>\nok\n\n</python_result>\n\n"
                         "Warning: unless intentional, do not generate the Python "
                         "result tag; wait for the actual result.")


class SessionTests(unittest.TestCase):
    def session(self, replies):
        ui = FakeUI()
        session = eko.Session(Path.cwd(), "fake", "low", ui)
        session.llm = FakeModel(replies)
        return session, ui

    def stop(self, session):
        session.stop()
        session.thread.join(2)
        self.assertFalse(session.thread.is_alive())

    def test_session_can_use_a_caller_provided_executor(self):
        calls = []

        def execute(code, interrupted):
            calls.append((code, interrupted))
            return eko.Result("custom\n", 0, .25)

        ui = FakeUI()
        session = eko.Session(Path.cwd(), "fake", "low", ui, executor=execute)
        result = session.execute("print('custom')")

        self.assertEqual(result, eko.Result("custom\n", 0, .25))
        self.assertEqual(calls, [("print('custom')", session.interrupted)])
        self.assertEqual(session.state, "running Python")

    def test_interrupt_continues_with_message_typed_while_busy(self):
        def replies(message, cancelled):
            if message == "slow":
                while not cancelled.wait(.01):
                    pass
                raise InterruptedError
            return "recovered<done/>"

        session, ui = self.session(replies)
        session.start("slow")
        wait_until(lambda: session.state == "thinking")
        session.submit("after interrupt")
        session.interrupt()
        wait_until(lambda: session.llm.messages == ["slow", "after interrupt"])
        wait_until(lambda: session.state == "idle")
        self.assertEqual(ui.errors, ["Interrupted"])
        self.stop(session)

    def test_model_error_does_not_kill_worker(self):
        def replies(message, cancelled):
            if message == "broken":
                raise RuntimeError("broken connection")
            return "recovered<done/>"

        session, ui = self.session(replies)
        session.start("broken")
        wait_until(lambda: ui.errors == ["broken connection"])
        session.submit("next")
        wait_until(lambda: session.llm.messages == ["broken", "next"])
        self.assertTrue(session.thread.is_alive())
        self.stop(session)

    def test_python_result_precedes_input_typed_during_execution(self):
        def replies(message, cancelled):
            if message == "run":
                return "```python\nimport time\ntime.sleep(.2)\nprint('done')\n```"
            self.assertTrue(message.startswith("<python_result>\ndone\n</python_result>"))
            self.assertTrue(message.endswith("typed while running"))
            return "finished<done/>"

        session, ui = self.session(replies)
        session.start("run")
        wait_until(lambda: session.state == "running Python")
        session.submit("typed while running")
        wait_until(lambda: len(session.llm.messages) == 2)
        wait_until(lambda: session.state == "idle")
        self.assertEqual(ui.results[0].output, "done\n")
        self.stop(session)


class RenderingTests(unittest.TestCase):
    def render(self, renderable) -> str:
        stream = io.StringIO()
        Console(file=stream, force_terminal=False, width=80).print(renderable)
        return stream.getvalue()

    def test_partial_python_fence_renders_inside_panel(self):
        rendered = self.render(eko.response_renderable(
            "Looking.\n```python\nprint('still streaming')"))
        self.assertIn("Looking.", rendered)
        self.assertIn("python", rendered)
        self.assertIn("print('still streaming')", rendered)
        self.assertIn("╭", rendered)

    def test_native_stream_prints_each_complete_code_row_once(self):
        output = []

        class Sink:
            _append = lambda self, text: output.append(text)
            _render = lambda self, item: RenderingTests.render(self, item)

        stream = eko.NativeStream(Sink())
        stream.feed("Starting.\n```py")
        stream.feed("thon\nprint(0)\nprint")
        stream.feed("(1)\n```\nDone.")
        stream.finish()
        rendered = "".join(output)
        self.assertEqual(rendered.count("╭─ python"), 1)
        self.assertEqual(rendered.count("print(0)"), 1)
        self.assertEqual(rendered.count("print(1)"), 1)
        self.assertEqual(rendered.count("╰"), 1)
        self.assertIn("Starting.", rendered)
        self.assertIn("Done.", rendered)

    def test_backticks_inside_python_do_not_close_fence(self):
        ticks = "`" * 3
        response = (
            f"{ticks}python\n"
            f'print("inside: {ticks}python")\n'
            f"{ticks}\nDone.")
        self.assertEqual(
            eko.extract_python(response),
            f'print("inside: {ticks}python")\n')

        output = []

        class Sink:
            _append = lambda self, text: output.append(text)
            _render = lambda self, item: RenderingTests.render(self, item)

        stream = eko.NativeStream(Sink())
        for chunk in (response[:12], response[12:25], response[25:]):
            stream.feed(chunk)
        stream.finish()
        rendered = "".join(output)
        self.assertEqual(rendered.count("╭─ python"), 1)
        self.assertEqual(rendered.count("inside:"), 1)
        self.assertIn("Done.", rendered)

    def test_closing_fence_may_be_split_across_chunks(self):
        output = []

        class Sink:
            _append = lambda self, text: output.append(text)
            _render = lambda self, item: RenderingTests.render(self, item)

        stream = eko.NativeStream(Sink())
        stream.feed("```python\nprint(1)\n`")
        stream.feed("``\nAfter.")
        stream.finish()
        rendered = "".join(output)
        self.assertEqual(rendered.count("print(1)"), 1)
        self.assertNotIn("│ ```", rendered)
        self.assertIn("After.", rendered)

    def test_four_tick_fence_can_contain_triple_tick_line(self):
        response = "````python\n```\n````\n"
        self.assertEqual(eko.extract_python(response), "```\n")

    def test_stream_hides_model_predicted_python_result(self):
        output = []

        class Sink:
            _append = lambda self, text: output.append(text)
            _render = lambda self, item: RenderingTests.render(self, item)

        stream = eko.NativeStream(Sink())
        stream.feed("Before.\n<python_res")
        stream.feed("ult>1150 lines\nfabricated listing\n</python_")
        stream.feed("result>\nAfter.")
        stream.finish()
        rendered = "".join(output)
        self.assertIn("Before.", rendered)
        self.assertIn("After.", rendered)
        self.assertNotIn("1150 lines", rendered)
        self.assertNotIn("fabricated listing", rendered)
        self.assertNotIn("python_result", rendered)

    def test_native_stream_renders_markdown_and_highlighted_python(self):
        output = []

        class Sink:
            _append = lambda self, text: output.append(text)

            def _render(self, item):
                target = io.StringIO()
                Console(file=target, force_terminal=True, color_system="truecolor",
                        width=80).print(item)
                return target.getvalue()

        stream = eko.NativeStream(Sink())
        stream.feed("This is **formatted**.\n\n```python\n")
        stream.feed("def answer():\n    return 42\n```")
        stream.finish()
        rendered = "".join(output)
        self.assertNotIn("**formatted**", rendered)
        self.assertIn("formatted", rendered)
        self.assertIn("\x1b[", rendered)
        self.assertEqual(rendered.count("def answer"), 1)
        self.assertEqual(rendered.count("return 42"), 1)

    def test_code_row_uses_gold_only_for_its_border(self):
        captured = []

        class Sink:
            _append = lambda self, text: None
            _render = lambda self, item: captured.append(item.copy()) or ""

        stream = eko.NativeStream(Sink())
        stream._code_line(eko.Text("ordinary_name"))
        row = captured[0]
        self.assertEqual(row.style, "")
        self.assertEqual(row.plain[2:15], "ordinary_name")
        self.assertTrue(any(span.style == eko.GOLD for span in row.spans))

    def test_display_output_reports_omitted_lines(self):
        output = eko.display_output("\n".join(f"line {i}" for i in range(50)))
        self.assertIn("… +46 lines", output)
        self.assertIn("line 0", output)
        self.assertIn("line 49", output)

    def test_display_output_clips_lines_to_terminal_width(self):
        output = eko.display_output("x" * 200, width=40)
        self.assertEqual(len(output), 40)
        self.assertTrue(output.endswith("…"))

    def test_python_result_transport_is_not_rendered(self):
        rendered = self.render(eko.response_renderable(
            "<python_result>secret</python_result>Visible"))
        self.assertEqual(rendered.strip(), "Visible")


class ModelTests(unittest.TestCase):
    def test_identity_is_configurable_and_location_is_neutral(self):
        prompt = eko.SYSTEM.format(name="Moa", folder="/workspace", mode="")
        self.assertTrue(prompt.startswith("You are Moa.\nYou are in /workspace.\n"))
        self.assertEqual(eko.NAME, "Eko")
        model = eko.ClaudeModel(
            Path("/host/private"), "fake", "low", name="Moa", folder="/workspace")
        self.assertEqual(model.cwd, Path("/host/private"))
        self.assertEqual(model.folder, "/workspace")

    def test_close_tolerates_concurrent_interrupt_clearing_process(self):
        model = eko.ClaudeModel(Path.cwd(), "fake", "low")
        stdout = io.BytesIO()

        class Process:
            def __init__(self):
                self.stdin = None
                self.stdout = stdout

            def poll(self):
                model.proc = None
                return 0

        model.proc = Process()
        model.close()
        self.assertTrue(stdout.closed)
        self.assertIsNone(model.proc)

    def test_failed_resume_can_recover_same_session(self):
        model = eko.ClaudeModel(Path.cwd(), "fake", "low")
        model.started = True
        session_id = model.session_id
        failed = json.dumps({
            "type": "result", "is_error": True,
            "result": "temporarily unavailable",
        }) + "\n"
        recovered = "".join(json.dumps(event) + "\n" for event in [
            {"type": "assistant", "message": {"content": [
                {"type": "text", "text": "recovered"}]}},
            {"type": "result", "is_error": False},
        ])
        starts = []

        def start():
            payload = failed if not starts else recovered
            starts.append(True)
            script = (
                "import sys; sys.stdin.buffer.readline(); "
                f"sys.stdout.buffer.write({payload.encode()!r}); "
                "sys.stdout.buffer.flush()")
            model.proc = subprocess.Popen(
                [sys.executable, "-c", script], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            model.started = True

        model._start = start
        self.assertEqual(model.ask("continue", lambda _: None), "recovered")
        self.assertEqual(len(starts), 2)
        self.assertEqual(model.session_id, session_id)
        model.close()

    def test_failed_resume_retries_without_resetting_session(self):
        model = eko.ClaudeModel(Path.cwd(), "fake", "low")
        model.started = True
        session_id = model.session_id
        event = json.dumps({
            "type": "result", "is_error": True,
            "result": "session unavailable",
        }) + "\n"
        script = (
            "import sys; sys.stdin.buffer.readline(); "
            f"sys.stdout.buffer.write({event.encode()!r}); "
            "sys.stdout.buffer.flush()")
        starts = []

        def start():
            starts.append(True)
            model.proc = subprocess.Popen(
                [sys.executable, "-c", script], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            model.started = True

        model._start = start
        with mock.patch.object(eko, "CALL_TIMEOUT", .5):
            with self.assertRaisesRegex(RuntimeError, "context was not reset"):
                model.ask(
                    "<python_result>orphaned</python_result>", lambda _: None)
        self.assertGreaterEqual(len(starts), 2)
        self.assertTrue(model.started)
        self.assertEqual(model.session_id, session_id)

    def test_multiple_events_buffered_in_one_write_do_not_stall(self):
        model = eko.ClaudeModel(Path.cwd(), "fake", "low")
        events = [
            {"type": "stream_event", "event": {
                "type": "content_block_delta",
                "delta": {"type": "text_delta", "text": "hello"}}},
            {"type": "assistant", "message": {
                "content": [{"type": "text", "text": "hello"}]}},
            {"type": "result", "is_error": False},
        ]
        payload = "".join(json.dumps(event) + "\n"
                          for event in events)
        script = (
            "import sys; sys.stdin.buffer.readline(); "
            f"sys.stdout.buffer.write({payload.encode()!r}); "
            "sys.stdout.buffer.flush()")

        def start():
            model.proc = subprocess.Popen(
                [sys.executable, "-c", script], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            model.started = True

        model._start = start
        streamed = []
        self.assertEqual(model.ask("hi", streamed.append), "hello")
        self.assertEqual(streamed, ["hello"])
        model.close()

    def test_interrupt_does_not_race_with_stdout_reader(self):
        model = eko.ClaudeModel(Path.cwd(), "fake", "low")
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True)
        model.proc = proc
        model.started = True
        errors = []

        def ask():
            try:
                model.ask("hello", lambda text: None)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=ask)
        thread.start()
        time.sleep(.05)
        model.interrupt()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], InterruptedError)
        assert proc.stdin and proc.stdout
        proc.stdin.close()
        proc.stdout.close()


@unittest.skipUnless(shutil.which("tmux"), "tmux is required")
class TerminalTests(unittest.TestCase):
    def test_rendered_panel_and_scrollback_retention(self):
        socket = f"eko-test-{os.getpid()}"

        def tmux(*args, capture=False):
            return subprocess.run(
                ["tmux", "-L", socket, *args], check=True, text=True,
                stdout=subprocess.PIPE if capture else subprocess.DEVNULL)

        command = f"cd {Path(__file__).parent} && {sys.executable} test_eko.py --demo"
        try:
            # Start from an interactive shell, as users do. The command line
            # consumes a terminal row that a direct-process launch does not.
            tmux("new-session", "-d", "-x", "100", "-y", "28",
                 "-s", "eko")
            time.sleep(.6)
            tmux("send-keys", "-t", "eko", command, "Enter")
            time.sleep(.8)
            initial = tmux(
                "capture-pane", "-p", "-t", "eko", capture=True).stdout
            self.assertIn("Eko\n", initial)
            self.assertIn(f"{Path(__file__).parent}  ·  fake  ·  signed in", initial)
            initial_lines = initial.splitlines()
            divider = initial_lines.index("─" * 100)
            self.assertEqual(initial_lines[divider + 1], "›")
            self.assertIn("enter send · ctrl+d exit", initial_lines[divider + 2])
            initial_cursor = tmux(
                "display-message", "-p", "-t", "eko", "#{cursor_y}",
                capture=True).stdout.strip()
            tmux("send-keys", "-t", "eko", "slow", "Enter")
            time.sleep(.1)
            after_send = tmux(
                "capture-pane", "-p", "-t", "eko", capture=True).stdout
            self.assertIn("› slow", after_send)
            self.assertIn("thinking", after_send)
            thinking_cursor = tmux(
                "display-message", "-p", "-t", "eko", "#{cursor_y}",
                capture=True).stdout.strip()
            tmux("send-keys", "-t", "eko", "after interrupt", "Enter")
            time.sleep(.1)
            queued_cursor = tmux(
                "display-message", "-p", "-t", "eko", "#{cursor_y}",
                capture=True).stdout.strip()
            tmux("send-keys", "-t", "eko", "Escape")
            time.sleep(.4)
            idle_cursor = tmux(
                "display-message", "-p", "-t", "eko", "#{cursor_y}",
                capture=True).stdout.strip()
            self.assertEqual(
                [initial_cursor, thinking_cursor, queued_cursor, idle_cursor],
                [initial_cursor] * 4)
            tmux("send-keys", "-t", "eko", "keep me visible", "Enter")
            time.sleep(.5)
            completed = tmux(
                "capture-pane", "-p", "-t", "eko", capture=True).stdout
            self.assertIn("› keep me visible", completed)
            self.assertIn("Received: keep me visible", completed)
            self.assertIn(
                "› keep me visible\n\nReceived: keep me visible", completed)
            tmux("send-keys", "-t", "eko", "long code", "Enter")
            time.sleep(1.2)
            streaming = tmux(
                "capture-pane", "-p", "-t", "eko", capture=True).stdout
            live_numbers = [int(number) for number in re.findall(
                r"│ print\((\d+)\)", streaming)]
            self.assertGreaterEqual(len(live_numbers), 1)
            self.assertGreaterEqual(max(live_numbers), 20)
            self.assertIn("esc interrupt · enter send", streaming)
            streaming_history = tmux(
                "capture-pane", "-p", "-t", "eko", "-S", "-500",
                capture=True).stdout
            self.assertEqual(streaming_history.count("› keep me visible"), 1)
            self.assertEqual(
                streaming_history.count("Received: keep me visible"), 1)
            self.assertIn("╭─ python", streaming_history)
            self.assertIn("│ print(0)", streaming_history)
            tmux("send-keys", "-t", "eko", "Escape")
            time.sleep(.8)
            settled_history = tmux(
                "capture-pane", "-p", "-t", "eko", "-S", "-500",
                capture=True).stdout
            self.assertIn("│ print(0)", settled_history)
            self.assertIn("╰", settled_history)
            self.assertEqual(settled_history.count("› keep me visible"), 1)
            tmux("send-keys", "-t", "eko", "complete long code", "Enter")
            time.sleep(.6)
            completed_long = tmux(
                "capture-pane", "-p", "-t", "eko", "-S", "-500",
                capture=True).stdout
            self.assertIn("│ print(0)", completed_long)
            self.assertIn("│ print(99)", completed_long)
            tmux("send-keys", "-t", "eko", "run code", "Enter")
            time.sleep(.3)
            for number in range(1, 31):
                tmux("send-keys", "-t", "eko", f"history-{number}", "Enter")
                time.sleep(.04)
            time.sleep(.5)
            screen = tmux(
                "capture-pane", "-p", "-t", "eko", "-S", "-500",
                capture=True).stdout
            self.assertIn("╭─ python ", screen)
            self.assertIn("VISIBLE_RESULT", screen)
            self.assertIn("╯\nExit 0 ·", screen)
            self.assertNotIn("✓ Exit", screen)
            self.assertIn("› history-1\n", screen)
            self.assertIn("Received: history-1\n", screen)
            self.assertIn("› history-30\n", screen)
            self.assertIn("Received: history-30\n", screen)
            self.assertTrue(screen.rstrip().endswith(
                "enter send · ctrl+d exit"))
            tmux("resize-window", "-t", "eko", "-y", "10")
            time.sleep(.2)
            small = tmux(
                "capture-pane", "-p", "-t", "eko", capture=True).stdout
            self.assertNotIn("Window too small", small)
            self.assertIn("enter send · ctrl+d exit", small)
        finally:
            subprocess.run(
                ["tmux", "-L", socket, "kill-server"],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class DemoModel(FakeModel):
    """Slow fake used by the tmux rendered-screen test."""

    def __init__(self):
        super().__init__(self.reply)

    def ask(self, message, write):
        if message != "long code":
            return super().ask(message, write)
        self.messages.append(message)
        self.cancelled.clear()
        parts = ["Streaming a large block.\n```python\n"]
        write(parts[0])
        for number in range(100):
            if self.cancelled.wait(.02):
                raise InterruptedError
            part = f"print({number})\n"
            parts.append(part)
            write(part)
        parts.append("```")
        write(parts[-1])
        return "".join(parts)

    def reply(self, message, cancelled):
        cancelled.clear()
        if message == "slow":
            for _ in range(100):
                if cancelled.wait(.02):
                    raise InterruptedError
            return "unexpected<done/>"
        if message == "after interrupt":
            return "Recovered after interrupt.<done/>"
        if message == "run code":
            return "Checking.\n```python\nprint('VISIBLE_RESULT')\n```"
        if message == "complete long code":
            code = "".join(f"print({number})\n" for number in range(100))
            return f"Completing a large block.\n```python\n{code}```"
        if message.startswith("<python_result>"):
            return "Finished.<done/>"
        return f"Received: {message}<done/>"


def run_demo() -> None:
    ui = eko.UI()
    ui.header(Path.cwd(), "fake")
    session = eko.Session(Path.cwd(), "fake", "low", ui)
    session.llm = DemoModel()
    ui.status = session.status
    ui.pending = session.pending
    ui.on_submit = session.submit
    ui.on_interrupt = session.interrupt

    def exit_app():
        session.stop()
        ui.exit()

    ui.on_exit = exit_app
    ui.on_start = session.start
    ui.run()
    session.thread.join(2)


if __name__ == "__main__" and "--demo" in sys.argv:
    run_demo()
elif __name__ == "__main__":
    unittest.main()
