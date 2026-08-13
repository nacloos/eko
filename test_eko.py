"""Deterministic state-machine and rendered-terminal tests for eko.py."""

from __future__ import annotations

import base64
import io
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from pathlib import Path

import eko
from rich.console import Console


def wait_until(predicate, timeout: float = 3) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("condition was not reached")


def event_text(events) -> str:
    return "\n\n".join(part.text for event in events for part in event.content
                         if isinstance(part, eko.Text))


def conversation(text: str) -> tuple[eko.Message, ...]:
    return (eko.Message("user", (eko.Text(text),)),)


class FakeUI:
    def __init__(self) -> None:
        self.messages: list[str] = []
        self.errors: list[str] = []
        self.results: list[eko.Result] = []


class FakeModel:
    def __init__(self, replies) -> None:
        self.replies = replies
        self.messages: list[str] = []
        self.histories: list[tuple[eko.Message, ...]] = []
        self.cancelled = threading.Event()

    def complete(self, messages, write):
        self.histories.append(messages)
        text = eko._message_text(messages[-1]).split("\n", 1)[-1]
        self.messages.append(text)
        reply = self.replies(text, self.cancelled)
        write(reply)
        return eko.Message("assistant", (eko.Text(reply),))

    def interrupt(self):
        self.cancelled.set()

    def close(self):
        pass


class CoreTests(unittest.TestCase):
    def test_each_input_has_one_text_budget_across_all_of_its_parts(self):
        image = eko.Image("image/png", b"image")
        incoming = eko.Input(
            "process-1", (eko.Text("a" * 12), image, eko.Text("b" * 12)), 7)

        limited = eko._limit_input(incoming, 10)

        self.assertEqual(limited.source, "process-1")
        self.assertEqual(limited.returncode, 7)
        self.assertEqual(limited.content, (
            eko.Text("a" * 5),
            eko.Text("\n\n… 14 characters omitted …\n\n"),
            image,
            eko.Text("b" * 5),
        ))

    def test_inputs_are_limited_independently(self):
        first = eko._limit_input(eko.Input("terminal", (eko.Text("a" * 12),)), 4)
        second = eko._limit_input(eko.Input("python", (eko.Text("b" * 12),)), 4)

        self.assertEqual(first.content[0], eko.Text("aa"))
        self.assertEqual(first.content[-1], eko.Text("aa"))
        self.assertEqual(second.content[0], eko.Text("bb"))
        self.assertEqual(second.content[-1], eko.Text("bb"))

    def test_model_content_uses_plain_attribution_headers(self):
        message = eko._user_message((
            eko.Input(eko.TERMINAL, (eko.Text("hello"),)),
            eko.Input(eko.PYTHON, (eko.Text("output"),), 1),
        ))
        content = eko._claude_content(message)
        self.assertEqual(content, [
            {"type": "text", "text": "[terminal]\n"},
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "\n\n[python exit=1]\n"},
            {"type": "text", "text": "output"},
        ])


@unittest.skipUnless(shutil.which("bwrap"), "Bubblewrap is not installed")
class SandboxTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.workspace = self.root / "workspace"
        self.workspace.mkdir()
        self.session = self.root / "session.sock"
        self.listener = socket.socket(socket.AF_UNIX)
        self.listener.bind(str(self.session))
        self.listener.listen()
        self.sandbox = eko.Sandbox(self.workspace, self.session)
        self.sandbox.start()

    def tearDown(self):
        self.sandbox.close()
        self.listener.close()
        self.temporary.cleanup()

    def test_background_process_survives_actions_and_dies_with_sandbox(self):
        result = self.sandbox.execute(r'''
import subprocess, sys
subprocess.Popen(
    [sys.executable, "-c",
     "import time; time.sleep(.2); open('alive', 'w').write('yes'); "
     "time.sleep(.5); open('escaped', 'w').write('no')"],
    stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL, start_new_session=True)
''', threading.Event())
        self.assertEqual(result.returncode, 0)
        time.sleep(.3)
        result = self.sandbox.execute("print(open('alive').read())", threading.Event())
        self.assertEqual(result.output, "yes\n")

        self.sandbox.close()
        time.sleep(.5)
        self.assertFalse((self.workspace / "escaped").exists())

    def test_process_can_create_a_nested_sandbox(self):
        (self.workspace / "child").mkdir()
        result = self.sandbox.execute(r'''
import subprocess
result = subprocess.run([
    "/usr/bin/bwrap", "--clearenv", "--unshare-user", "--unshare-pid",
    "--unshare-ipc", "--unshare-uts", "--unshare-net",
    "--ro-bind", "/usr", "/usr", "--symlink", "usr/bin", "/bin",
    "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
    "--bind", "/workspace/child", "/workspace",
    "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp",
    "--chdir", "/workspace", "/usr/bin/python3", "-c",
    "import os; print(os.getcwd())",
], text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
print(result.returncode)
print(result.stdout, end="")
''', threading.Event())

        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.output, "0\n/workspace\n")

    def test_background_process_can_reach_session_socket(self):
        result = self.sandbox.execute(r'''
import subprocess, sys
code = """import os, socket, time
time.sleep(.2)
with socket.socket(socket.AF_UNIX) as connection:
    connection.connect(os.environ[\"EKO_SESSION\"])
    connection.sendall(b'{\"type\":\"input\",\"content\":[{\"type\":\"text\",\"text\":\"done\"}]}\\n')
"""
subprocess.Popen(
    [sys.executable, "-c", code], stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
''', threading.Event())
        self.assertEqual(result.returncode, 0)

        self.listener.settimeout(2)
        connection, _ = self.listener.accept()
        with connection:
            event = json.loads(connection.makefile().readline())
        self.assertEqual(event["content"][0]["text"], "done")

    def test_interrupt_kills_only_the_action(self):
        interrupted = threading.Event()
        results = []
        thread = threading.Thread(target=lambda: results.append(
            self.sandbox.execute("import time; time.sleep(30)", interrupted)))
        thread.start()
        time.sleep(.2)
        interrupted.set()
        thread.join(2)

        self.assertFalse(thread.is_alive())
        self.assertIn("Interrupted", results[0].output)
        recovered = self.sandbox.execute("print('ready')", threading.Event())
        self.assertEqual(recovered.output, "ready\n")


class EkoTests(unittest.TestCase):
    def agent(self, replies):
        ui = FakeUI()
        def observe(event):
            if event.type == "response":
                ui.messages.append(event.value[0])
            elif event.type == "result":
                ui.results.append(event.value)
            elif event.type == "error":
                ui.errors.append(str(event.value))
        agent = eko.Eko(Path.cwd(), FakeModel(replies), observer=observe)
        return agent, ui

    def stop(self, agent):
        agent.stop()
        agent.wait(2)
        assert agent.thread is not None
        self.assertFalse(agent.thread.is_alive())

    def test_agent_owns_provider_neutral_conversation(self):
        agent, _ui = self.agent(lambda *_: "finished<done/>")
        agent.start("hello")
        wait_until(lambda: len(agent.messages) == 2)

        self.assertEqual([message.role for message in agent.messages],
                         ["user", "assistant"])
        self.assertEqual(eko._message_text(agent.messages[0]), "[terminal]\nhello")
        self.assertEqual(eko._message_text(agent.messages[1]), "finished<done/>")
        self.assertEqual(agent.completer.histories, [(agent.messages[0],)])
        self.stop(agent)

    def test_terminal_input_is_limited_before_the_model(self):
        agent, _ui = self.agent(lambda *_: "<done/>")
        agent.start("a" * (eko.MAX_INPUT_TEXT + 100))
        wait_until(lambda: len(agent.completer.messages) == 1)

        message = agent.completer.messages[0]
        self.assertIn("100 characters omitted", message)
        self.assertLess(len(message), eko.MAX_INPUT_TEXT + 100)
        self.stop(agent)

    def test_session_can_use_a_caller_provided_executor(self):
        calls = []

        def execute(code, interrupted):
            calls.append((code, interrupted))
            return eko.Result("custom\n", 0, .25)

        ui = FakeUI()
        agent = eko.Eko(Path.cwd(), FakeModel(lambda *_: "<done/>"), executor=execute)
        result = agent._execute("print('custom')")

        self.assertEqual(result, eko.Result("custom\n", 0, .25))
        self.assertEqual(calls, [("print('custom')", agent.interrupted)])
        self.assertEqual(agent.state, "running Python")
        agent.stop()

    def test_interrupt_continues_with_message_typed_while_busy(self):
        def replies(message, cancelled):
            if message == "slow":
                while not cancelled.wait(.01):
                    pass
                raise InterruptedError
            return "recovered<done/>"

        agent, ui = self.agent(replies)
        agent.start("slow")
        wait_until(lambda: agent.state == "thinking")
        agent.send("after interrupt")
        agent.interrupt()
        wait_until(lambda: agent.completer.messages == ["slow", "after interrupt"])
        wait_until(lambda: agent.state == "idle")
        self.assertEqual(ui.errors, ["Interrupted"])
        self.stop(agent)

    def test_model_error_does_not_kill_worker(self):
        def replies(message, cancelled):
            if message == "broken":
                raise RuntimeError("broken connection")
            return "recovered<done/>"

        agent, ui = self.agent(replies)
        agent.start("broken")
        wait_until(lambda: ui.errors == ["broken connection"])
        agent.send("next")
        wait_until(lambda: agent.completer.messages == ["broken", "next"])
        assert agent.thread is not None
        self.assertTrue(agent.thread.is_alive())
        self.stop(agent)

    def test_python_output_precedes_input_received_during_execution(self):
        def replies(message, cancelled):
            if message == "run":
                return "```python-run\nimport time\ntime.sleep(.2)\nprint('done')\n```"
            self.assertTrue(message.startswith("done\n"))
            self.assertTrue(message.endswith("typed while running"))
            return "finished<done/>"

        agent, ui = self.agent(replies)
        agent.start("run")
        wait_until(lambda: agent.state == "running Python")
        agent.send("typed while running")
        wait_until(lambda: len(agent.completer.messages) == 2)
        wait_until(lambda: agent.state == "idle")
        self.assertEqual(ui.results[0].output, "done\n")
        self.stop(agent)

    def test_predicted_attribution_adds_a_harness_warning(self):
        def replies(message, _cancelled):
            if message == "start":
                return ("```python-run\nprint('actual')\n```\n\n"
                        "[python exit=0]\npredicted")
            self.assertIn("actual", message)
            self.assertIn("do not predict the contents", message)
            return "<done/>"

        agent, _ui = self.agent(replies)
        agent.start("start")
        wait_until(lambda: len(agent.completer.messages) == 2)
        self.stop(agent)

    def test_attribution_text_inside_executable_python_does_not_warn(self):
        def replies(message, _cancelled):
            if message == "start":
                return "```python-run\nprint('[python exit=0]')\n```"
            self.assertNotIn("do not predict the contents", message)
            return "<done/>"

        agent, _ui = self.agent(replies)
        agent.start("start")
        wait_until(lambda: len(agent.completer.messages) == 2)
        self.stop(agent)

    def test_detached_python_can_send_attributed_input(self):
        callback = "background job finished"

        def replies(message, cancelled):
            if message == "start":
                return """```python-run
import json, os, socket, subprocess, sys
code = '''
import json, os, socket, time
time.sleep(.2)
event = {"type":"input", "content":[
    {"type":"text", "text":"background job finished"}]}
payload = json.dumps(event).encode()
with socket.socket(socket.AF_UNIX) as client:
    client.connect(os.environ["EKO_SESSION"])
    client.sendall(payload + b"\\\\n")
'''
subprocess.Popen(
    [sys.executable, "-c", code],
    stdin=subprocess.DEVNULL,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)
print("started")
```"""
            if message.startswith("started"):
                return "foreground turn finished<done/>"
            self.assertEqual(message, callback)
            return "callback received<done/>"

        agent, ui = self.agent(replies)
        agent.start("start")
        wait_until(lambda: callback in agent.completer.messages)
        wait_until(lambda: agent.state == "idle")
        self.assertEqual(agent.completer.messages[-1], callback)
        self.assertEqual(ui.results[0].output, "started\n")
        self.stop(agent)

    def test_external_input_while_busy_joins_the_next_model_turn(self):
        release = threading.Event()

        def replies(message, cancelled):
            if message == "busy":
                release.wait(2)
                return "done<done/>"
            return "callback handled<done/>"

        agent, ui = self.agent(replies)
        agent.start("busy")
        wait_until(lambda: agent.state == "thinking")
        event = {"type": "input", "content": [
            {"type": "text", "text": "arrived while busy"}]}
        payload = json.dumps(event).encode()
        with socket.socket(socket.AF_UNIX) as client:
            client.connect(str(agent.socket_path))
            client.sendall(payload + b"\n")
        wait_until(lambda: agent.pending() == ["arrived while busy"])
        release.set()
        wait_until(lambda: agent.completer.messages == ["busy", "arrived while busy"])
        self.stop(agent)

    def test_external_multimodal_input_has_attested_process_provenance(self):
        png = base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode()
        event = eko._parse_input({
            "type": "input",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image", "media_type": "image/png", "data": png},
            ],
            "source": {"kind": "operator", "label": "forged"},
        }, "process-1", Path.cwd())

        self.assertEqual(event.source, "process-1")
        self.assertEqual(event.content[0], eko.Text("look"))
        self.assertIsInstance(event.content[1], eko.Image)


class RenderingTests(unittest.TestCase):
    def test_python_execution_uses_the_original_running_activity_label(self):
        ui = eko.UI.__new__(eko.UI)
        labels = []
        ui._start_activity = labels.append
        ui._stop_activity = lambda: None
        agent = mock.Mock()

        ui.connect(agent)
        agent.observer(eko.Event("state", "running Python"))

        self.assertEqual(labels, ["running"])

    def render(self, renderable) -> str:
        stream = io.StringIO()
        Console(file=stream, force_terminal=False, width=80).print(renderable)
        return stream.getvalue()

    def test_partial_python_fence_renders_inside_panel(self):
        rendered = self.render(eko.response_renderable(
            "Looking.\n```python-run\nprint('still streaming')"))
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
        stream.feed("Starting.\n```python-")
        stream.feed("run\nprint(0)\nprint")
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
            f"{ticks}python-run\n"
            f'print("inside: {ticks}python")\n'
            f"{ticks}\nDone.")
        self.assertEqual(
            eko._python(response),
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
        stream.feed("```python-run\nprint(1)\n`")
        stream.feed("``\nAfter.")
        stream.finish()
        rendered = "".join(output)
        self.assertEqual(rendered.count("print(1)"), 1)
        self.assertNotIn("│ ```", rendered)
        self.assertIn("After.", rendered)

    def test_py_fence_is_displayed_but_not_executed(self):
        response = "Example:\n```py\nprint('display only')\n```\n"

        self.assertIsNone(eko._python(response))
        stream = io.StringIO()
        Console(file=stream, force_terminal=True, color_system="truecolor",
                width=80).print(eko.response_renderable(response))
        rendered = stream.getvalue()
        self.assertIn("display only", rendered)
        self.assertIn(" print('display only')", rendered)

    def test_plain_python_fence_is_displayed_but_not_executed(self):
        response = "Example:\n```python\nprint('display only')\n```\n"

        self.assertIsNone(eko._python(response))
        self.assertIn("display only", self.render(eko.response_renderable(response)))

    def test_stream_has_no_hidden_transport_tags(self):
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
        self.assertIn("1150 lines", rendered)
        self.assertIn("fabricated listing", rendered)

    def test_native_stream_renders_markdown_python_without_executing_it(self):
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
        self.assertIsNone(eko._python(stream.text))
        self.assertNotIn("**formatted**", rendered)
        self.assertIn("formatted", rendered)
        self.assertEqual(rendered.count("def answer"), 1)
        self.assertEqual(rendered.count("return 42"), 1)

    def test_code_row_uses_gold_only_for_its_border(self):
        captured = []

        class Sink:
            _append = lambda self, text: None
            _render = lambda self, item: captured.append(item.copy()) or ""

        stream = eko.NativeStream(Sink())
        stream._code_line(eko.RichText("ordinary_name"))
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

    def test_response_renderer_has_no_special_result_transport(self):
        rendered = self.render(eko.response_renderable(
            "<python_result>secret</python_result>Visible"))
        self.assertEqual(rendered.strip(),
                         "<python_result>secret</python_result>Visible")


class ModelTests(unittest.TestCase):
    def test_identity_is_configurable_and_location_is_neutral(self):
        prompt = eko.SYSTEM.format(name="Moa", folder="/workspace", mode="")
        self.assertTrue(prompt.startswith("You are Moa.\nYou are in /workspace.\n"))
        self.assertEqual(eko.NAME, "Eko")
        model = eko.Claude(
            Path("/host/private"), "fake", "low", name="Moa", folder="/workspace")
        self.assertEqual(model.cwd, Path("/host/private"))
        self.assertEqual(model.folder, "/workspace")

    def test_sandbox_hides_host_location_from_model(self):
        agent = mock.Mock(socket_path=Path("/tmp/session.sock"))
        with mock.patch.object(eko, "ensure_auth"), \
             mock.patch.object(eko, "Claude") as claude, \
             mock.patch.object(eko, "Eko", return_value=agent):
            eko.run(Path.cwd(), headless=True, sandbox=True)

        self.assertEqual(claude.call_args.args[-1], "/workspace")

    def test_close_tolerates_concurrent_interrupt_clearing_process(self):
        model = eko.Claude(Path.cwd(), "fake", "low")
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

    def test_exact_empty_text_resume_error_repairs_only_its_assistant_block(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "projects" / "project"
            project.mkdir(parents=True)
            model = eko.Claude(Path.cwd(), "fake", "low")
            model.started = True
            agent = project / f"{model.session_id}.jsonl"
            unchanged = '{"type":"user","message":{"content":[{"type":"text","text":""}]}}\n'
            broken = '{"type":"assistant","message":{"content":[{"type":"thinking","thinking":""},{"type":"text","text":""}]}}\n'
            agent.write_text(unchanged + broken)
            failed = json.dumps({
                "type": "result", "is_error": True,
                "result": "API Error: 400 messages: text content blocks must be non-empty",
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
                script = ("import sys; sys.stdin.buffer.readline(); "
                          f"sys.stdout.buffer.write({payload.encode()!r}); "
                          "sys.stdout.buffer.flush()")
                model.proc = subprocess.Popen(
                    [sys.executable, "-c", script], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)

            model._start = start
            with mock.patch.dict(os.environ, {"CLAUDE_CONFIG_DIR": directory}):
                reply = model.complete(conversation("continue"), lambda _: None)
                self.assertEqual(eko._message_text(reply), "recovered")
                model.close()

            lines = agent.read_text().splitlines(keepends=True)
            self.assertEqual(lines[0], unchanged)
            content = json.loads(lines[1])["message"]["content"]
            self.assertEqual(content[0]["thinking"], "")
            self.assertEqual(content[1]["text"], " ")
            self.assertEqual(len(starts), 2)

    def test_failed_resume_can_recover_same_session(self):
        model = eko.Claude(Path.cwd(), "fake", "low")
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
        reply = model.complete(conversation("continue"), lambda _: None)
        self.assertEqual(eko._message_text(reply), "recovered")
        self.assertEqual(len(starts), 2)
        self.assertEqual(model.session_id, session_id)
        model.close()

    def test_failed_resume_retries_without_resetting_session(self):
        model = eko.Claude(Path.cwd(), "fake", "low")
        model.started = True
        session_id = model.session_id
        event = json.dumps({
            "type": "result", "is_error": True,
            "result": "agent unavailable",
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
                model.complete(conversation("orphaned output"), lambda _: None)
        self.assertGreaterEqual(len(starts), 2)
        self.assertTrue(model.started)
        self.assertEqual(model.session_id, session_id)

    def test_multiple_events_buffered_in_one_write_do_not_stall(self):
        model = eko.Claude(Path.cwd(), "fake", "low")
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
        reply = model.complete(conversation("hi"), streamed.append)
        self.assertEqual(eko._message_text(reply), "hello")
        self.assertEqual(streamed, ["hello"])
        model.close()

    def test_interrupt_does_not_race_with_stdout_reader(self):
        model = eko.Claude(Path.cwd(), "fake", "low")
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True)
        model.proc = proc
        model.started = True
        errors = []

        def complete():
            try:
                model.complete(conversation("hello"), lambda text: None)
            except BaseException as error:
                errors.append(error)

        thread = threading.Thread(target=complete)
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
            self.assertIn("esc interrupt · enter send", after_send)
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

    def complete(self, messages, write):
        text = eko._message_text(messages[-1]).split("\n", 1)[-1]
        if text != "long code":
            return super().complete(messages, write)
        self.messages.append("long code")
        self.cancelled.clear()
        parts = ["Streaming a large block.\n```python-run\n"]
        write(parts[0])
        for number in range(100):
            if self.cancelled.wait(.02):
                raise InterruptedError
            part = f"print({number})\n"
            parts.append(part)
            write(part)
        parts.append("```")
        write(parts[-1])
        return eko.Message("assistant", (eko.Text("".join(parts)),))

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
            return "Checking.\n```python-run\nprint('VISIBLE_RESULT')\n```"
        if message == "complete long code":
            code = "".join(f"print({number})\n" for number in range(100))
            return f"Completing a large block.\n```python-run\n{code}```"
        if message.startswith("VISIBLE_RESULT"):
            return "Finished.<done/>"
        return f"Received: {message}<done/>"


def run_demo() -> None:
    ui = eko.UI()
    ui.header(Path.cwd(), "fake")
    agent = eko.Eko(Path.cwd(), DemoModel())
    ui.connect(agent)

    def exit_app():
        agent.stop()
        ui.exit()

    ui.on_exit = exit_app
    ui.on_start = agent.start
    ui.run()
    agent.wait(2)


if __name__ == "__main__" and "--demo" in sys.argv:
    run_demo()
elif __name__ == "__main__":
    unittest.main()
