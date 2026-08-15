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
from types import SimpleNamespace

import eko
import host
import tinker_model
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
        self.events: list[eko.Event] = []
        self.messages: list[str] = []
        self.errors: list[str] = []
        self.results: list[eko.Result] = []


class FakeModel:
    def __init__(self, replies) -> None:
        self.replies = replies
        self.messages: list[str] = []
        self.system = None
        self.cancelled = threading.Event()
        self.context_used = 0
        self.resets = 0

    def start(self, system):
        self.system = system

    def send(self, message, write):
        text = eko.message_text(message).split("\n", 1)[-1]
        self.messages.append(text)
        reply = self.replies(text, self.cancelled)
        write(reply)
        return eko.Message("assistant", (eko.Text(reply),))

    def complete(self, system, message, write):
        self.start(system)
        return self.send(message, write)

    def interrupt(self):
        self.cancelled.set()

    def reset(self, system):
        self.system = system
        self.context_used = 0
        self.resets += 1

    def close(self):
        pass


class CoreTests(unittest.TestCase):
    def test_python_actions_default_to_thirty_seconds(self):
        agent = eko.Eko(Path.cwd(), FakeModel(lambda *_: "<done/>"))
        self.assertEqual(agent.python_timeout, 30)

    def test_python_action_timeout_is_configurable(self):
        result = eko._run_python(
            "import time; time.sleep(10)", Path.cwd(), threading.Event(),
            timeout=.01,
        )
        self.assertIn("TIMEOUT after 0.01s", result.output)
        self.assertNotEqual(result.returncode, 0)

    def test_clean_workspace_guidance_is_opt_in(self):
        plain = eko.Eko(Path.cwd(), FakeModel(lambda *_: "<done/>"))
        clean = eko.Eko(
            Path.cwd(), FakeModel(lambda *_: "<done/>"), clean_workspace=True
        )

        self.assertNotIn(eko.CLEAN_WORKSPACE, plain.system)
        self.assertIn(eko.CLEAN_WORKSPACE, clean.system)

    def test_context_status_line_is_not_a_provenance_header(self):
        self.assertEqual(
            eko.context_status_line(64_000, 128_000),
            "context 64k/128k (50%)",
        )

    def test_context_usage_is_observable_when_enabled(self):
        model = FakeModel(lambda *_: "<done/>")
        model.context_used = 64_000
        events = []
        agent = eko.Eko(Path.cwd(), model, observer=events.append, context=128_000)
        agent.start("hello")
        wait_until(lambda: len(model.messages) == 1 and agent.state == "idle")

        self.assertIn(eko.Event("context", (64_000, 128_000)), events)
        agent.stop()
        agent.wait(2)

    def test_context_usage_is_not_observable_when_disabled(self):
        model = FakeModel(lambda *_: "<done/>")
        model.context_used = 64_000
        events = []
        agent = eko.Eko(Path.cwd(), model, observer=events.append)
        agent.start("hello")
        wait_until(lambda: len(model.messages) == 1 and agent.state == "idle")

        self.assertFalse(any(event.type == "context" for event in events))
        agent.stop()
        agent.wait(2)

    def test_each_input_has_one_text_budget_across_all_of_its_parts(self):
        image = eko.Image("image/png", b"image")
        incoming = eko.Input(
            "process-1", (eko.Text("a" * 12), image, eko.Text("b" * 12)), 7)

        limited = eko.limit_input(incoming, 10)

        self.assertEqual(limited.source, "process-1")
        self.assertEqual(limited.returncode, 7)
        self.assertEqual(limited.content, (
            eko.Text("a" * 5),
            eko.Text("\n\n… 14 characters omitted …\n\n"),
            image,
            eko.Text("b" * 5),
        ))

    def test_inputs_are_limited_independently(self):
        first = eko.limit_input(eko.Input("terminal", (eko.Text("a" * 12),)), 4)
        second = eko.limit_input(eko.Input("python", (eko.Text("b" * 12),)), 4)

        self.assertEqual(first.content[0], eko.Text("aa"))
        self.assertEqual(first.content[-1], eko.Text("aa"))
        self.assertEqual(second.content[0], eko.Text("bb"))
        self.assertEqual(second.content[-1], eko.Text("bb"))

    def test_model_content_uses_plain_attribution_headers(self):
        message = eko.user_message((
            eko.Input(eko.TERMINAL, (eko.Text("hello"),)),
            eko.Input(eko.PYTHON, (eko.Text("output"),), 1),
        ))
        content = tinker_model._message(message)
        self.assertEqual(content, {
            "role": "user",
            "content": "[terminal]\nhello\n\n[python exit=1]\noutput",
        })


class ModelSocketTests(unittest.TestCase):
    def test_malformed_model_events_close_the_connection(self):
        payloads = (
            b"", b"\n", b"null\n", b"{}\n",
            b'{"system":"test"}\ninvalid\n',
            b'{"system":"test"}\n[]\n',
        )
        for payload in payloads:
            with self.subTest(payload=payload):
                server, client = socket.socketpair()
                client.sendall(payload)
                client.shutdown(socket.SHUT_WR)
                model = FakeModel(lambda text, _cancelled: text)
                with mock.patch.object(host, "Tinker", return_value=model):
                    host._model_client(server, Path.cwd(), "fake", "low")
                client.close()

    def test_model_connection_streams_and_returns_messages(self):
        with tempfile.TemporaryDirectory() as directory:
            endpoint = Path(directory) / "model.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(endpoint))
            listener.listen()
            model = FakeModel(lambda text, _cancelled: text.upper())

            def serve():
                connection, _ = listener.accept()
                host._model_client(
                    connection, Path.cwd(), "fake", "low")

            with mock.patch.object(host, "Tinker", return_value=model):
                thread = threading.Thread(target=serve, daemon=True)
                thread.start()
                remote = eko.Model(endpoint)
                remote.start(eko.SYSTEM)
                streamed = []
                reply = remote.send(conversation("hello")[0], streamed.append)
                remote.close()
                thread.join(2)
            listener.close()

        self.assertEqual(reply, eko.Message("assistant", (eko.Text("HELLO"),)))
        self.assertEqual(streamed, ["HELLO"])

    def test_message_socket_round_trips_images(self):
        message = eko.Message("user", (
            eko.Text("look"), eko.Image("image/png", b"image", "x.png")))
        self.assertEqual(eko.decode_message(eko.encode_message(message)), message)

    def test_process_events_round_trip(self):
        events = (
            eko.Event("state", "thinking"),
            eko.Event("input", eko.Message("user", (
                eko.Text("[terminal]\n"), eko.Text("hello")))),
            eko.Event("response", ("answer", "print(1)")),
            eko.Event("result", eko.Result("1\n", 0, .1)),
        )
        self.assertEqual(
            tuple(eko.decode_event(eko.encode_event(event)) for event in events),
            events)


class ProcessInterfaceTests(unittest.TestCase):
    def test_child_does_not_reuse_inherited_parent_session(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            endpoint = root / "model.sock"
            parent_session = root / "parent.sock"
            parent_session.write_text("parent")
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(endpoint))
            listener.listen()
            environment = os.environ.copy()
            environment.update({
                "EKO_MODEL": str(endpoint),
                "EKO_SESSION": str(parent_session),
            })
            process = subprocess.Popen([
                sys.executable, str(Path(eko.__file__).resolve()),
                "--cwd", str(root),
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
               stderr=subprocess.PIPE, text=True, env=environment)
            assert process.stdin and process.stdout and process.stderr
            while True:
                event = json.loads(process.stdout.readline())
                if event["type"] == "ready":
                    break
            self.assertNotEqual(event["session"], str(parent_session))
            self.assertEqual(parent_session.read_text(), "parent")
            process.stdin.write('{"type":"stop"}\n')
            process.stdin.flush()
            process.wait(5)
            process.stdin.close()
            process.stdout.close()
            process.stderr.close()
            listener.close()

    def test_core_runs_behind_json_stdio_and_a_model_socket(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            endpoint = root / "model.sock"
            session = root / "session.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(endpoint))
            listener.listen()
            requests = []

            def complete():
                connection, _ = listener.accept()
                with connection, connection.makefile("rb") as reader:
                    initial = json.loads(reader.readline())
                    request = json.loads(reader.readline())
                    request["system"] = initial["system"]
                    requests.append(request)
                    reply = eko.encode_message(eko.Message(
                        "assistant", (eko.Text("finished<done/>"),)))
                    for event in (
                            {"delta": "finished<done/>"},
                            {"message": reply}):
                        connection.sendall((json.dumps(event) + "\n").encode())

            thread = threading.Thread(target=complete, daemon=True)
            thread.start()
            process = subprocess.Popen([
                sys.executable, str(Path(eko.__file__).resolve()),
                "--cwd", str(root), "--model-socket", str(endpoint),
                "--session-socket", str(session), "--name", "Moa",
            ], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
               stderr=subprocess.PIPE, text=True)
            assert process.stdin and process.stdout
            startup = []
            while True:
                startup.append(json.loads(process.stdout.readline()))
                if startup[-1]["type"] == "ready":
                    break
            ready = startup[-1]
            self.assertEqual(ready, {"type": "ready", "session": str(session)})
            process.stdin.write(json.dumps({
                "type": "input",
                "content": [{"type": "text", "text": "hello"}],
            }) + "\n")
            process.stdin.flush()
            events = []
            while not any(event["type"] == "response" for event in events):
                events.append(json.loads(process.stdout.readline()))
            process.stdin.write('{"type":"stop"}\n')
            process.stdin.flush()
            process.wait(5)
            process.stdin.close()
            process.stdout.close()
            assert process.stderr
            process.stderr.close()
            listener.close()
            thread.join(2)

        self.assertEqual(process.returncode, 0)
        self.assertIn("You are Moa.", requests[0]["system"])
        self.assertEqual(
            eko.message_text(eko.decode_message(requests[0]["message"])),
            "[terminal]\nhello")
        self.assertTrue(any(event["type"] == "delta" for event in events))


class EkoTests(unittest.TestCase):
    def agent(self, replies):
        ui = FakeUI()
        def observe(event):
            ui.events.append(event)
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
        self.assertEqual(eko.message_text(agent.messages[0]), "[terminal]\nhello")
        self.assertEqual(eko.message_text(agent.messages[1]), "finished<done/>")
        self.assertIn("You are Eko.", agent.model.system)
        self.stop(agent)

    def test_feral_agent_starts_without_a_prompt(self):
        model = FakeModel(lambda *_: "```python-run\nimport time\ntime.sleep(60)\n```")
        agent = eko.Eko(Path.cwd(), model, feral=True)
        agent.start()
        wait_until(lambda: len(agent.messages) == 2)

        self.assertNotIn("Feral mode", model.system)
        self.assertEqual(
            eko.message_text(agent.messages[0]), "[harness]\nBegin."
        )
        self.stop(agent)

    def test_context_notices_warn_then_reset_to_opening_input(self):
        usages = iter((50_000, 90_000, 95_000, 96_000, 0))
        model = None

        def replies(message, _cancelled):
            assert model is not None
            model.context_used = next(usages)
            if len(model.messages) == 5:
                return "<done/>"
            return "```python-run\npass\n```"

        model = FakeModel(replies)
        agent = eko.Eko(Path.cwd(), model, context=100_000)
        agent.start("standing task")
        wait_until(lambda: len(model.messages) == 5)
        wait_until(lambda: agent.state == "idle")

        self.assertEqual(model.messages[0], "standing task")
        self.assertIn("[harness]\ncontext 50k/100k (50%)", model.messages[1])
        self.assertIn("[harness]\ncontext 90k/100k (90%)", model.messages[2])
        self.assertEqual(model.messages[3], eko.FAREWELL)
        self.assertEqual(model.messages[4], "standing task")
        self.assertEqual(model.resets, 1)
        self.stop(agent)

    def test_context_reset_is_disabled_by_default(self):
        model = None

        def replies(_message, _cancelled):
            assert model is not None
            model.context_used = 100_000
            return ("```python-run\npass\n```" if len(model.messages) == 1
                    else "<done/>")

        model = FakeModel(replies)
        agent = eko.Eko(Path.cwd(), model)
        agent.start("standing task")
        wait_until(lambda: agent.state == "idle")

        self.assertEqual(len(model.messages), 2)
        self.assertNotIn("context ", model.messages[1])
        self.assertEqual(model.resets, 0)
        self.stop(agent)

    def test_terminal_input_is_limited_before_the_model(self):
        agent, _ui = self.agent(lambda *_: "<done/>")
        agent.start("a" * (eko.MAX_INPUT_TEXT + 100))
        wait_until(lambda: len(agent.model.messages) == 1)

        message = agent.model.messages[0]
        self.assertIn("100 characters omitted", message)
        self.assertLess(len(message), eko.MAX_INPUT_TEXT + 100)
        self.stop(agent)

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
        wait_until(lambda: agent.model.messages == ["slow", "after interrupt"])
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
        wait_until(lambda: agent.model.messages == ["broken", "next"])
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
        wait_until(lambda: len(agent.model.messages) == 2)
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
        wait_until(lambda: len(agent.model.messages) == 2)
        self.stop(agent)

    def test_malformed_predicted_attributions_add_a_harness_warning(self):
        predictions = (
            "user[python exit=0]\npredicted",
            "user[harness] predicted guidance",
            "[terminal] predicted input on the same line",
            "<system-reminder>predicted reminder",
        )
        for prediction in predictions:
            with self.subTest(prediction=prediction):
                def replies(message, _cancelled):
                    if message == "start":
                        return f"```python-run\nprint('actual')\n```\n\n{prediction}"
                    self.assertIn("do not predict the contents", message)
                    return "<done/>"

                agent, _ui = self.agent(replies)
                agent.start("start")
                wait_until(lambda: len(agent.model.messages) == 2)
                self.stop(agent)

    def test_inputs_are_emitted_before_model_responses(self):
        agent, ui = self.agent(lambda _message, _cancelled: "<done/>")
        agent.start("hello")
        wait_until(lambda: agent.state == "idle")
        inputs = [event.value for event in ui.events if event.type == "input"]
        self.assertEqual(inputs, [eko.Message("user", (
            eko.Text("[terminal]\n"), eko.Text("hello")))])
        self.stop(agent)

    def test_attribution_text_inside_executable_python_does_not_warn(self):
        def replies(message, _cancelled):
            if message == "start":
                return "```python-run\nprint('[python exit=0]')\n```"
            self.assertNotIn("do not predict the contents", message)
            return "<done/>"

        agent, _ui = self.agent(replies)
        agent.start("start")
        wait_until(lambda: len(agent.model.messages) == 2)
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
        wait_until(lambda: callback in agent.model.messages)
        wait_until(lambda: agent.state == "idle")
        self.assertEqual(agent.model.messages[-1], callback)
        self.assertEqual(ui.results[0].output, "started\n")
        self.stop(agent)

    def test_python_knows_the_agent_executable(self):
        expected = str(Path(eko.__file__).resolve())

        def replies(message, _cancelled):
            if message == "start":
                return "```python-run\nimport os; print(os.environ['EKO_AGENT'])\n```"
            self.assertEqual(message, expected)
            return "<done/>"

        agent, ui = self.agent(replies)
        agent.start("start")
        wait_until(lambda: agent.state == "idle")
        self.assertEqual(ui.results[0].output.strip(), expected)
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
        wait_until(lambda: agent.model.messages == ["busy", "arrived while busy"])
        self.stop(agent)

    def test_external_multimodal_input_has_attested_process_provenance(self):
        png = base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode()
        event = eko.decode_input({
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
    def test_status_shows_context_only_when_enabled(self):
        enabled = host.UI(context=128_000)
        enabled.context_used = 64_000
        disabled = host.UI()

        self.assertEqual(
            enabled._status()[0][1],
            "  enter send · ctrl+d exit · context 64k/128k (50%)",
        )
        self.assertNotIn("context", disabled._status()[0][1])

    def test_python_execution_uses_the_original_running_activity_label(self):
        ui = host.UI.__new__(host.UI)
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
        rendered = self.render(host.response_renderable(
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

        stream = host.NativeStream(Sink())
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
            eko.executable_python(response),
            f'print("inside: {ticks}python")\n')

        output = []

        class Sink:
            _append = lambda self, text: output.append(text)
            _render = lambda self, item: RenderingTests.render(self, item)

        stream = host.NativeStream(Sink())
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

        stream = host.NativeStream(Sink())
        stream.feed("```python-run\nprint(1)\n`")
        stream.feed("``\nAfter.")
        stream.finish()
        rendered = "".join(output)
        self.assertEqual(rendered.count("print(1)"), 1)
        self.assertNotIn("│ ```", rendered)
        self.assertIn("After.", rendered)

    def test_py_fence_is_displayed_but_not_executed(self):
        response = "Example:\n```py\nprint('display only')\n```\n"

        self.assertIsNone(eko.executable_python(response))
        stream = io.StringIO()
        Console(file=stream, force_terminal=True, color_system="truecolor",
                width=80).print(host.response_renderable(response))
        rendered = stream.getvalue()
        self.assertIn("display only", rendered)
        self.assertIn(" print('display only')", rendered)

    def test_plain_python_fence_is_displayed_but_not_executed(self):
        response = "Example:\n```python\nprint('display only')\n```\n"

        self.assertIsNone(eko.executable_python(response))
        self.assertIn("display only", self.render(host.response_renderable(response)))

    def test_stream_has_no_hidden_transport_tags(self):
        output = []

        class Sink:
            _append = lambda self, text: output.append(text)
            _render = lambda self, item: RenderingTests.render(self, item)

        stream = host.NativeStream(Sink())
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

        stream = host.NativeStream(Sink())
        stream.feed("This is **formatted**.\n\n```python\n")
        stream.feed("def answer():\n    return 42\n```")
        stream.finish()
        rendered = "".join(output)
        self.assertIsNone(eko.executable_python(stream.text))
        self.assertNotIn("**formatted**", rendered)
        self.assertIn("formatted", rendered)
        self.assertEqual(rendered.count("def answer"), 1)
        self.assertEqual(rendered.count("return 42"), 1)

    def test_code_row_uses_gold_only_for_its_border(self):
        captured = []

        class Sink:
            _append = lambda self, text: None
            _render = lambda self, item: captured.append(item.copy()) or ""

        stream = host.NativeStream(Sink())
        stream._code_line(host.RichText("ordinary_name"))
        row = captured[0]
        self.assertEqual(row.style, "")
        self.assertEqual(row.plain[2:15], "ordinary_name")
        self.assertTrue(any(span.style == host.GOLD for span in row.spans))

    def test_display_output_reports_omitted_lines(self):
        output = host.display_output("\n".join(f"line {i}" for i in range(50)))
        self.assertIn("… +46 lines", output)
        self.assertIn("line 0", output)
        self.assertIn("line 49", output)

    def test_display_output_clips_lines_to_terminal_width(self):
        output = host.display_output("x" * 200, width=40)
        self.assertEqual(len(output), 40)
        self.assertTrue(output.endswith("…"))

    def test_response_renderer_has_no_special_result_transport(self):
        rendered = self.render(host.response_renderable(
            "<python_result>secret</python_result>Visible"))
        self.assertEqual(rendered.strip(),
                         "<python_result>secret</python_result>Visible")


class TinkerModelTests(unittest.TestCase):
    def test_agent_command_forwards_python_timeout(self):
        command = host._agent_command(
            Path.cwd(), Path("/tmp/runtime"), sandbox=False, feral=False,
            name="Eko", python_timeout=12.5,
        )
        index = command.index("--python-timeout")
        self.assertEqual(command[index + 1], "12.5")

    class Prompt:
        def __init__(self, number):
            self.number = number

        def to_ints(self):
            return list(range(self.number))

    class Renderer:
        def __init__(self):
            self.calls = []

        def build_generation_prompt(self, messages, **options):
            self.calls.append((messages, options))
            return TinkerModelTests.Prompt(len(messages) * 10)

        def get_stop_sequences(self):
            return [123]

        def parse_response(self, tokens):
            return tokens, "stop"

    class Client:
        def __init__(self, *responses):
            self.responses = list(responses)

        def get_base_model(self):
            return "thinkingmachines/Inkling-Small"

        def sample(self, _prompt, **_options):
            value = self.responses.pop(0)
            result = SimpleNamespace(sequences=[SimpleNamespace(tokens=value)])
            return SimpleNamespace(result=lambda timeout=None: result)

    def setUp(self):
        self.state = tempfile.TemporaryDirectory()
        self.environment = mock.patch.dict(
            os.environ, {"XDG_STATE_HOME": self.state.name})
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.state.cleanup()

    def model(self, *responses, **options):
        renderer = self.Renderer()
        model = tinker_model.Tinker(
            Path.cwd(), client=self.Client(*responses), renderer=renderer,
            **options)
        return model, renderer

    def test_turns_preserve_full_cookbook_state_and_context_usage(self):
        response = {"role": "assistant", "content": [
            {"type": "thinking", "thinking": "private"},
            {"type": "text", "text": "answer"},
        ]}
        model, renderer = self.model(response, model="fake", effort="high")
        streamed = []
        reply = model.complete(eko.SYSTEM, conversation("hello")[0], streamed.append)

        self.assertEqual(eko.message_text(reply), "answer")
        self.assertEqual(streamed, ["answer"])
        self.assertEqual(model.messages[-1], response)
        self.assertEqual(model.context_used, 20)
        self.assertEqual(renderer.calls[0][1], {"effort": .9})
        state = json.loads(model.state_file.read_text())
        self.assertEqual(state["messages"], model.messages)
        self.assertEqual(state["version"], tinker_model.SESSION_VERSION)
        self.assertEqual(model.state_file.stat().st_mode & 0o777, 0o600)

    def test_saved_session_resumes(self):
        original, _ = self.model(
            {"role": "assistant", "content": "first"}, model="fake")
        original.complete(eko.SYSTEM, conversation("one")[0], lambda _: None)
        resumed = tinker_model.Tinker(
            Path.cwd(), "fake", session_id=original.session_id, resume=True,
            client=self.Client({"role": "assistant", "content": "second"}),
            renderer=self.Renderer())

        reply = resumed.complete(eko.SYSTEM, conversation("two")[0], lambda _: None)

        self.assertEqual(eko.message_text(reply), "second")
        self.assertEqual(resumed.messages[-4], {"role": "user", "content": "one"})

    def test_resume_rejects_changed_context(self):
        original, _ = self.model(
            {"role": "assistant", "content": "first"}, model="fake")
        original.complete(eko.SYSTEM, conversation("one")[0], lambda _: None)
        resumed = tinker_model.Tinker(
            Path.cwd(), "fake", session_id=original.session_id, resume=True,
            client=self.Client(), renderer=self.Renderer())

        with self.assertRaisesRegex(ValueError, "system context"):
            resumed.complete("different", conversation("two")[0], lambda _: None)

    def test_image_input_fails_explicitly(self):
        model, _ = self.model(model="fake")
        message = eko.Message("user", (eko.Image("image/png", b"image"),))
        with self.assertRaisesRegex(ValueError, "does not support image"):
            model.complete(eko.SYSTEM, message, lambda _: None)

    def test_auth_requires_tinker_key(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(SystemExit, "TINKER_API_KEY"):
                tinker_model.ensure_auth()
        with mock.patch.dict(os.environ, {"TINKER_API_KEY": "secret"}):
            tinker_model.ensure_auth()


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

    def send(self, message, write):
        text = eko.message_text(message).split("\n", 1)[-1]
        if text != "long code":
            return super().send(message, write)
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
    ui = host.UI()
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
