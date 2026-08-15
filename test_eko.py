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
import host
import models
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


class PythonCall:
    def __init__(self, code: str) -> None:
        self.code = code


class FakeModel:
    def __init__(self, replies) -> None:
        self.replies = replies
        self.messages: list[str] = []
        self.system = None
        self.cancelled = threading.Event()
        self.context_used = 0
        self.resets = 0
        self.max_turns = None

    def start(self, system):
        self.system = system

    def send(self, message, write, python=lambda _code: None):
        text = "\n".join(
            line for line in eko.message_text(message).splitlines()
            if not line.startswith("<input source=") and line != "</input>"
        )
        self.messages.append(text)
        reply = self.replies(text, self.cancelled)
        while isinstance(reply, PythonCall):
            result = python(reply.code)
            observation = "".join(
                part.text for part in models.tool_result_parts(result)
                if isinstance(part, eko.Text))
            self.messages.append(observation)
            reply = self.replies(observation, self.cancelled)
        write(reply)
        return eko.Message("assistant", (eko.Text(reply),))

    def complete(self, system, message, write, python=lambda _code: None,
                 max_turns=0):
        self.max_turns = max_turns
        self.start(system)
        return self.send(message, write, python)

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

    def test_system_distinguishes_user_steering_from_async_observations(self):
        self.assertIn(
            "Express your reasoning in ordinary assistant text throughout the task.",
            eko.SYSTEM)
        self.assertIn("what you learned from prior results", eko.SYSTEM)
        self.assertIn("what you currently believe", eko.SYSTEM)
        self.assertIn("why the next action will help", eko.SYSTEM)
        self.assertIn('Source "user-terminal" is a controlling user', eko.SYSTEM)
        self.assertIn('before continuing the current plan', eko.SYSTEM)
        self.assertIn('Process and harness inputs are observations', eko.SYSTEM)
        self.assertIn('never demote\na user-terminal instruction', eko.SYSTEM)
        self.assertIn('<done/> and a Python tool call are mutually exclusive',
                      eko.NORMAL_MODE)
        self.assertNotIn('Python block', eko.NORMAL_MODE)

    def test_mcp_python_tool_raises_claude_text_result_limit(self):
        tool = models.mcp_python_tool()
        self.assertEqual(tool["name"], "python")
        self.assertEqual(tool["_meta"]["anthropic/maxResultSizeChars"], 500_000)

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
        first = eko.limit_input(eko.Input(eko.TERMINAL, (eko.Text("a" * 12),)), 4)
        second = eko.limit_input(eko.Input("python", (eko.Text("b" * 12),)), 4)

        self.assertEqual(first.content[0], eko.Text("aa"))
        self.assertEqual(first.content[-1], eko.Text("aa"))
        self.assertEqual(second.content[0], eko.Text("bb"))
        self.assertEqual(second.content[-1], eko.Text("bb"))

    def test_model_content_uses_canonical_attributed_elements(self):
        message = eko.user_message((
            eko.Input(eko.TERMINAL, (eko.Text("hello"),)),
            eko.Input("python", (eko.Text("output"),), 1),
        ))
        content = host._claude_content(message)
        self.assertEqual(content, [
            {"type": "text", "text": '<input source="user-terminal">\n'},
            {"type": "text", "text": "hello"},
            {"type": "text", "text": "\n</input>"},
            {"type": "text", "text":
             "\n\n<input source=\"python\" exit=\"1\">\n"},
            {"type": "text", "text": "output"},
            {"type": "text", "text": "\n</input>"},
        ])

    def test_native_tool_result_preserves_sources_order_and_images(self):
        image = eko.Image("image/png", b"png", "frame.png")
        result = eko.Result("done\n", 0, 1.2, (
            eko.Input("process-123", (eko.Text("frame"), image)),
            eko.Input(eko.TERMINAL, (eko.Text("stop"),)),
        ))

        parts = models.tool_result_parts(result)
        self.assertEqual(parts, (
            eko.Text("done\n"),
            eko.Text("\n\n"),
            eko.Text('<input source="process-123">\n'),
            eko.Text("frame"), image, eko.Text("\n</input>"),
            eko.Text('\n\n<input source="user-terminal">\n'),
            eko.Text("stop"), eko.Text("\n</input>"),
        ))
        mcp = models.mcp_tool_content(result)
        self.assertEqual([part["type"] for part in mcp],
                         ["text", "text", "text", "text", "image",
                          "text", "text", "text", "text"])
        self.assertEqual(mcp[4]["mimeType"], "image/png")

    def test_async_sources_after_user_terminal_preserve_exact_order(self):
        result = eko.Result("ok", 0, .1, (
            eko.Input(eko.TERMINAL, (eko.Text("do this instead"),)),
            eko.Input(eko.HARNESS, (eko.Text("Context is 90% full."),)),
            eko.Input("process-404", (eko.Text("worker completed"),)),
        ))
        text = "".join(part.text for part in models.tool_result_parts(result)
                       if isinstance(part, eko.Text))
        self.assertLess(text.index('source="user-terminal"'),
                        text.index('source="harness"'))
        self.assertLess(text.index('source="harness"'),
                        text.index('source="process-404"'))

    def test_failed_native_tool_result_matches_command_convention(self):
        result = eko.Result("traceback\n", 3, .1)
        self.assertEqual(models.tool_result_parts(result),
                         (eko.Text("Exit code 3\ntraceback\n"),))

    def test_python_result_is_bounded_before_queued_inputs(self):
        output = "a" * (eko.MAX_INPUT_TEXT + 100)
        bounded = eko.limit_text(output)
        self.assertIn("100 characters omitted", bounded)
        result = eko.Result(bounded, 0, .1, (
            eko.Input("process-7", (eko.Text("important"),)),
        ))
        text = "".join(part.text for part in models.tool_result_parts(result)
                       if isinstance(part, eko.Text))
        self.assertTrue(text.endswith(
            '<input source="process-7">\nimportant\n</input>'))

    def test_tinker_user_and_tool_messages_keep_native_images(self):
        image = eko.Image("image/png", b"png", "frame.png")
        user = models._message(eko.Message(
            "user", (eko.Text("before"), image, eko.Text("after"))))
        tool = models.tinker_tool_content(eko.Result(
            "done", 0, .1, (eko.Input("process-1", (image,)),)))

        self.assertEqual([part["type"] for part in user["content"]],
                         ["text", "image", "text"])
        self.assertEqual([part["type"] for part in tool],
                         ["text", "image", "text"])
        self.assertIn('<input source="process-1">', tool[0]["text"])
        self.assertEqual(tool[2]["text"], "\n</input>")
        self.assertTrue(user["content"][1]["image"].startswith(
            "data:image/png;base64,"))

    def test_tinker_coalesces_one_text_tool_result_into_one_block(self):
        result = eko.Result("done\n", 0, .1, (
            eko.Input(eko.TERMINAL, (eko.Text("do this instead"),)),
            eko.Input(eko.HARNESS, (eko.Text("Context is 90% full."),)),
            eko.Input("process-7", (eko.Text("worker completed"),)),
        ))
        content = models.tinker_tool_content(result)
        self.assertEqual(len(content), 1)
        self.assertEqual(content[0]["type"], "text")
        self.assertIn('<input source="user-terminal">', content[0]["text"])
        self.assertIn('<input source="harness">', content[0]["text"])
        self.assertIn('<input source="process-7">', content[0]["text"])


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
                with mock.patch.object(host, "Claude", return_value=model):
                    host._model_client(server, Path.cwd(), "claude-fake", "low")
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
                    connection, Path.cwd(), "claude-fake", "low")

            with mock.patch.object(host, "Claude", return_value=model):
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

    def test_connection_handshake_selects_model_and_session(self):
        with tempfile.TemporaryDirectory() as directory:
            endpoint = Path(directory) / "model.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(endpoint))
            listener.listen()
            model = FakeModel(lambda text, _cancelled: text.upper())
            session_id = "01234567-89ab-cdef-0123-456789abcdef"

            def serve():
                connection, _ = listener.accept()
                host._model_client(
                    connection, Path.cwd(), "claude-opus-5", "high")

            with mock.patch.object(host, "Tinker", return_value=model) as factory:
                thread = threading.Thread(target=serve, daemon=True)
                thread.start()
                remote = eko.Model(
                    endpoint, "thinkingmachines/Inkling-Small", "low",
                    session_id, True, 1)
                remote.start(eko.SYSTEM)
                reply = remote.send(conversation("hello")[0], lambda _: None)
                remote.close()
                thread.join(2)
            listener.close()

        factory.assert_called_once_with(
            Path.cwd(), "thinkingmachines/Inkling-Small", "low",
            session_id, True)
        self.assertEqual(reply, eko.Message("assistant", (eko.Text("HELLO"),)))
        self.assertEqual(model.max_turns, 1)
        self.assertFalse(model.cancelled.is_set())

    def test_message_socket_round_trips_images(self):
        message = eko.Message("user", (
            eko.Text("look"), eko.Image("image/png", b"image", "x.png")))
        self.assertEqual(eko.decode_message(eko.encode_message(message)), message)

    def test_process_events_round_trip(self):
        events = (
            eko.Event("state", "thinking"),
            eko.Event("input", eko.Message("user", (
                eko.Text('<input source="user-terminal">\n'), eko.Text("hello"),
                eko.Text("\n</input>")))),
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
            '<input source="user-terminal">\nhello\n</input>')
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
        self.assertEqual(eko.message_text(agent.messages[0]),
                         '<input source="user-terminal">\nhello\n</input>')
        self.assertEqual(eko.message_text(agent.messages[1]), "finished<done/>")
        self.assertIn("You are Eko.", agent.model.system)
        self.stop(agent)

    def test_feral_agent_starts_without_a_prompt(self):
        def reply(_message, cancelled):
            return "stopped" if cancelled.is_set() else PythonCall(
                "import time\ntime.sleep(60)\n")

        model = FakeModel(reply)
        agent = eko.Eko(Path.cwd(), model, feral=True)
        agent.start()
        wait_until(lambda: agent.state == "running Python")

        self.assertNotIn("Feral mode", model.system)
        self.assertEqual(model.messages[0], "Begin.")
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

    def test_python_completion_returns_to_thinking_state(self):
        def replies(message, _cancelled):
            if message == "run":
                return PythonCall("print('done')")
            return "finished<done/>"

        agent, ui = self.agent(replies)
        agent.start("run")
        wait_until(lambda: agent.state == "idle")
        states = [event.value for event in ui.events if event.type == "state"]
        running = states.index("running Python")
        self.assertEqual(states[running + 1], "thinking")
        self.stop(agent)

    def test_interrupting_python_also_cancels_the_model_tool_loop(self):
        calls = []

        def replies(message, cancelled):
            if message == "run":
                return PythonCall(
                    "import time\nwhile True: time.sleep(.1)\n")
            calls.append(message)
            if cancelled.is_set():
                raise InterruptedError
            return PythonCall("print('should not run')")

        agent, ui = self.agent(replies)
        agent.start("run")
        wait_until(lambda: agent.state == "running Python")
        agent.interrupt()
        wait_until(lambda: agent.state == "idle")

        states = [event.value for event in ui.events if event.type == "state"]
        running = states.index("running Python")
        self.assertNotIn("thinking", states[running + 1:])
        self.assertEqual(len(calls), 1)
        self.assertIn("Interrupted", calls[0])
        self.assertEqual(ui.errors, ["Interrupted"])
        self.stop(agent)

    def test_input_during_interrupted_python_starts_the_next_clean_turn(self):
        def replies(message, cancelled):
            if message == "run":
                return PythonCall(
                    "import time\nwhile True: time.sleep(.1)\n")
            if "Interrupted" in message:
                raise InterruptedError
            if message == "new request":
                cancelled.clear()
                return "recovered<done/>"
            self.fail(f"unexpected model input: {message!r}")

        agent, ui = self.agent(replies)
        agent.start("run")
        wait_until(lambda: agent.state == "running Python")
        agent.send("new request")
        agent.interrupt()
        wait_until(lambda: agent.state == "idle")

        self.assertEqual(agent.model.messages[-1], "new request")
        self.assertEqual(ui.errors, ["Interrupted"])
        responses = [event.value[0] for event in ui.events
                     if event.type == "response" and event.value[1] is None]
        self.assertIn("recovered<done/>", responses)
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
                return PythonCall("import time\ntime.sleep(.2)\nprint('done')\n")
            self.assertTrue(message.startswith("done\n"))
            self.assertIn('<input source="user-terminal">', message)
            self.assertIn("typed while running", message)
            return "finished<done/>"

        agent, ui = self.agent(replies)
        agent.start("run")
        wait_until(lambda: agent.state == "running Python")
        agent.send("typed while running")
        wait_until(lambda: len(agent.model.messages) == 2)
        wait_until(lambda: agent.state == "idle")
        self.assertEqual(ui.results[0].output, "done\n")
        self.stop(agent)

    def test_inputs_are_emitted_before_model_responses(self):
        agent, ui = self.agent(lambda _message, _cancelled: "<done/>")
        agent.start("hello")
        wait_until(lambda: agent.state == "idle")
        inputs = [event.value for event in ui.events if event.type == "input"]
        self.assertEqual(inputs, [eko.Message("user", (
            eko.Text('<input source="user-terminal">\n'), eko.Text("hello"),
            eko.Text("\n</input>")))])
        self.stop(agent)

    def test_detached_python_can_send_attributed_input(self):
        callback = "background job finished"

        def replies(message, cancelled):
            if message == "start":
                return PythonCall("""
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
""")
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
                return PythonCall("import os; print(os.environ['EKO_AGENT'])\n")
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

    def test_native_python_call_renders_inside_panel(self):
        rendered = self.render(host.python_renderable("print('running')"))
        self.assertIn("python", rendered)
        self.assertIn("print('running')", rendered)
        self.assertIn("╭", rendered)

    def test_py_fence_is_displayed_but_not_executed(self):
        response = "Example:\n```py\nprint('display only')\n```\n"

        stream = io.StringIO()
        Console(file=stream, force_terminal=True, color_system="truecolor",
                width=80).print(host.response_renderable(response))
        rendered = stream.getvalue()
        self.assertIn("display only", rendered)
        self.assertIn(" print('display only')", rendered)

    def test_plain_python_fence_is_displayed_but_not_executed(self):
        response = "Example:\n```python\nprint('display only')\n```\n"

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
        self.assertNotIn("**formatted**", rendered)
        self.assertIn("formatted", rendered)
        self.assertEqual(rendered.count("def answer"), 1)
        self.assertEqual(rendered.count("return 42"), 1)

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


class ModelTests(unittest.TestCase):
    def test_tinker_trajectory_records_exact_tokens_logprobs_and_prompt(self):
        class Chunk:
            length = 2

        class Prompt:
            chunks = [Chunk()]

            def model_dump(self, mode=None):
                self.mode = mode
                return {"chunks": [{"type": "encoded_text", "tokens": [10, 11]}]}

        class Renderer:
            def create_conversation_prefix_with_tools(self, _tools, _system):
                return []

            def build_generation_prompt(self, _messages, **_options):
                return Prompt()

            def get_stop_sequences(self):
                return []

            def parse_response(self, _tokens):
                return {"role": "assistant", "content": "done"}, None

        class Sequence:
            tokens = [20, 21]
            logprobs = [-0.2, -0.1]
            stop_reason = "stop"

        class Future:
            def result(self, timeout=None):
                return type("Response", (), {"sequences": [Sequence()]})()

        class Client:
            def get_base_model(self):
                return "thinkingmachines/Inkling-Small"

            def sample(self, *_args, **_kwargs):
                return Future()

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.jsonl"
            model = models.Tinker(
                Path.cwd(), client=Client(), renderer=Renderer(),
                trajectory_path=path)
            reply = model.complete(
                "system", conversation("start")[0], lambda _text: None,
                lambda _code: self.fail("unexpected Python call"))

            records = [json.loads(line) for line in path.read_text().splitlines()]
            transition, end = records
            self.assertEqual(reply, eko.Message("assistant", (eko.Text("done"),)))
            self.assertEqual(transition["action"]["tokens"], [20, 21])
            self.assertEqual(transition["action"]["logprobs"], [-0.2, -0.1])
            self.assertEqual(transition["observation"]["chunks"][0]["tokens"],
                             [10, 11])
            self.assertEqual(end["reason"], "completed")
            self.assertEqual(end["final_observation"]["chunks"][0]["tokens"],
                             [10, 11])

    def test_claude_process_exit_clears_python_broker_handler(self):
        model = host.Claude(Path.cwd(), "fake", "low")

        def start(_system):
            model.proc = subprocess.Popen(
                [sys.executable, "-c", "import sys; sys.stdin.buffer.readline()"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, bufsize=0)
            model.started = True

        model._start = start
        with self.assertRaisesRegex(
                RuntimeError, "Model process exited without a result"):
            model.complete(eko.SYSTEM, conversation("stop")[0], lambda _: None)
        with model.broker.lock:
            self.assertIsNone(model.broker.handler)
        model.close()

    def test_claude_max_turn_boundary_flushes_process_before_returning(self):
        model = host.Claude(Path.cwd(), "fake", "low")
        payload = json.dumps({
            "type": "result", "subtype": "error_max_turns",
            "is_error": True, "usage": {"input_tokens": 3},
        }) + "\n"

        def start(_system, _max_turns):
            script = ("import sys; sys.stdin.buffer.readline(); "
                      f"sys.stdout.buffer.write({payload.encode()!r}); "
                      "sys.stdout.buffer.flush(); sys.stdin.buffer.read()")
            model.proc = subprocess.Popen(
                [sys.executable, "-c", script], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            model.started = True

        model._start = start
        reply = model.complete(
            eko.SYSTEM, conversation("one turn")[0], lambda _: None,
            max_turns=1)

        self.assertEqual(reply, eko.Message("assistant", ()))
        self.assertIsNone(model.proc)
        self.assertEqual(model.context_used, 3)
        model.close()

    def test_agent_command_forwards_python_timeout(self):
        command = host._agent_command(
            Path.cwd(), Path("/tmp/runtime"), sandbox=False, feral=False,
            name="Eko", python_timeout=12.5,
        )

        index = command.index("--python-timeout")
        self.assertEqual(command[index + 1], "12.5")

    def test_agent_command_forwards_max_turns(self):
        command = host._agent_command(
            Path.cwd(), Path("/tmp/runtime"), sandbox=False, feral=False,
            name="Eko", max_turns=1,
        )

        index = command.index("--max-turns")
        self.assertEqual(command[index + 1], "1")

    def test_agent_command_forwards_clean_workspace_flag(self):
        command = host._agent_command(
            Path.cwd(), Path("/tmp/runtime"), sandbox=False, feral=False,
            name="Eko", clean_workspace=True,
        )

        self.assertIn("--clean-workspace", command)

    def test_headless_observer_ignores_initial_idle_then_finishes(self):
        observer = host.HeadlessObserver()
        with mock.patch.object(host, "print_event"):
            observer(eko.Event("state", "idle"))
            self.assertFalse(observer.finished.is_set())
            observer(eko.Event("state", "thinking"))
            observer(eko.Event("state", "running Python"))
            observer(eko.Event("state", "idle"))
        self.assertTrue(observer.finished.is_set())

    def test_agent_startup_error_includes_child_stderr(self):
        agent = host.AgentProcess([
            sys.executable, "-c",
            "import sys; print('sandbox failed', file=sys.stderr); sys.exit(1)",
        ])
        self.assertTrue(agent.ready.wait(2))
        agent.proc.wait(2)
        self.assertEqual(agent.startup_error(),
                         "agent did not start:\nsandbox failed")
        agent.stop()

    def test_agent_startup_error_includes_protocol_failure(self):
        agent = host.AgentProcess([
            sys.executable, "-c", "print('not-json', flush=True)",
        ])
        self.assertTrue(agent.ready.wait(2))
        agent.proc.wait(2)
        self.assertIn("Agent connection failed", agent.startup_error())
        agent.stop()

    def test_host_accepts_new_and_resumed_primary_sessions(self):
        session_id = "12345678-1234-5678-1234-567812345678"
        for option, expected_resume in (
            ("--session-id", False), ("--resume", True)
        ):
            with (
                mock.patch.object(sys, "argv", ["eko", option, session_id]),
                mock.patch.object(host, "run") as run,
            ):
                host.main()
            self.assertEqual(run.call_args.kwargs["session_id"], session_id)
            self.assertEqual(run.call_args.kwargs["resume"], expected_resume)

    def test_model_server_assigns_persisted_session_only_to_primary(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session_id = "12345678-1234-5678-1234-567812345678"
            server = host.ModelServer(
                root / "model.sock", root, "fake", "low", session_id, True
            )
            observed = []

            def model_client(connection, _cwd, _model, _effort,
                             assigned=None, resume=False):
                observed.append((assigned, resume))
                connection.close()

            with mock.patch.object(host, "_model_client", side_effect=model_client):
                server.start()
                for _ in range(2):
                    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    client.connect(str(root / "model.sock"))
                    client.close()
                wait_until(lambda: len(observed) == 2)
                server.close()

        self.assertCountEqual(observed, [(session_id, True), (None, False)])

    def test_model_server_passes_injected_tinker_rollout_context(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            client_object = object()
            trajectory = root / "trajectory.jsonl"
            state = root / "state"
            server = host.ModelServer(
                root / "model.sock", root / "workspace", "model", "high",
                tinker_client=client_object, trajectory_path=trajectory,
                state_root=state)
            observed = []

            def model_client(connection, *_args, **kwargs):
                observed.append(kwargs)
                connection.close()

            with mock.patch.object(host, "_model_client", side_effect=model_client):
                server.start()
                client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client.connect(str(root / "model.sock")); client.close()
                wait_until(lambda: len(observed) == 1)
                server.close()

        self.assertEqual(observed, [{
            "tinker_client": client_object,
            "trajectory_path": trajectory,
            "state_root": state,
        }])

    def test_host_accepts_an_explicit_world_socket(self):
        with (
            mock.patch.object(
                sys, "argv", ["eko", "--world-socket", "/tmp/world.sock"]
            ),
            mock.patch.object(host, "run") as run,
        ):
            host.main()

        self.assertEqual(run.call_args.kwargs["world_socket"], Path("/tmp/world.sock"))

    def test_host_accepts_an_upstream_model_socket(self):
        with (
            mock.patch.object(
                sys, "argv", ["eko", "--upstream-model-socket", "/tmp/model.sock"]
            ),
            mock.patch.object(host, "run") as run,
        ):
            host.main()

        self.assertEqual(run.call_args.kwargs["upstream_model_socket"],
                         Path("/tmp/model.sock"))

    def test_feral_host_defaults_to_an_empty_temporary_workspace(self):
        observed = {}

        def inspect_workspace(cwd, *_args, **_kwargs):
            observed["cwd"] = cwd
            observed["exists"] = cwd.is_dir()
            observed["contents"] = list(cwd.iterdir())

        with (
            mock.patch.object(sys, "argv", ["eko", "--feral"]),
            mock.patch.object(host, "run", side_effect=inspect_workspace),
        ):
            host.main()

        self.assertTrue(observed["exists"])
        self.assertEqual(observed["contents"], [])
        self.assertFalse(observed["cwd"].exists())

    def test_feral_host_preserves_an_explicit_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            workspace = Path(directory)
            with (
                mock.patch.object(
                    sys, "argv", ["eko", "--feral", "--cwd", directory]
                ),
                mock.patch.object(host, "run") as run,
            ):
                host.main()

        self.assertEqual(run.call_args.args[0], workspace)

    def test_identity_is_configurable_and_location_is_neutral(self):
        prompt = eko.SYSTEM.format(name="Moa", folder="/workspace", mode="")
        self.assertTrue(prompt.startswith("You are Moa.\nYou are in /workspace.\n"))
        self.assertEqual(eko.NAME, "Eko")
        model = host.Claude(Path("/host/private"), "fake", "low")
        self.assertEqual(model.cwd, Path("/host/private"))

    def test_host_launches_the_provider_neutral_core(self):
        with tempfile.TemporaryDirectory() as directory:
            command = host._agent_command(
                Path.cwd(), Path(directory), sandbox=False,
                feral=False, name="Moa")

        self.assertEqual(Path(command[1]).resolve(), Path(eko.__file__).resolve())
        self.assertIn("--model-socket", command)
        self.assertIn("--session-socket", command)

    def test_sandbox_maps_only_external_package_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            environment = root / "environment"
            base = root / "base"
            internal = environment / "lib/python3.12/site-packages"
            external = root / "overlay/lib/python3.12/site-packages"
            project = root / "project"
            for path in (internal, external, project):
                path.mkdir(parents=True)
            with mock.patch.object(
                sys, "path", [str(internal), str(external), str(project)]
            ):
                mounts, paths = host._package_mounts(environment, base)

        self.assertEqual(mounts, [
            "--ro-bind", str(external), "/opt/eko-packages/0"
        ])
        self.assertEqual(paths, [
            "/opt/eko/lib/python3.12/site-packages",
            "/opt/eko-packages/0",
        ])

    def test_host_exports_model_socket_to_child_agents(self):
        observed = {}

        class Server:
            def __init__(self, path, *_args):
                observed["socket"] = str(path)

            def start(self):
                pass

            def close(self):
                pass

        class Agent:
            def __init__(self, _command, env):
                observed["environment"] = env
                self.ready = threading.Event()
                self.ready.set()
                self.proc = mock.Mock()
                self.proc.poll.side_effect = [None, 0]
                self.observer = None

            def stop(self):
                pass

        with (
            mock.patch.object(host, "ensure_claude_auth"),
            mock.patch.object(host, "ModelServer", Server),
            mock.patch.object(host, "AgentProcess", Agent),
        ):
            host.run(Path.cwd(), None, model="claude-fake", effort="low",
                     feral=False, name="Eko", headless=True, sandbox=False)

        self.assertEqual(observed["environment"]["EKO_MODEL"],
                         observed["socket"])

    def test_sandbox_exposes_default_world_socket_path(self):
        with tempfile.TemporaryDirectory() as directory:
            command = host._agent_command(
                Path.cwd(), Path(directory), sandbox=True,
                feral=False, name="Moa")

        index = command.index("EKO_WORLD")
        self.assertEqual(command[index - 1], "--setenv")
        self.assertEqual(command[index + 1], "/run/eko/world.sock")

    def test_sandbox_executes_resolved_interpreter_not_relocated_symlink(self):
        with tempfile.TemporaryDirectory() as directory:
            command = host._agent_command(
                Path.cwd(), Path(directory), sandbox=True,
                feral=False, name="Moa")

        source_index = len(command) - 1 - command[::-1].index("/run/eko.py")
        self.assertEqual(command[source_index - 1], str(Path(sys.executable).resolve()))
        self.assertNotIn("/opt/eko/bin/python", command)
        pythonpath = command.index("PYTHONPATH")
        self.assertEqual(command[pythonpath - 1], "--setenv")
        paths = command[pythonpath + 1].split(os.pathsep)
        self.assertTrue(all(path.startswith("/opt/eko") for path in paths))
        self.assertTrue(any(path.startswith("/opt/eko/") for path in paths))

    def test_world_relay_forwards_stream_without_interpreting_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            upstream_path = root / "upstream.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(upstream_path))
            listener.listen()

            def echo():
                connection, _ = listener.accept()
                with connection:
                    while data := connection.recv(65536):
                        connection.sendall(data)

            server = threading.Thread(target=echo)
            server.start()
            relay = host.WorldRelay(root / "world.sock", upstream_path)
            relay.start()
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
                    client.connect(str(root / "world.sock"))
                    request = b'{"jsonrpc":"2.0","id":1,"method":"rpc.discover"}\n'
                    client.sendall(request)
                    self.assertEqual(client.recv(len(request)), request)
            finally:
                relay.close()
                listener.close()
                server.join(2)
            self.assertFalse((root / "world.sock").exists())

    @unittest.skipUnless(shutil.which("bwrap"), "Bubblewrap is not installed")
    def test_sandbox_shutdown_removes_socket_and_detached_descendants(self):
        """Namespace teardown must not leave a live daemon or session socket."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            workspace = root / "workspace"
            runtime = root / "runtime"
            workspace.mkdir()
            runtime.mkdir()
            endpoint = runtime / "model.sock"
            listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            listener.bind(str(endpoint))
            listener.listen()

            code = '''\
import os, subprocess, sys, time
graceful = """import pathlib, signal, sys, time
def stop(_signal, _frame):
    pathlib.Path('/workspace/terminated').write_text('graceful')
    sys.exit(0)
signal.signal(signal.SIGTERM, stop)
pathlib.Path('/workspace/graceful-ready').touch()  # GRACEFUL_READY
time.sleep(300)
"""
stubborn = """import pathlib, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path('/workspace/stubborn-ready').touch()
time.sleep(300)  # STUBBORN_DAEMON
"""
double_fork = """import subprocess, sys
child = ("import pathlib, signal, time; "
         "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
         "pathlib.Path('/workspace/double-ready').touch(); "
         "time.sleep(300)  # DOUBLE_FORK_DAEMON")
subprocess.Popen([sys.executable, '-c', child], start_new_session=True,
                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL)
"""
processes = [
    subprocess.Popen([sys.executable, "-c", graceful], start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL),
    subprocess.Popen([sys.executable, "-c", stubborn], start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL),
    subprocess.Popen([sys.executable, "-c", double_fork], start_new_session=True,
                     stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                     stderr=subprocess.DEVNULL),
]

child_env = {key: value for key, value in os.environ.items()
             if key != 'EKO_SESSION'}
child_input, child_hold = os.pipe()
processes.append(subprocess.Popen(
    [sys.executable, os.environ['EKO_AGENT'], '--cwd', '/workspace',
     '--name', 'CLEANUP_CHILD', '--session-socket', '/tmp/cleanup-child.sock'],
    start_new_session=True, stdin=child_input, stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL, env=child_env, pass_fds=(child_hold,)))
os.close(child_input)
os.close(child_hold)

nested = """import pathlib, signal, time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
pathlib.Path('/workspace/nested-ready').touch()
time.sleep(300)  # NESTED_SANDBOX_DAEMON
"""
processes.append(subprocess.Popen([
    'bwrap', '--new-session', '--as-pid-1', '--unshare-user', '--unshare-pid',
    '--ro-bind', '/', '/', '--bind', '/workspace', '/workspace',
    '--proc', '/proc', '/usr/bin/python3', '-c', nested],
    start_new_session=True, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL))

deadline = time.monotonic() + 5
while not os.path.exists('/tmp/cleanup-child.sock') and time.monotonic() < deadline:
    time.sleep(.02)
if os.path.exists('/tmp/cleanup-child.sock'):
    open('/workspace/child-ready', 'w').close()
print("DAEMONS", *(process.pid for process in processes))
'''

            stopping = threading.Event()
            clients = []

            def model_client(connection):
                with connection, connection.makefile("rb") as reader:
                    json.loads(reader.readline())  # system prompt
                    line = reader.readline()
                    if not line:  # An idle child agent.
                        return
                    json.loads(line)  # terminal input
                    connection.sendall((json.dumps({
                        "tool_call": {"id": "cleanup", "code": code},
                    }) + "\n").encode())
                    json.loads(reader.readline())  # Python result
                    response = eko.Message("assistant", (eko.Text("<done/>"),))
                    connection.sendall((json.dumps({
                        "message": eko.encode_message(response),
                    }) + "\n").encode())
                    reader.readline()

            def model_service():
                listener.settimeout(.1)
                while not stopping.is_set():
                    try:
                        connection, _ = listener.accept()
                    except socket.timeout:
                        continue
                    except OSError:
                        return
                    client = threading.Thread(
                        target=model_client, args=(connection,))
                    client.start()
                    clients.append(client)

            server = threading.Thread(target=model_service)
            server.start()
            agent = host.AgentProcess(host._agent_command(
                workspace, runtime, sandbox=True, feral=False, name="Eko"))
            events = []
            agent.observer = events.append
            try:
                self.assertTrue(agent.ready.wait(10))
                self.assertIsNone(agent.proc.poll())
                agent.send("start a detached process")
                wait_until(lambda: any(event.type == "result" for event in events),
                           timeout=10)
                result = next(event.value for event in events
                              if event.type == "result")
                ready = ("graceful-ready", "stubborn-ready", "double-ready",
                         "child-ready", "nested-ready")
                deadline = time.monotonic() + 10
                while (not all((workspace / name).exists() for name in ready)
                       and time.monotonic() < deadline):
                    time.sleep(.02)
                missing = [name for name in ready
                           if not (workspace / name).exists()]
                self.assertEqual(missing, [], result.output)

                def descendants(pid):
                    found = []
                    pending = [pid]
                    while pending:
                        parent = pending.pop()
                        path = Path(f"/proc/{parent}/task/{parent}/children")
                        try:
                            children = [int(item) for item in path.read_text().split()]
                        except OSError:
                            children = []
                        found.extend(children)
                        pending.extend(children)
                    return found

                descendants_before_stop = descendants(agent.proc.pid)
                commands = b"\n".join(
                    Path(f"/proc/{pid}/cmdline").read_bytes()
                    for pid in descendants_before_stop
                    if Path(f"/proc/{pid}/cmdline").exists())
                for marker in (b"GRACEFUL_READY", b"STUBBORN_DAEMON",
                               b"DOUBLE_FORK_DAEMON", b"CLEANUP_CHILD",
                               b"NESTED_SANDBOX_DAEMON"):
                    self.assertIn(marker, commands)
                self.assertTrue((runtime / "session.sock").exists())

                agent.stop()

                wait_until(lambda: all(not Path(f"/proc/{pid}").exists()
                                       for pid in descendants_before_stop),
                           timeout=5)
                self.assertEqual((workspace / "terminated").read_text(),
                                 "graceful")
                self.assertFalse((runtime / "session.sock").exists())
                self.assertEqual(agent.proc.returncode, 0)
            finally:
                agent.stop()
                stopping.set()
                listener.close()
                server.join(5)
                for client in clients:
                    client.join(5)

    def test_close_tolerates_concurrent_interrupt_clearing_process(self):
        model = host.Claude(Path.cwd(), "fake", "low")
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

    def test_interrupt_closes_claude_process_streams(self):
        model = host.Claude(Path.cwd(), "fake", "low")
        model.proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(30)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True)
        process = model.proc

        model.interrupt()

        self.assertIsNone(model.proc)
        self.assertIsNotNone(process.returncode)
        self.assertTrue(process.stdin.closed)
        self.assertTrue(process.stdout.closed)

    def test_exact_empty_text_resume_error_repairs_only_its_assistant_block(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory) / "projects" / "project"
            project.mkdir(parents=True)
            model = host.Claude(Path.cwd(), "fake", "low")
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

            def start(_system):
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
                reply = model.complete(eko.SYSTEM, conversation("continue")[0], lambda _: None)
                self.assertEqual(eko.message_text(reply), "recovered")
                model.close()

            lines = agent.read_text().splitlines(keepends=True)
            self.assertEqual(lines[0], unchanged)
            content = json.loads(lines[1])["message"]["content"]
            self.assertEqual(content[0]["thinking"], "")
            self.assertEqual(content[1]["text"], " ")
            self.assertEqual(len(starts), 2)

    def test_failed_resume_can_recover_same_session(self):
        model = host.Claude(Path.cwd(), "fake", "low")
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

        def start(_system):
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
        reply = model.complete(eko.SYSTEM, conversation("continue")[0], lambda _: None)
        self.assertEqual(eko.message_text(reply), "recovered")
        self.assertEqual(len(starts), 2)
        self.assertEqual(model.session_id, session_id)
        model.close()

    def test_failed_resume_retries_without_resetting_session(self):
        model = host.Claude(Path.cwd(), "fake", "low")
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

        def start(_system):
            starts.append(True)
            model.proc = subprocess.Popen(
                [sys.executable, "-c", script], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            model.started = True

        model._start = start
        with self.assertRaisesRegex(RuntimeError, "context was not reset"):
            model.complete(eko.SYSTEM, conversation("orphaned output")[0], lambda _: None)
        self.assertEqual(len(starts), models.RESUME_RETRIES + 1)
        self.assertTrue(model.started)
        self.assertEqual(model.session_id, session_id)

    def test_multiple_events_buffered_in_one_write_do_not_stall(self):
        model = host.Claude(Path.cwd(), "fake", "low")
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

        def start(_system):
            model.proc = subprocess.Popen(
                [sys.executable, "-c", script], stdin=subprocess.PIPE,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, bufsize=0)
            model.started = True

        model._start = start
        streamed = []
        reply = model.complete(eko.SYSTEM, conversation("hi")[0], streamed.append)
        self.assertEqual(eko.message_text(reply), "hello")
        self.assertEqual(streamed, ["hello"])
        model.close()

    def test_interrupt_does_not_race_with_stdout_reader(self):
        model = host.Claude(Path.cwd(), "fake", "low")
        proc = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(10)"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, start_new_session=True)
        model.proc = proc
        model.started = True
        errors = []

        def complete():
            try:
                model.complete(eko.SYSTEM, conversation("hello")[0], lambda text: None)
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
            self.assertGreaterEqual(max(live_numbers), 90)
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
            self.assertIn("╯\n\nExit 0 ·", screen)
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
            return PythonCall("print('VISIBLE_RESULT')\n")
        if message in {"complete long code", "long code"}:
            code = "".join(f"print({number})\n" for number in range(100))
            return PythonCall(code)
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
