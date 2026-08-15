"""Model providers for Eko's host."""

from __future__ import annotations

import base64
import json
import shutil
import signal
import socket
import subprocess
import os
import sys
import tempfile
import threading
import uuid
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Callable

import eko as core

MAX_TOKENS = 8192
RESUME_RETRIES = 3
DEFAULT_MODEL = "thinkingmachines/Inkling-Small"
SESSION_VERSION = 3
EFFORT = {"low": .2, "medium": .7, "high": .9, "xhigh": .99, "max": .99}
PYTHON_TOOL = {
    "name": "python",
    "description": "Run Python in the agent's workspace and return its result.",
    "parameters": {
        "type": "object",
        "properties": {"code": {"type": "string"}},
        "required": ["code"],
        "additionalProperties": False,
    },
}


def mcp_python_tool() -> dict:
    """Describe Python to MCP clients, including Claude's text result budget."""
    return {
        "name": "python", "description": PYTHON_TOOL["description"],
        "inputSchema": PYTHON_TOOL["parameters"],
        "_meta": {"anthropic/maxResultSizeChars": 500_000},
    }


def tool_result_parts(result: core.Result) -> tuple[core.Content, ...]:
    """Build the canonical ordered, attributed Python observation stream."""
    output = result.output or "(no output)"
    if result.returncode:
        output = f"Exit code {result.returncode}\n{output}"
    parts: list[core.Content] = [core.Text(output)]
    if result.inputs:
        parts.append(core.Text("\n\n"))
        parts.extend(core.attributed_content(result.inputs))
    return tuple(parts)


def tinker_content(parts: tuple[core.Content, ...]) -> list[dict]:
    """Encode provider-neutral parts for Tinker's multimodal renderer."""
    content: list[dict] = []
    for part in parts:
        if isinstance(part, core.Text):
            if content and content[-1]["type"] == "text":
                content[-1]["text"] += part.text
            else:
                content.append({"type": "text", "text": part.text})
        else:
            data = base64.b64encode(part.data).decode()
            content.append({"type": "image", "image":
                            f"data:{part.media_type};base64,{data}"})
    return content


def tinker_tool_content(result: core.Result) -> list[dict]:
    """Encode canonical result parts for Tinker's multimodal renderer."""
    return tinker_content(tool_result_parts(result))


def mcp_tool_content(result: core.Result) -> list[dict]:
    """Encode canonical result parts as MCP text and image blocks."""
    content: list[dict] = []
    for part in tool_result_parts(result):
        if isinstance(part, core.Text):
            content.append({"type": "text", "text": part.text})
        else:
            content.append({
                "type": "image", "mimeType": part.media_type,
                "data": base64.b64encode(part.data).decode(),
            })
    return content


def session_file(session_id: str) -> Path:
    root = Path(os.environ.get(
        "XDG_STATE_HOME", Path.home() / ".local" / "state"))
    return root / "eko" / "sessions" / f"{session_id}.json"


def _message(message: core.Message) -> dict:
    if all(isinstance(part, core.Text) for part in message.content):
        content: str | list[dict] = "".join(
            part.text for part in message.content if isinstance(part, core.Text))
    else:
        content = tinker_content(message.content)
    return {"role": message.role, "content": content}


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
                 on_text: Callable[[str], None],
                 on_python: Callable[[str], core.Result],
                 max_turns: int = 0) -> core.Message:
        if message.role != "user":
            raise ValueError("model input must be a user message")
        if self.system is None:
            self.system = system
            _client, renderer = self._connect()
            self.messages = renderer.create_conversation_prefix_with_tools(
                [PYTHON_TOOL], system)
        elif self.system != system:
            raise ValueError("session system context does not match this invocation")
        self.interrupted.clear()
        user = _message(message)
        client, renderer = self._connect()
        options = ({"effort": EFFORT[self.effort]}
                   if self.effort is not None else {})
        messages = [*self.messages, user]
        turn_count = 0
        import tinker
        while True:
            turn_count += 1
            prompt = renderer.build_generation_prompt(messages, **options)
            future = client.sample(
                prompt, num_samples=1,
                sampling_params=tinker.SamplingParams(
                    max_tokens=MAX_TOKENS, stop=renderer.get_stop_sequences()))
            while True:
                if self.interrupted.is_set():
                    raise InterruptedError
                try:
                    response = future.result(timeout=.1)
                    break
                except FutureTimeout:
                    continue
            parsed, _termination = renderer.parse_response(
                response.sequences[0].tokens)
            assistant = dict(parsed)
            calls = assistant.get("tool_calls") or []
            if not calls:
                answer = self._text(assistant)
                on_text(answer)
                messages.append(assistant)
                break
            assistant["tool_calls"] = [
                call.model_dump(mode="json") if hasattr(call, "model_dump") else call
                for call in calls
            ]
            messages.append(assistant)
            for call in calls:
                if call.function.name != "python":
                    raise RuntimeError(f"unsupported tool: {call.function.name}")
                arguments = json.loads(call.function.arguments)
                code = arguments.get("code")
                if not isinstance(code, str):
                    raise RuntimeError("python tool requires string code")
                result = on_python(code)
                messages.append({
                    "role": "tool", "name": "python",
                    "tool_call_id": call.id or "",
                    "content": tinker_tool_content(result),
                })
            if max_turns and turn_count >= max_turns:
                self._save(messages)
                self.messages = messages
                self.context_used = sum(
                    int(chunk.length) for chunk in prompt.chunks)
                return core.Message("assistant", ())
        self._save(messages)
        self.messages = messages
        self.context_used = sum(int(chunk.length) for chunk in prompt.chunks)
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


class PythonBroker:
    """Forward one MCP Python call to the active sandbox executor."""

    def __init__(self) -> None:
        self.directory = tempfile.TemporaryDirectory(prefix="eko-python-")
        self.path = Path(self.directory.name) / "tool.sock"
        self.token = uuid.uuid4().hex
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(str(self.path))
        self.listener.listen()
        self.handler: Callable[[str], core.Result] | None = None
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        while not self.stopping.is_set():
            try:
                connection, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(
                target=self._serve, args=(connection,), daemon=True).start()

    def _serve(self, connection: socket.socket) -> None:
        with connection, connection.makefile("rb") as stream:
            try:
                request = json.loads(stream.readline())
                if request.get("token") != self.token:
                    raise ValueError("invalid tool token")
                code = request.get("code")
                if not isinstance(code, str):
                    raise ValueError("python tool requires string code")
                with self.lock:
                    handler = self.handler
                if handler is None:
                    raise RuntimeError("no active model request")
                result = handler(code)
                response = {
                    "output": result.output, "returncode": result.returncode,
                    "elapsed": result.elapsed,
                    "inputs": [core.encode_input(item) for item in result.inputs],
                }
            except Exception as error:
                response = {"error": str(error)}
            try:
                connection.sendall(
                    (json.dumps(response, separators=(",", ":")) + "\n").encode())
            except OSError:
                # The provider may close its MCP connection while Eko is
                # completing the interrupted tool-result handshake.
                pass

    def close(self) -> None:
        if self.stopping.is_set():
            return
        self.stopping.set()
        self.listener.close()
        self.thread.join(2)
        self.directory.cleanup()


def serve_mcp(endpoint: Path, token: str) -> None:
    """Serve the single Python MCP tool over stdio."""

    def send(value: dict) -> None:
        print(json.dumps(value, separators=(",", ":")), flush=True)

    for line in sys.stdin:
        request = json.loads(line)
        request_id = request.get("id")
        method = request.get("method")
        if method == "initialize":
            result = {
                "protocolVersion": request.get("params", {}).get(
                    "protocolVersion", "2025-06-18"),
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "eko", "version": "1"},
            }
        elif method == "tools/list":
            result = {"tools": [mcp_python_tool()]}
        elif method == "tools/call":
            code = request.get("params", {}).get("arguments", {}).get("code")
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
                connection.connect(str(endpoint))
                connection.sendall((json.dumps({
                    "token": token, "code": code,
                }, separators=(",", ":")) + "\n").encode())
                response = json.loads(connection.makefile("rb").readline())
            if "error" in response:
                result = {"content": [{"type": "text", "text": response["error"]}],
                          "isError": True}
            else:
                value = core.Result(
                    response["output"], response["returncode"], response["elapsed"],
                    tuple(core.decode_encoded_input(item)
                          for item in response.get("inputs", [])))
                result = {"content": mcp_tool_content(value),
                          "isError": value.returncode != 0}
        else:
            if request_id is not None:
                send({"jsonrpc": "2.0", "id": request_id, "error": {
                    "code": -32601, "message": "Method not found"}})
            continue
        if request_id is not None:
            send({"jsonrpc": "2.0", "id": request_id, "result": result})


# ── Claude model connection ───────────────────────────────────────────────────


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
    """A persistent native-tool connection to the LLM through the Claude CLI.

    Stream JSON shares one conversation across Eko turns. Built-ins, settings, hooks,
    plugins, and skills are disabled; one explicitly configured MCP server exposes the
    sandbox-backed Python tool.
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
        self.broker = PythonBroker()
        self.write_lock = threading.Lock()

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

    def _start(self, system: str, max_turns: int = 0) -> None:
        session = (["--resume", self.session_id] if self.started else
                   ["--session-id", self.session_id])
        config = json.dumps({"mcpServers": {"eko": {
            "command": sys.executable,
            "args": [str(Path(__file__).resolve()), "--mcp",
                     str(self.broker.path), self.broker.token],
        }}})
        command = [
            "claude", "-p", "--verbose", "--tools", "",
            "--strict-mcp-config", "--mcp-config", config,
            "--allowedTools", "mcp__eko__python", "--permission-mode", "dontAsk",
            "--setting-sources", "", "--settings", "{}",
            "--disable-slash-commands",
            "--model", self.model, "--effort", self.effort,
            *(["--max-turns", str(max_turns)] if max_turns else []),
            *session,
            "--input-format", "stream-json", "--output-format", "stream-json",
            "--include-partial-messages",
            "--system-prompt", system,
        ]
        self.proc = subprocess.Popen(
            command, cwd=self.cwd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, bufsize=0, start_new_session=True)
        self.started = True

    def _write(self, event: dict) -> None:
        proc = self.proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("Claude session is not running")
        with self.write_lock:
            proc.stdin.write((json.dumps(event) + "\n").encode())
            proc.stdin.flush()

    def _terminate(self, signum: int, grace: float = 2) -> None:
        """Signal the CLI process group and ensure it is collected."""
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                os.killpg(proc.pid, signum)
            except ProcessLookupError:
                pass
            try:
                proc.wait(timeout=grace)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                proc.wait()
        for stream in (proc.stdin, proc.stdout, proc.stderr):
            if stream is not None and not stream.closed:
                stream.close()
        self.proc = None

    def _finish(self) -> None:
        """Close a completed CLI input stream and wait for session persistence."""
        proc = self.proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
                    proc.stdin = None
                proc.wait(timeout=3)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                self._terminate(signal.SIGTERM)
                return
        for stream in (getattr(proc, "stdin", None),
                       getattr(proc, "stdout", None),
                       getattr(proc, "stderr", None)):
            if stream is not None and not stream.closed:
                stream.close()
        if self.proc is proc:
            self.proc = None

    def complete(self, system: str, message: core.Message,
                 on_text: Callable[[str], None],
                 on_python: Callable[[str], core.Result] | None = None,
                 retry_delay: float = .2,
                 retries: int = 0,
                 max_turns: int = 0) -> core.Message:
        """Complete a history using the CLI's internally persisted conversation."""
        if message.role != "user":
            raise ValueError("model input must be a user message")
        self.interrupted.clear()
        resuming = self.started
        if self.proc is None or self.proc.poll() is not None:
            if max_turns:
                self._start(system, max_turns)
            else:
                self._start(system)
        proc = self.proc
        assert proc is not None and proc.stdin and proc.stdout
        event = {"type": "user", "message": {
            "role": "user", "content": _claude_content(message)}}
        self._write(event)
        with self.broker.lock:
            self.broker.handler = on_python

        parts: list[str] = []
        complete = ""
        try:
            while line := proc.stdout.readline():
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
                    if data.get("subtype") == "error_max_turns" and max_turns:
                        usage = data.get("usage") or {}
                        self.context_used = (int(usage["prompt_tokens"])
                                             if usage.get("prompt_tokens") is not None
                                             else sum(int(usage.get(name) or 0)
                                                      for name in (
                                                          "input_tokens",
                                                          "cache_read_input_tokens",
                                                          "cache_creation_input_tokens")))
                        self._finish()
                        return core.Message("assistant", ())
                    if data.get("is_error"):
                        detail = data.get("result") or data.get("error")
                        if resuming:
                            self._terminate(signal.SIGTERM)
                            proc.stdin.close()
                            proc.stdout.close()
                            if ("text content blocks must be non-empty" in str(detail)
                                    and self._repair_session()):
                                return self.complete(
                                    system, message, on_text, on_python,
                                    retry_delay, retries, max_turns)
                            if (retries < RESUME_RETRIES
                                    and not parts and not complete):
                                if self.interrupted.wait(retry_delay):
                                    raise InterruptedError
                                return self.complete(
                                    system, message, on_text, on_python,
                                    min(retry_delay * 2, 5), retries + 1,
                                    max_turns)
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
        finally:
            with self.broker.lock:
                self.broker.handler = None
        if self.interrupted.is_set():
            raise InterruptedError
        raise RuntimeError("Model process exited without a result")

    def close(self) -> None:
        """Give the CLI a brief chance to flush its session, then stop it."""
        if self.proc is None:
            self.broker.close()
            return
        self._finish()
        self.broker.close()

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


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--mcp":
        serve_mcp(Path(sys.argv[2]), sys.argv[3])
    else:
        raise SystemExit("models.py is an internal module")
