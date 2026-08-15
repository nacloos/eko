# Eko

Eko is a coding agent with almost no harness: an LLM, Python, and a persistent
folder.

The agent is [`eko.py`](eko.py). It contains the provider-neutral agent loop,
attributed multimodal inbox, Python executor, and process interface. It has no
terminal, model-provider, or sandbox dependency.

An Eko process requires a host that provides a model conversation over the
`EKO_MODEL` Unix socket. One connection to that socket is one independent model
conversation. The included [`host.py`](host.py) supplies that service through the
Claude Code CLI and adds the terminal interface, headless operation, authentication,
and optional Bubblewrap sandbox.

The whole agent is essentially:

```python
message = None
while True:
    message = message or user()
    if message is None:
        break
    if (response := llm(message)).is_done():
        message = None
    else:
        message = run(response.python(), cwd)
```

Python is its only built-in tool. Through Python, the model can inspect its folder,
use ordinary operating-system facilities, and write its own tools.

## Running Eko

Start the included host and terminal interface:

```bash
uv run --script host.py
uv run --script host.py --cwd ~/projects/my-project "Find and fix a bug"
```

Run the same host without the terminal interface, or place the agent in a
Bubblewrap sandbox:

```bash
uv run --script host.py --headless --cwd ~/projects/my-project "Fix the tests"
uv run --script host.py --sandbox --cwd ~/projects/my-project
```

Python actions time out after 30 seconds by default. Override this for either the
included host or the provider-neutral agent with `--python-timeout SECONDS`.

Feral mode starts immediately and keeps acting autonomously. When `--cwd` is
omitted, it runs in a fresh empty workspace that is deleted when Eko exits. Pass
`--cwd` explicitly when the agent should access a persistent project:

```bash
eko --sandbox --feral --world-socket /path/to/world.sock
eko --sandbox --feral --cwd ~/projects/my-project --world-socket /path/to/world.sock
```

`eko.py` can also be run directly behind another host:

```bash
EKO_MODEL=/path/to/model.sock python eko.py --cwd ~/projects/my-project
```

It reads JSON-line commands from standard input and writes JSON-line events to
standard output. This is how a terminal, service, or parent agent controls it
without becoming part of the agent core.

## Processes and other agents

Python programs run with two useful environment variables:

- `EKO_SESSION` is this agent's attributed inbox socket.
- `EKO_AGENT` points to this agent's own executable.

Running `EKO_AGENT` creates another agent. It inherits `EKO_MODEL`, and its new
model-socket connection gives it an independent conversation. Children can run
concurrently, communicate through their session sockets, or be placed in another
sandbox using ordinary process and namespace mechanisms.

When sandboxed, Eko is PID 1. On shutdown it gives remaining descendants a short
`SIGTERM` grace period, then kills and reaps anything still running. This includes
detached processes, double-forks, child agents, and processes in nested PID
namespaces. Files written to the mounted workspace intentionally persist.

## Sending input

Every agent exposes a Unix socket through `EKO_SESSION`. Any local process can send
text using only the Python standard library:

```python
import json, os, socket

event = {
    "type": "input",
    "content": [{"type": "text", "text": "The background job finished."}],
}
with socket.socket(socket.AF_UNIX) as connection:
    connection.connect(os.environ["EKO_SESSION"])
    connection.sendall((json.dumps(event) + "\n").encode())
```

Content can include images by workspace-relative path:

```json
{"type":"input","content":[{"type":"image","path":"render.png"}]}
```

or as base64 data from memory or another workspace:

```json
{"type":"input","content":[{"type":"image","media_type":"image/png","data":"..."}]}
```

The receiver assigns provenance using the delivery path and Unix peer credentials;
senders do not claim a source. Model inputs are headed `[terminal]`,
`[python exit=N]`, `[process-PID]`, or `[harness]`. Multiple inputs may be grouped
into one user-role message without losing their individual provenance.

Send `{"type":"interrupt"}` to interrupt the agent's current model call or Python
execution.

## Requirements and tests

Eko requires Python 3.10 or newer. The included host requires the Claude Code CLI
and a working Claude subscription; it prompts for authentication when necessary.
Sandboxing additionally requires Bubblewrap.

Run the tests with:

```bash
uv run --with prompt-toolkit --with rich python -m unittest -q test_eko.py
```
