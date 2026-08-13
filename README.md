# Eko

Eko is a coding agent with almost no harness: an LLM, Python, and a persistent
folder.

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

Python is the only tool built into the harness. Through it, the model can inspect
the folder, use the shell, and write its own tools.

The core is one `Eko` object. It accepts any model object with `ask`, `interrupt`,
and `close` methods; the CLI uses the included Claude adapter. The terminal
interface lives after the provider-neutral core and only observes its events.

Run it interactively:

```bash
uv run eko.py
```

Or give it a working folder and an initial prompt:

```bash
uv run eko.py --cwd ~/projects/my-project "Find and fix a bug"
```

Feral mode removes the completion state and keeps Eko acting until you interrupt
it with Escape:

```bash
uv run eko.py --feral "Keep improving this project"
```

Run without the terminal interface with `--headless`. Every session exposes a
Unix socket through `EKO_SESSION`; `--session-socket` selects a known path.
External processes send one JSON object per line using only the Python standard
library:

```python
import json, os, socket

event = {
    "type": "input",
    "content": [{"type": "text", "text": "The background job finished."}],
}
payload = json.dumps(event).encode()
with socket.socket(socket.AF_UNIX) as connection:
    connection.connect(os.environ["EKO_SESSION"])
    connection.sendall(payload + b"\n")
```

Content may also contain images, either by workspace-relative path:

```json
{"type":"image","path":"render.png"}
```

or as base64 data from another workspace or an in-memory computation:

```json
{"type":"image","media_type":"image/png","data":"..."}
```

The harness assigns provenance from the delivery path. Terminal input is
`terminal`, execution output is `python`, harness guidance is `harness`, and socket
input is `process-<pid>` using kernel-attested Unix peer credentials. Senders do
not specify their source. Each user message contains one or more inputs beginning
with a header such as `[terminal]`, `[python exit=0]`, or `[process-1842]`.

The socket also accepts `{"type":"interrupt"}` to interrupt current work.

The script requires Python 3.10 or newer. `uv` installs its Python dependencies
automatically. It uses the Claude Code CLI in print mode, so `claude` must be
installed and able to access a model through your subscription. If needed, Eko
will prompt you to sign in.

Run the tests with:

```bash
uv run --with prompt-toolkit --with rich test_eko.py
```
