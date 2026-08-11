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

The script requires Python 3.10 or newer. `uv` installs its Python dependencies
automatically. It uses the Claude Code CLI in print mode, so `claude` must be
installed and able to access a model through your subscription. If needed, Eko
will prompt you to sign in.

Run the tests with:

```bash
uv run --with prompt-toolkit --with rich test_eko.py
```
