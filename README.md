# Eko

Eko is a coding agent with almost no harness: an LLM, Python, and a persistent
folder.

The whole agent is one loop:

```python
while prompt := user():
    message = prompt
    while not (response := llm(message)).is_done():
        message = run(response.python(), cwd)
```

Python is the model's only interface. Through it, the model can inspect the
folder, use the shell, and write its own tools. Everything it writes there
persists.

Run it interactively:

```bash
uv run eko.py
```

Or give it a working folder and an initial prompt:

```bash
uv run eko.py --cwd ~/projects/my-project "Find and fix a bug"
```

The script requires Python 3.10 or newer. `uv` installs its Python dependencies
automatically. It uses the Claude Code CLI in print mode, so `claude` must be
installed and able to access a model through your subscription. If needed, Eko
will prompt you to sign in.
