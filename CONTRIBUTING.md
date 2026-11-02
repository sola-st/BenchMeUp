# Contributing

Issues and pull requests are welcome.

## Setup

```
python3 -m venv .venv && . .venv/bin/activate
pip install -e '.[dev]'
```

## Tests

```
pytest -m "not docker and not e2e and not network"     # offline tier, a few minutes
pytest                                                   # everything; needs Docker and network
```

Tests marked `docker`, `network` or `e2e` are skipped in the offline tier. A new behavior comes
with a test that fails without it and passes with it.

## Style

- Comments and docstrings state what the code does and why, in the present tense.
- A task, a model or a build system is never hard-coded in the framework: it belongs in a
  specification, and the framework reads it from there.
- Keep `bench` commands and the packaged `bench` interpreter backward compatible where you can.

## Pull requests

Describe the change and how you tested it. Keep a change focused: one behavior per pull request.
