# Contributing

Python 3.12+ and [uv](https://docs.astral.sh/uv/) are required. Node 20+ is only needed for the VS Code extension.

## Setup

```bash
make setup
```

That runs `uv sync` and installs pre-commit hooks (ruff check + format). `make help` lists every target.

## Checks

CI (`.github/workflows/pipeline.yaml`) runs `make all`. That is check-only: format, [ruff](https://docs.astral.sh/ruff/), [mypy](https://mypy-lang.org/), then pytest. It does not rewrite files.

| Command | What it does |
|---------|----------------|
| `make format` | Write ruff formatting in `src` and `tests` |
| `make lint-fix` | Apply ruff auto-fixes |
| `make unsafe` | Apply ruff unsafe auto-fixes |
| `make all` | Same checks as CI |

Pre-commit runs ruff only. mypy is Makefile/CI (`uv run mypy -p shpe`).

## Tests

Checker behavior is covered by snippets in [`tests/checker_cases/`](tests/checker_cases/). Add a code snippet to the appropriate test file and include it in the `pytest.param(...)` list. [`tests/test_checker.py`](tests/test_checker.py) parses each snippet with the checker, then `exec`s it with NumPy:

- If a runtime error is raised, the checker must report an error on one of those lines.
- If runtime succeeds, the checker must report no errors, and inferred shapes must match `.shape` for names that exist in the runtime namespace.

```bash
make test
```

Check out the checker architecture in [`docs/architecture.md`](docs/architecture.md).

## VS Code extension

```bash
make vscode-setup
```

Then use **Run Extension** in [`shpe-vscode/.vscode/launch.json`](shpe-vscode/.vscode/launch.json) (F5 from that folder). The selected Python interpreter must be able to `import shpe`.

`make vscode-package` copies the root README and LICENSE into `shpe-vscode/` (those copies are gitignored) and builds a `.vsix`.
