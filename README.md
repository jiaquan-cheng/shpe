# shpe

[![CI](https://github.com/jiaquan-cheng/shpe/actions/workflows/ci.yaml/badge.svg)](https://github.com/jiaquan-cheng/shpe/actions/workflows/ci.yaml)

A lightweight static analyzer for tracking and validating NumPy tensor shapes without running the code. It is designed to both be a CLI tool that can be used in CI/CD pipelines and a VS Code extension for real-time shape inference and error messages during development.

> **Important Note for VS Code Extension:** This extension requires the core Python CLI tool to function. Please make sure you run **`pip install shpe`** in your environment!

## VS Code Extension Preview

![Shpe VS Code Extension Demo is not loaded.](https://raw.githubusercontent.com/jiaquan-cheng/shpe/main/assets/image.png)

## CLI Usage

Alternatively, you can use the CLI tool to check for shape errors in your code.

```bash
shpe examples/intro.py
```

```bash
examples/intro.py:15: [Elementwise] cannot combine a (3, 2) and b (2, 2) with element-wise operator. 
examples/intro.py:17: [MatMul] cannot multiply d (3, 2)and a (3, 2): inner dimensions must match (2 != 3). 
examples/intro.py:18: [Annotation] f annotated as (3, 2), but expression has the shape (2, 3). 

Found 3 error(s) across 1 file(s).
```

You can use the `--show-shapes` flag to display the inferred shapes of all expressions in the code:
```bash
shpe path/to/your/file_or_directory --show-shapes
```

## Installation

Prerequisites: Python 3.12+

- For VS Code extension, search for `shpe` in the VS Code marketplace and install it.

- For both CLI and VS Code extension, install `shpe`:

```bash
pip install shpe
```

## Features
- Infers shapes from NumPy array (`np.array([1, 2, 3])`) and NumPy functions (`np.zeros((3, 2))`, `a.T`).
- Validates shape annotations for NumPy arrays (`c: Annotated[np.ndarray, (3, 2)]`).
- Validates NumPy operations for shape compatibility (`a + b`, `a @ b`).
- Infers shapes for simple functions calls and function bodies (`c = custom_func(a, b)`).
- Tracks scalar variables used in shape definitions (`np.zeros((dim, 2))`).

Checkout `examples/demo.py` for a more comprehensive demonstration of `shpe`'s capabilities.

## Limitations
We prioritize soundness over completeness, so `shpe` might miss errors. When `shpe` is uncertain it, does not infer the shape. 

False positive:
- We do not track inplace function modification like ` b = a.resize((3, 2))`, so it might infer the wrong shape.

False negatives:
- Only supports a subset of NumPy arrays and functions.
- No control flow support (if, for, while), variables touched are not inferred.
- No support for recursive functions.
- To keep development simple,`shpe` identifies functions by suffix, so it may not trigger an error in cases like (`var.expand_dims` without `np.` prefix). 

If you noticed any bugs or have any feature requests, please report them on [GitHub Issues](https://github.com/jiaquan-cheng/shpe/issues).

## Development

Prerequisites: Python 3.12+, [uv](https://docs.astral.sh/uv/)

To get started locally:
```bash
git clone https://github.com/jiaquan-cheng/shpe.git
cd shpe
make setup
```

We would recommend to using the VS Code extension or the `--show-shapes` flag to check the inferred shapes of your code while developing.

- `make` : Runs the test suite and quality checks.
- `make lint` : Runs [Ruff](https://docs.astral.sh/ruff/) and [Mypy](https://mypy-lang.org/) for code quality and type safety.
- `make format` : Auto-format code.
- `make test` : Runs [Pytest](https://pytest.org/).
- `make unsafe`: Runs Ruff unsafe fixes.
- `make clean` : Cleans up the project by removing build artifacts and caches.