.PHONY: all help setup lint lint-fix unsafe format format-check test vscode-setup vscode-package clean

all: format-check lint test

# core

help:
	@echo "setup            Install Python deps and pre-commit hooks"
	@echo "format           Format src and tests with ruff"
	@echo "format-check     Check formatting without writing files"
	@echo "lint             Ruff check (no fix) and mypy"
	@echo "lint-fix         Apply ruff auto-fixes"
	@echo "unsafe           Apply ruff unsafe auto-fixes"
	@echo "test             Run pytest"
	@echo "all              format-check, lint, and test (CI)"
	@echo "vscode-setup     Install Python and extension npm deps"
	@echo "vscode-package   Build a .vsix (copies README and LICENSE)"
	@echo "clean            Remove caches, node_modules, and build artifacts"

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check src tests
	uv run mypy -p shpe

lint-fix:
	uv run ruff check src tests --fix

unsafe:
	uv run ruff check src tests --fix --unsafe-fixes

format:
	uv run ruff format src tests

format-check:
	uv run ruff format src tests --check

test:
	uv run pytest

# vscode extension

vscode-setup:
	uv sync
	cd shpe-vscode && npm install

vscode-package: vscode-setup
	cp README.md shpe-vscode/README.md
	cp LICENSE shpe-vscode/LICENSE
	cd shpe-vscode && npx @vscode/vsce package

# maintenance

clean:
	rm -rf dist shpe-vscode/*.vsix shpe-vscode/node_modules
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".ruff_cache" -o -name ".pytest_cache" -o -name "*.egg-info" \) -prune -exec rm -rf {} +
