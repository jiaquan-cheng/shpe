.PHONY: all help setup lint lint-fix unsafe format format-check test pypi-check vscode-setup vscode-check vscode-package pycharm-check pycharm-package clean

all: format-check lint test pypi-check vscode-check pycharm-check

# core

help:
	@echo "setup            Install Python deps and pre-commit hooks"
	@echo "format           Format src and tests with ruff"
	@echo "format-check     Check formatting without writing files"
	@echo "lint             Ruff check (no fix) and mypy"
	@echo "lint-fix         Apply ruff auto-fixes"
	@echo "unsafe           Apply ruff unsafe auto-fixes"
	@echo "test             Run pytest"
	@echo "pypi-check       Build and check PyPI distributions without publishing"
	@echo "all              format-check, lint, tests, and package checks (CI)"
	@echo "vscode-setup     Install Python and extension npm deps"
	@echo "vscode-check     Build the VS Code extension package"
	@echo "vscode-package   Build a .vsix (copies README and LICENSE)"
	@echo "pycharm-check    Build the PyCharm plugin ZIP"
	@echo "pycharm-package  Build the PyCharm plugin ZIP"
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

pypi-check:
	uv build
	test -n "$$(find dist -maxdepth 1 -type f \( -name '*.whl' -o -name '*.tar.gz' \) -print -quit)"
	rm -rf dist

# vscode extension

vscode-setup:
	uv sync
	cd editors/vscode && npm ci

vscode-check:
	cp README.md editors/vscode/README.md
	cp LICENSE editors/vscode/LICENSE
	cd editors/vscode && npm ci --ignore-scripts && npm install --no-save --package-lock=false @vscode/vsce && npx vsce package --no-dependencies --out $(CURDIR)/editors/vscode/vscode-check.vsix
	test -f editors/vscode/vscode-check.vsix
	rm -f editors/vscode/vscode-check.vsix
	rm -f editors/vscode/README.md editors/vscode/LICENSE

vscode-package: vscode-setup
	cp README.md editors/vscode/README.md
	cp LICENSE editors/vscode/LICENSE
	cd editors/vscode && npx @vscode/vsce package

# pycharm extension

pycharm-check:
	cp README.md editors/pycharm/README.md
	cp LICENSE editors/pycharm/LICENSE
	cd editors/pycharm && ./gradlew clean buildPlugin
	test -n "$$(find editors/pycharm/build/distributions -maxdepth 1 -type f -name '*.zip' -print -quit)"
	cd editors/pycharm && ./gradlew verifyPlugin
	rm -rf editors/pycharm/build
	rm -f editors/pycharm/README.md editors/pycharm/LICENSE

pycharm-package:
	cp README.md editors/pycharm/README.md
	cp LICENSE editors/pycharm/LICENSE
	cd editors/pycharm && ./gradlew buildPlugin

# maintenance

clean:
	rm -rf dist editors/vscode/*.vsix editors/vscode/node_modules editors/pycharm/build
	rm -f editors/vscode/LICENSE editors/vscode/README.md editors/pycharm/LICENSE editors/pycharm/README.md
	find . -type d \( -name "__pycache__" -o -name ".mypy_cache" -o -name ".ruff_cache" -o -name ".pytest_cache" -o -name "*.egg-info" \) -prune -exec rm -rf {} +
