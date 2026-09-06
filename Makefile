.PHONY: lint format test all

all: format lint test

# core 

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check src tests --fix
	uv run mypy -p shpy

format:
	uv run ruff format src tests

test:
	uv run pytest

# vscode extension

vscode-setup: uv sync
	cd shpy-vscode && npm install

vscode-package: vscode-setup
	cd shpy-vscode && npx @vscode/vsce package --allow-missing-repository

# maintenance

clean:
	rm -rf shpy-vscode/*.vsix shpy-vscode/node_modules
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +