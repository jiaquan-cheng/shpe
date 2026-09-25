.PHONY: lint format test all

all: format lint test

# core 

setup:
	uv sync
	uv run pre-commit install

lint:
	uv run ruff check src tests --fix
	uv run mypy -p shpe

unsafe:
	uv run ruff check src tests --fix --unsafe-fixes

format:
	uv run ruff format src tests

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
	rm -rf shpe-vscode/*.vsix shpe-vscode/node_modules
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name ".ruff_cache" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +