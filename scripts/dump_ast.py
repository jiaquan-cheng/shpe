import argparse
import ast
import sys
from pathlib import Path


def print_file_ast(filepath: Path) -> None:
    """Reads a Python file and prints its pretty-printed AST structure."""
    if not filepath.exists():
        print(f"Error: File '{filepath}' not found.", file=sys.stderr)
        return

    code = filepath.read_text(encoding="utf-8")
    tree = ast.parse(code, filename=str(filepath))

    print(ast.dump(tree, indent=4))


def main() -> None:
    parser = argparse.ArgumentParser(description="Dumps ast of Python files.")
    parser.add_argument(
        "filepaths", nargs="+", type=Path, help="Files or directories to dump"
    )
    args = parser.parse_args()
    for filepath in args.filepaths:
        print_file_ast(filepath)


if __name__ == "__main__":
    main()
