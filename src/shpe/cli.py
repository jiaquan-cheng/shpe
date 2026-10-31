import argparse
import ast
import importlib.metadata
import sys
from pathlib import Path

from shpe.checker import Checker
from shpe.environment import (
    AbstractObject,
    ScalarValue,
    ShapeState,
    Tensor,
    ValueState,
)

try:
    __version__ = importlib.metadata.version("shpe")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"


def main() -> None:
    """CLI entry point for the local Python shape checker."""
    parser = argparse.ArgumentParser(description="A local Python shape checker.")
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Files or directories to check"
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {__version__}"
    )
    parser.add_argument(
        "--show-shapes",
        action="store_true",
        help="Display inferred shapes, scalar values, and object attributes",
    )
    args = parser.parse_args()

    files = discover_files(args.paths)
    all_errors: list[str] = []
    all_shapes: dict[Path, dict[str, Tensor]] = {}
    all_scalars: dict[Path, dict[str, ScalarValue]] = {}
    all_objects: dict[Path, dict[str, AbstractObject | None]] = {}

    for filepath in files:
        file_errors, shapes, scalars, objects = _process_file(filepath)
        all_errors.extend(file_errors)
        if shapes:
            all_shapes[filepath] = shapes
        if scalars:
            all_scalars[filepath] = scalars
        if any(abstract_object is not None for abstract_object in objects.values()):
            all_objects[filepath] = objects

    if args.show_shapes and (all_shapes or all_scalars or all_objects):
        _print_shape_report(all_shapes, all_scalars, all_objects)

    for err in all_errors:
        print(err)

    if all_errors:
        print(f"\nFound {len(all_errors)} error(s) across {len(files)} file(s).")
        sys.exit(1)
    else:
        print(f"Success: Checked {len(files)} file(s), no shape errors found.")
        sys.exit(0)


def discover_files(paths: list[Path]) -> list[Path]:
    """Discovers and collects all Python (.py) source files from the provided paths.

    Args:
        paths: A list of file or directory Path objects to scan.

    Returns:
        A list of resolved Path objects pointing to valid Python source files.
    """
    files_to_check = []
    for path in paths:
        if not path.exists():
            print(f"Error: Path '{path}' not found.", file=sys.stderr)
            continue

        if path.is_file() and path.suffix == ".py":
            files_to_check.append(path)
        elif path.is_dir():
            for subpath in path.rglob("*.py"):
                # Skip hidden directories like .git, .venv, __pycache__
                if not any(part.startswith(".") for part in subpath.parts):
                    files_to_check.append(subpath)

    return files_to_check


def _process_file(
    filepath: Path,
) -> tuple[
    list[str],
    dict[str, Tensor],
    dict[str, ScalarValue],
    dict[str, AbstractObject | None],
]:
    """Processes a single source file, running syntax checks and shape analysis.

    Args:
        filepath: The path to the file to check.

    Returns:
        A tuple containing a list of formatted error messages,
        the extracted shapes, scalars, and object dictionaries.
    """
    errors: list[str] = []
    try:
        code = filepath.read_text(encoding="utf-8")
        lines = code.splitlines()
        tree = ast.parse(code, filename=str(filepath))
    except SyntaxError as e:
        errors.append(f"{filepath}:{e.lineno}: [SyntaxError] {e.msg} ")
        return errors, {}, {}, {}

    checker = Checker()
    checker.visit(tree)

    for warning in checker.warnings:
        line = warning["line"]
        line_idx = max(0, line - 1)
        if line_idx < len(lines):
            line_content = lines[line_idx]
            if "# shpe: ignore" in line_content:
                continue

        code = warning["code"]
        msg = warning["message"]
        errors.append(f"{filepath}:{line}: warning: [{code}] {msg}")

    for error in checker.errors:
        line = error["line"]
        line_idx = max(0, line - 1)
        if line_idx < len(lines):
            line_content = lines[line_idx]
            if "# shpe: ignore" in line_content:
                continue

        code = error["code"]
        msg = error["message"]
        errors.append(f"{filepath}:{line}: [{code}] {msg}")

    return (
        errors,
        checker.env.shapes,
        checker.env.scalar_values,
        checker.env.objects,
    )


def _print_shape_report(
    all_shapes: dict[Path, dict[str, Tensor]],
    all_scalars: dict[Path, dict[str, ScalarValue]],
    all_objects: dict[Path, dict[str, AbstractObject | None]],
) -> None:
    """Prints a formatted report of all inferred shapes and scalar values across files.

    Args:
        all_shapes: Mapping of file paths to their symbol shape definitions.
        all_scalars: Mapping of file paths to their tracked scalar variables.
        all_objects: Mapping of file paths to their tracked object instances.
    """
    print("\n-------- Symbol State (Shapes, Scalars & Objects) --------")

    filepaths = set(all_shapes) | set(all_scalars) | set(all_objects)
    for filepath in sorted(filepaths, key=str):
        print(f"\n{filepath}:")
        symbols = all_shapes.get(filepath, {})
        scalars = all_scalars.get(filepath, {})
        objects = all_objects.get(filepath, {})

        if scalars:
            print("  Scalars:")
            for name, val in sorted(scalars.items()):
                val_str = (
                    "unknown"
                    if val.value is ValueState.UNKNOWN
                    else "undefined"
                    if val.value is ValueState.UNDEFINED
                    else val.value
                )
                print(f"    - {name} = {val_str}")

        if symbols:
            print("  Shapes:")
            symbols_with_unknown_shape: list[str] = []
            symbols_with_undefined_shape: list[str] = []
            for var_name, shape in sorted(symbols.items()):
                if shape.shape is ShapeState.UNKNOWN:
                    symbols_with_unknown_shape.append(var_name)
                elif shape.shape is ShapeState.UNDEFINED:
                    symbols_with_undefined_shape.append(var_name)
                else:
                    print(f"    - {var_name}: {shape.shape}")

            if symbols_with_unknown_shape:
                print(
                    f"    - unknown shape for: {', '.join(symbols_with_unknown_shape)}"
                )
            if symbols_with_undefined_shape:
                print(
                    "    - undefined shape for: "
                    f"{', '.join(symbols_with_undefined_shape)}"
                )

        tracked_objects = [
            (name, abstract_object)
            for name, abstract_object in sorted(objects.items())
            if abstract_object is not None
        ]
        if tracked_objects:
            print("  Objects:")
            seen_objects: set[int] = set()
            for name, abstract_object in tracked_objects:
                if id(abstract_object) in seen_objects:
                    continue
                _print_object_attributes(name, abstract_object, 4, seen_objects)

        if not symbols and not scalars and not tracked_objects:
            print("  - (no symbols tracked)")

    print("\n" + "-" * 30 + "\n")


def _print_object_attributes(
    name: str,
    abstract_object: AbstractObject,
    indent: int,
    seen_objects: set[int],
) -> None:
    """Print an object and its tracked attributes, including nested objects."""
    prefix = " " * indent
    print(f"{prefix}- {name} ({abstract_object.class_def.name}):", end="")
    object_id = id(abstract_object)
    if object_id in seen_objects:
        print(" (already shown)")
        return
    seen_objects.add(object_id)
    print()

    for attribute, value in sorted(abstract_object.attributes.items()):
        attribute_prefix = " " * (indent + 2)
        if isinstance(value, Tensor):
            shape = value.shape
            if shape is ShapeState.UNKNOWN:
                description = "unknown shape"
            elif shape is ShapeState.UNDEFINED:
                description = "undefined shape"
            else:
                description = str(shape)
            print(f"{attribute_prefix}- {attribute}: {description}")
        elif isinstance(value, ScalarValue):
            scalar = value.value
            if scalar is ValueState.UNKNOWN:
                description = "unknown"
            elif scalar is ValueState.UNDEFINED:
                description = "undefined"
            else:
                description = str(scalar)
            print(f"{attribute_prefix}- {attribute} = {description}")
        elif isinstance(value, AbstractObject):
            _print_object_attributes(attribute, value, indent + 2, seen_objects)
        else:
            description = "unknown" if value is ValueState.UNKNOWN else "undefined"
            print(f"{attribute_prefix}- {attribute}: {description}")


if __name__ == "__main__":
    main()
