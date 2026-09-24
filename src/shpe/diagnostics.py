import ast
from enum import Enum
from typing import Any


class ErrorCode(Enum):
    ANNOTATION = "Annotation"
    ELEMENTWISE = "Elementwise"
    MATMUL = "MatMul"
    RESHAPE = "Reshape"
    VALUE = "Value"


def _position(node: ast.AST) -> tuple[int, int, int]:
    col = getattr(node, "col_offset", 0)
    return getattr(node, "lineno", 0), col, getattr(node, "end_col_offset", col)


class Diagnostics:
    """Collects errors, warnings, and inlay hints during a check pass."""

    def __init__(self) -> None:
        self.errors: list[dict[str, Any]] = []
        self.warnings: list[dict[str, Any]] = []
        self.inlay_hints: list[dict[str, Any]] = []
        self.active_error_collection = True
        self.active_warning_collection = True
        self.active_hint_collection = True

    def deactivate_hints(self) -> None:
        """Deactivate inlay hints collection."""
        self.active_hint_collection = False

    def activate_hints(self) -> None:
        """Activate inlay hints collection."""
        self.active_hint_collection = True

    def deactivate_errors(self) -> None:
        """Deactivate error collection."""
        self.active_error_collection = False

    def activate_errors(self) -> None:
        """Activate error collection."""
        self.active_error_collection = True

    def deactivate_warnings(self) -> None:
        """Deactivate warning collection."""
        self.active_warning_collection = False

    def activate_warnings(self) -> None:
        """Activate warning collection."""
        self.active_warning_collection = True

    def error(self, node: ast.AST, code: ErrorCode, message: str) -> None:
        if not self.active_error_collection:
            return
        line, col, end_col = _position(node)
        self.errors.append(
            {
                "line": line,
                "col": col,
                "end_col": end_col,
                "code": code.value,
                "message": message,
            }
        )

    def warning(self, node: ast.AST, message: str) -> None:
        if not self.active_warning_collection:
            return
        line, col, end_col = _position(node)
        self.warnings.append(
            {"line": line, "col": col, "end_col": end_col, "message": message}
        )

    def hint(self, node: ast.AST, shape: tuple[Any, ...]) -> None:
        if not self.active_hint_collection:
            return
        self.inlay_hints.append(
            {
                "line": getattr(node, "lineno", 0),
                "col": getattr(node, "end_col_offset", 0),
                "shape": shape,
            }
        )
