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

    def error(self, node: ast.AST, code: ErrorCode, message: str) -> None:
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
        line, col, end_col = _position(node)
        self.warnings.append(
            {"line": line, "col": col, "end_col": end_col, "message": message}
        )

    def hint(self, node: ast.AST, shape: tuple[Any, ...]) -> None:
        self.inlay_hints.append(
            {
                "line": getattr(node, "lineno", 0),
                "col": getattr(node, "end_col_offset", 0),
                "shape": shape,
            }
        )
