import ast
from typing import Any

from shpy.diagnostics import Diagnostics, ErrorCode
from shpy.environment import Environment, ShapeState
from shpy.extractor import Extractor
from shpy.resolver import Resolver


class Checker(ast.NodeVisitor):
    """AST visitor that walks through Python code to infer and check NumPy shapes"""

    def __init__(self) -> None:
        self.env = Environment()
        self.diagnostics = Diagnostics()
        self.extractor = Extractor(self.env, self.diagnostics)
        self.resolver = Resolver(
            self.extractor,
            self.env,
            self.diagnostics,
            self,
        )

    @property
    def shapes(self) -> dict[str, tuple[Any, ...] | ShapeState]:
        """Convenience property for tests and CLI to access global shapes."""
        return self.env.shapes

    @property
    def scalar_values(self) -> dict[str, int | float | ShapeState]:
        """Convenience property for tests and CLI to access global scalar values."""
        return self.env.scalar_values

    @property
    def errors(self) -> list[dict[str, Any]]:
        """Convenience property for tests and CLI to access collected errors."""
        return self.diagnostics.errors

    @property
    def warnings(self) -> list[dict[str, Any]]:
        """Convenience property for tests and CLI to access collected warnings."""
        return self.diagnostics.warnings

    @property
    def inlay_hints(self) -> list[dict[str, Any]]:
        """Convenience property for tests and CLI to access collected inlay hints."""
        return self.diagnostics.inlay_hints

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        """Handles explicit annotated variable assignments."""

        if not isinstance(node.target, ast.Name):
            return

        var_name = node.target.id

        if isinstance(node.value, ast.Constant) and isinstance(
            node.value.value, (int, float)
        ):
            self.env.set_scalar(var_name, node.value.value)

        annotated_shape = self.extractor.annotation_shape(node.annotation)
        inferred_shape = self.resolver.shape(node.value) if node.value else None

        resolved_shape = (
            annotated_shape if annotated_shape is not None else inferred_shape
        )
        if resolved_shape is None:
            resolved_shape = ShapeState.UNKNOWN
        self.env.set_shape(var_name, resolved_shape)

        if inferred_shape is not None and inferred_shape is not ShapeState.UNKNOWN:
            self.diagnostics.hint(node, inferred_shape)
        if (
            annotated_shape is not None
            and inferred_shape is not None
            and annotated_shape is not ShapeState.UNKNOWN
            and inferred_shape is not ShapeState.UNKNOWN
            and annotated_shape != inferred_shape
        ):
            self.diagnostics.error(
                node,
                ErrorCode.ANNOTATION,
                (
                    f"{var_name} annotated as {annotated_shape}, "
                    f"but expression has the shape {inferred_shape}. "
                ),
            )

    def visit_Assign(self, node: ast.Assign) -> None:
        """Handles implicit variable assignments."""

        if (
            node.value
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, (int, float))
        ):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    self.env.set_scalar(target.id, node.value.value)

        inferred_shape = self.resolver.shape(node.value) if node.value else None

        if inferred_shape is not None and inferred_shape is not ShapeState.UNKNOWN:
            self.diagnostics.hint(node, inferred_shape)

        shape_to_set = (
            inferred_shape if inferred_shape is not None else ShapeState.UNKNOWN
        )
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.env.set_shape(target.id, shape_to_set)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Tracks custom function definitions."""
        if len(self.env) == 1:
            self.resolver.functions[node.name] = node

        # no generic vists as they could overwrite function args in local scope

    def visit_If(self, node: ast.If) -> None:
        """Handles if statements by tainting variables modified inside branches."""
        test_str = ast.unparse(node.test).replace('"', "'")

        # Evaluate normally for the standard main block
        if test_str == "__name__ == '__main__'":
            self.generic_visit(node)
        else:
            self._visit_tainted_block(node.body)
            if node.orelse:
                self._visit_tainted_block(node.orelse)

    def visit_For(self, node: ast.For) -> None:
        """Handles for loops by tainting variables modified inside the loop."""
        self._visit_tainted_block(node.body)

    def visit_While(self, node: ast.While) -> None:
        """Handles while loops by tainting variables modified inside the loop."""
        self._visit_tainted_block(node.body)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Handles async for loops."""
        self._visit_tainted_block(node.body)

    def _visit_tainted_block(self, body: list[ast.stmt]) -> None:
        """Walks a block of code and forces any assigned variables to Unknown.
        Used for complex code blocks where we cannot guarantee shape inference.
        We would rather have Unkown than incorrect shapes.

        Args:
            body: A list of AST statements representing the code block.
        """
        for stmt in body:
            self.visit(stmt)
            for target in ast.walk(stmt):
                if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                    self.env.set_shape(target.id, ShapeState.UNKNOWN)
                    if target.id in self.env.scalar_values:
                        self.env.scalar_values[target.id] = ShapeState.UNKNOWN
