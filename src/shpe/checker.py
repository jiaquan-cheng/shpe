import ast
from typing import Any

from shpe.diagnostics import Diagnostics, ErrorCode
from shpe.environment import (
    AbstractObject,
    Environment,
    ScalarValue,
    Shape,
    ShapeState,
    Tensor,
    ValueState,
)
from shpe.extractor import Extractor
from shpe.resolver import Resolver


class Checker(ast.NodeVisitor):
    """AST visitor that walks through Python code to infer and check NumPy shapes"""

    def __init__(self) -> None:
        self.env = Environment()
        self.diagnostics = Diagnostics()
        self._class_definition_depth = 0
        self.extractor = Extractor(self.env, self.diagnostics)
        self.resolver = Resolver(
            self.extractor,
            self.env,
            self.diagnostics,
            self,
        )

    @property
    def shapes(self) -> dict[str, Tensor]:
        """Convenience property for tests and CLI to access global shapes."""
        return self.env.shapes

    @property
    def scalar_values(self) -> dict[str, ScalarValue]:
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

        if isinstance(node.target, ast.Attribute):
            inferred_shape = self.resolver.shape(node.value)
            self._assign_instance_attribute(node.target, node.value, node.annotation)
            if isinstance(inferred_shape, Shape):
                self.diagnostics.hint(node, inferred_shape)
            return
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
        if node.value is not None:
            self._bind_object_assignment(var_name, node.value)

        if isinstance(inferred_shape, Shape):
            self.diagnostics.hint(node, inferred_shape)
        if (
            isinstance(annotated_shape, Shape)
            and isinstance(inferred_shape, Shape)
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

        if isinstance(inferred_shape, Shape):
            self.diagnostics.hint(node, inferred_shape)

        shape_to_set = (
            inferred_shape if inferred_shape is not None else ShapeState.UNKNOWN
        )
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.env.set_shape(target.id, shape_to_set)
                self._bind_object_assignment(target.id, node.value)
            elif isinstance(target, ast.Attribute):
                self._assign_instance_attribute(target, node.value)

    def visit_Expr(self, node: ast.Expr) -> None:
        """Evaluate method calls whose only effect is mutating an object."""
        if isinstance(node.value, ast.Call):
            self.resolver.call_shape(node.value)
        else:
            self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Register a function and analyze its body at the definition."""
        self.env.set_function(node.name, node)
        self.env.push_child()
        hints_were_active = self.diagnostics.active_hint_collection
        if self._class_definition_depth:
            self.diagnostics.deactivate_hints()
        try:
            self.resolver.analyze_function_definition(node)
        finally:
            self.env.pop_child()
            if hints_were_active:
                self.diagnostics.activate_hints()

    def visit_If(self, node: ast.If) -> None:
        """Handles if statements by tainting variables modified inside branches."""
        test_str = ast.unparse(node.test).replace('"', "'")

        # Evaluate normally for the standard main block
        if test_str == "__name__ == '__main__'":
            self.generic_visit(node)
        else:
            self._visit_block_with_unknown_writes(node.body)
            if node.orelse:
                self._visit_block_with_unknown_writes(node.orelse)

    def visit_For(self, node: ast.For) -> None:
        """Handles for loops by tainting variables modified inside the loop."""
        self._visit_block_with_unknown_writes(node.body)

    def visit_While(self, node: ast.While) -> None:
        """Handles while loops by tainting variables modified inside the loop."""
        self._visit_block_with_unknown_writes(node.body)

    def visit_AsyncFor(self, node: ast.AsyncFor) -> None:
        """Handles async for loops."""
        self._visit_block_with_unknown_writes(node.body)

    def visit_Import(self, node: ast.Import) -> None:
        """Tracks imported modules (e.g., import numpy, import numpy as np)."""
        for alias in node.names:
            if alias.name in {"numpy", "torch"}:
                imported_name = alias.asname if alias.asname else alias.name
                self.env.set_module(alias.name, imported_name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Tracks class definitions (e.g., class MyClass:)."""
        self.env.set_class(node.name, node)
        self.env.push_child()
        errors_were_active = self.diagnostics.active_error_collection
        self.diagnostics.deactivate_errors()
        self._class_definition_depth += 1
        try:
            self.generic_visit(node)
        finally:
            self._class_definition_depth -= 1
            self.env.pop_child()
            if errors_were_active:
                self.diagnostics.activate_errors()

    def _bind_object_assignment(self, name: str, value: ast.AST) -> None:
        """Bind a name to a new class instance or an existing object alias."""
        if isinstance(value, ast.Call) and isinstance(value.func, ast.Name):
            class_def = self.env.get_class(value.func.id)
            if class_def is not None:
                abstract_object = AbstractObject(name, class_def)
                self.env.set_object(name, abstract_object)
                self.resolver.initialize_instance(value, class_def, abstract_object)
                return
        elif isinstance(value, ast.Name):
            existing_object = self.env.get_object(value.id)
            if existing_object is not None:
                self.env.set_object(name, existing_object)
                return

        self.env.set_object(name, None)

    def _infer_abstract_value(
        self, value: ast.AST | None
    ) -> Tensor | ScalarValue | AbstractObject | ValueState:
        if isinstance(value, ast.Constant) and isinstance(value.value, (int, float)):
            return ScalarValue(value.value)

        if isinstance(value, ast.Name):
            abstract_object = self.env.get_object(value.id)
            if abstract_object is not None:
                return abstract_object
            scalar = self.env.get_scalar(value.id)
            if scalar is not None:
                return scalar
            shape = self.env.get_shape(value.id)
            if isinstance(shape, (Shape, ShapeState)):
                return Tensor(shape)

        if isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name):
            abstract_value = self.env.get_attribute(value.value.id, value.attr)
            if abstract_value is not None:
                return abstract_value

        inferred_shape = self.resolver.shape(value)
        if isinstance(inferred_shape, (Shape, ShapeState)):
            return Tensor(inferred_shape)
        return ValueState.UNKNOWN

    def _assign_instance_attribute(
        self,
        target: ast.Attribute,
        value: ast.AST | None,
        annotation: ast.AST | None = None,
    ) -> None:
        if not isinstance(target.value, ast.Name):
            return

        abstract_value = self._infer_abstract_value(value)
        if annotation is not None and isinstance(abstract_value, Tensor):
            annotated_shape = self.extractor.annotation_shape(annotation)
            if isinstance(annotated_shape, Shape):
                abstract_value = Tensor(annotated_shape)
        self.env.set_attribute(target.value.id, target.attr, abstract_value)

    def _visit_block_with_unknown_writes(self, body: list[ast.stmt]) -> None:
        """Visit a block, then mark its assigned variables and fields unknown.
        Used for complex code blocks where we cannot guarantee shape inference.
        Unknown is safer than an incorrect shape.

        Args:
            body: A list of AST statements representing the code block.
        """
        tainted_variables = set()
        tainted_attributes: set[tuple[str, str]] = set()
        for stmt in body:
            self.visit(stmt)
            for target in ast.walk(stmt):
                if isinstance(target, ast.Name) and isinstance(target.ctx, ast.Store):
                    tainted_variables.add(target.id)
                elif (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.ctx, ast.Store)
                    and isinstance(target.value, ast.Name)
                ):
                    tainted_attributes.add((target.value.id, target.attr))

        for var in tainted_variables:
            self.env.set_shape(var, ShapeState.UNKNOWN)
            if var in self.env.scalar_values:
                self.env.set_scalar(var, ValueState.UNKNOWN)

        for object_name, attribute in tainted_attributes:
            self.env.set_attribute(object_name, attribute, ValueState.UNKNOWN)
