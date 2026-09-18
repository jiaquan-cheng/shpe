import ast
from collections.abc import Callable
from typing import Any, Protocol

from shpy.diagnostics import Diagnostics, ErrorCode
from shpy.environment import Environment
from shpy.extractor import Extractor, ShapeState
from shpy.handlers import Handler


class StatementWalker(Protocol):
    def visit(self, stmt: ast.stmt) -> None: ...


class Resolver:
    """The catch-all shape resolver. It routes the node to the correct handler."""

    def __init__(
        self,
        extractor: Extractor,
        env: Environment,
        diagnostics: Diagnostics,
        walker: StatementWalker,
    ) -> None:
        self.extractor: Extractor = extractor
        self.env: Environment = env
        self.diagnostics: Diagnostics = diagnostics
        self.active_calls: set[str] = set()
        self.handler = Handler(self.extractor, self.diagnostics, self)
        self.walker: StatementWalker = walker

    @property
    def call_handlers(self) -> dict[Any, Callable]:
        return self.handler.call_handlers

    @property
    def attr_handlers(self) -> dict[Any, Callable]:
        return self.handler.attr_handlers

    @property
    def binop_handlers(self) -> dict[Any, Callable]:
        return self.handler.binop_handlers

    def shape(self, node: ast.AST | None) -> tuple[int | str, ...] | ShapeState | None:
        """Extracts and infers shape properties from an arbitrary AST expression.

        Args:
            node: The AST node to evaluate, or None.

        Returns:
            A tuple representing the shape dimensions, or None if uninferrable.
        """
        if node is None:
            return None

        if isinstance(node, ast.Name):
            return self.env.get_shape(node.id)

        if isinstance(node, ast.Call):
            return self.call_shape(node)

        if isinstance(node, ast.BinOp):
            return self.binop_shape(node)

        if isinstance(node, ast.Constant) and isinstance(
            node.value, (int, float, bool)
        ):
            return ()

        if isinstance(node, (ast.Tuple, ast.List)):
            return self.extractor.literal_shape(node)

        if isinstance(node, ast.Attribute):
            return self.attribute_shape(node)

        if isinstance(node, ast.Subscript):
            return self.subscript_shape(node)
        return None

    def call_shape(self, node: ast.Call) -> tuple[int | str, ...] | ShapeState | None:
        """Infers the shape of a function or method call node.
        Handles both user-defined function and NumPy functions.

        Args:
            node: The `ast.Call` node representing the function invocation.

        Returns:
            The inferred shape tuple, or None if the function is uninferrable.
        """
        func_name = ast.unparse(node.func)

        # Check if it's a user-defined function call
        function = self.env.get_function(func_name)
        if function is not None:
            return self.user_function_call(node, function)

        for suffix in self.call_handlers:
            if func_name.endswith(suffix):
                handler = self.call_handlers[suffix]
                return handler(node)
        return None

    def attribute_shape(
        self, node: ast.Attribute
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Handles attribute-based shape changes, such as `.T` (transpose).

        Args:
            node: The `ast.Attribute` node being evaluated.

        Returns:
            The transformed shape tuple, or None if uninferrable.
        """
        handler = self.attr_handlers.get(node.attr)
        if handler:
            return handler(node)
        return None

    def binop_shape(self, node: ast.BinOp) -> tuple[int | str, ...] | ShapeState | None:
        """Infers and validates shapes for binary operations like `+`, `-`, and `@`.

        Args:
            node: The `ast.BinOp` node containing left, right operands, and operator.

        Returns:
            The broadcasted or validated result shape tuple, or None on failure.
        """
        left_shape = self.shape(node.left)
        right_shape = self.shape(node.right)

        if left_shape is None or right_shape is None:
            return None
        if left_shape is ShapeState.UNKNOWN or right_shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN

        handler = self.binop_handlers.get(type(node.op))
        if handler:
            return handler(node, left_shape, right_shape)
        return None

    def user_function_def(
        self, node: ast.FunctionDef
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Handels user-defined function calls.

        Args:
            node: The `ast.Call` node representing the invocation.
            func_node: The `ast.FunctionDef` node of the target function.

        Returns:
            The inferred return shape tuple, or None if uninferrable.
        """

        func_name = node.name

        # does not handle recursion
        if func_name in self.active_calls:
            return None

        self.active_calls.add(func_name)

        # seperates local and gloabl variables
        self.env.push_child()

        try:
            self.bind_function_arguments(None, node)

            if node.returns:
                return self.extractor.annotation_shape(node.returns)

            return self.function_body_shape(node.body)
        finally:
            self.env.pop_child()
            self.active_calls.remove(func_name)

    def user_function_call(
        self, node: ast.Call, func_node: ast.FunctionDef
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Handels user-defined function calls.

        Args:
            node: The `ast.Call` node representing the invocation.
            func_node: The `ast.FunctionDef` node of the target function.

        Returns:
            The inferred return shape tuple, or None if uninferrable.
        """

        func_name = func_node.name

        # does not handle recursion
        if func_name in self.active_calls:
            return None

        self.active_calls.add(func_name)

        # seperates local and gloabl variables
        self.env.push_child()
        self.diagnostics.deactivate_hints()
        try:
            self.bind_function_arguments(node, func_node)

            if func_node.returns:
                return self.extractor.annotation_shape(func_node.returns)

            return self.function_body_shape(func_node.body)
        finally:
            self.env.pop_child()
            self.active_calls.remove(func_name)
            self.diagnostics.activate_hints()

    def function_body_shape(
        self, body: list[ast.stmt]
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Walks through function body statements and tracks the final return shape.

        Args:
            body: A list of AST statement nodes inside the function definition.

        Returns:
            The inferred return shape tuple, or None.
        """
        inferred_return_shape = None
        for stmt in body:
            self.walker.visit(stmt)
            if isinstance(stmt, ast.Return):
                inferred_return_shape = self.shape(stmt.value)
            if (
                inferred_return_shape is not None
                and inferred_return_shape is not ShapeState.UNKNOWN
            ):
                self.diagnostics.hint(stmt, inferred_return_shape)
        return inferred_return_shape

    def bind_function_arguments(
        self, node: ast.Call | None, func_node: ast.FunctionDef
    ) -> None:
        """Binds positional, default, and annotated arg shapes to the local func scope.

        Args:
            node: The `ast.Call` node supplying the argument values.
            If `None`, no arguments are provided.
            func_node: The `ast.FunctionDef` node being invoked.
        """
        args_args = func_node.args.args
        defaults = func_node.args.defaults
        num_required = len(args_args) - len(defaults)
        provided_shapes = (
            [self.shape(arg) for arg in node.args] if node is not None else []
        )

        for i, param in enumerate(args_args):
            arg_shape = provided_shapes[i] if i < len(provided_shapes) else None

            if arg_shape is None and i >= num_required:
                arg_shape = self.shape(defaults[i - num_required])

            if arg_shape is not None:
                self.env.set_shape(param.arg, arg_shape)
            else:
                self.env.set_shape(param.arg, ShapeState.UNKNOWN)

            if param.annotation and arg_shape and arg_shape is not ShapeState.UNKNOWN:
                expected_shape = self.extractor.annotation_shape(param.annotation)
                if (
                    expected_shape is not None
                    and expected_shape is not ShapeState.UNKNOWN
                    and expected_shape != arg_shape
                ):
                    self.diagnostics.error(
                        node if node is not None else func_node,
                        ErrorCode.ANNOTATION,
                        (
                            f"Argument annotated as {expected_shape}, "
                            f"but expression has the shape {arg_shape}. "
                        ),
                    )
            # hint is not end of line, so we leave it out until it is fixed
            # if arg_shape is not None and arg_shape is not ShapeState.UNKNOWN:
            #     self.diagnostics.hint(func_node, arg_shape)

    def subscript_shape(
        self, node: ast.Subscript
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers shape resulting from array slicing and indexing operations (`a[1:3]`).

        Args:
            node: The `ast.Subscript` node containing the slice/index expression.

        Returns:
            The sliced shape tuple, or None.
        """
        shape = self.shape(node.value)
        if shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN
        if shape is None:
            return None

        slice_node = node.slice
        slices = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]

        result_shape: list[int | str] = []
        shape_idx = 0

        for s in slices:
            if isinstance(s, ast.Constant) and s.value is Ellipsis:
                ellipsis_count = len(shape) - len(slices) + 1
                for _ in range(max(0, ellipsis_count)):
                    if shape_idx < len(shape):
                        result_shape.append(shape[shape_idx])
                        shape_idx += 1
                continue

            if shape_idx >= len(shape):
                break

            current_dim = shape[shape_idx]

            if isinstance(s, ast.Slice):
                dim_len = self.extractor.slice_dimension_length(current_dim, s)
                result_shape.append(dim_len)
                shape_idx += 1
            elif isinstance(s, ast.Constant) and isinstance(s.value, int):
                shape_idx += 1
            else:
                shape_idx += 1

        while shape_idx < len(shape):
            result_shape.append(shape[shape_idx])
            shape_idx += 1

        return tuple(result_shape)
