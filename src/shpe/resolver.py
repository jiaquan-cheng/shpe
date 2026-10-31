import ast
from collections.abc import Callable
from typing import Any, Protocol

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
from shpe.handlers import Registry


class StatementWalker(Protocol):
    def visit(self, stmt: ast.stmt) -> None: ...


class Resolver:
    """Resolve expression shapes and analyze calls to user-defined code."""

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
        self.methods_analyzed_for_hints: set[tuple[int, str]] = set()
        self.initialized_instances: dict[int, AbstractObject] = {}
        self.registry = Registry(self.extractor, self.diagnostics, self.env, self)
        self.walker: StatementWalker = walker

    @property
    def call_handlers(self) -> dict[Any, Callable]:
        return self.registry.call_handlers

    @property
    def attr_handlers(self) -> dict[Any, Callable]:
        return self.registry.attr_handlers

    @property
    def binop_handlers(self) -> dict[Any, Callable]:
        return self.registry.binop_handlers

    def shape(self, node: ast.AST | None) -> Shape | ShapeState | None:
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
            return Shape(())

        if isinstance(node, (ast.Tuple, ast.List)):
            return self.extractor.literal_shape(node)

        if isinstance(node, ast.Attribute):
            return self.attribute_shape(node)

        if isinstance(node, ast.Subscript):
            return self.subscript_shape(node)
        return None

    def call_shape(self, node: ast.Call) -> Shape | ShapeState | None:
        """Infer a call's shape using user-defined code or a registered handler.

        Args:
            node: The `ast.Call` node representing the function invocation.

        Returns:
            The inferred shape or state, or None when the call is unsupported.
        """
        func_name = ast.unparse(node.func)

        # Resolve methods on tracked instances before matching library handlers.
        if isinstance(node.func, ast.Attribute) and isinstance(
            node.func.value, ast.Name
        ):
            abstract_object = self.env.get_object(node.func.value.id)
            if abstract_object is not None:
                method = self._find_class_method(
                    abstract_object.class_def, node.func.attr
                )
                if method is not None:
                    return self._analyze_instance_method_call(
                        node, method, abstract_object
                    )

        # Check if it's a user-defined function call
        function = self.env.get_function(func_name)
        if function is not None:
            return self._analyze_function_call(node, function)

        for suffix in self.call_handlers:
            if func_name.endswith(suffix):
                handler = self.call_handlers[suffix]
                return handler(node)
        return None

    def attribute_shape(self, node: ast.Attribute) -> Shape | ShapeState | None:
        """Handles attribute-based shape changes, such as `.T` (transpose).

        Args:
            node: The `ast.Attribute` node being evaluated.

        Returns:
            The transformed shape tuple, or None if uninferrable.
        """
        if isinstance(node.value, ast.Name):
            abstract_value = self.env.get_attribute(node.value.id, node.attr)
            if isinstance(abstract_value, Tensor):
                return abstract_value.shape
            if isinstance(abstract_value, ValueState):
                return (
                    ShapeState.UNKNOWN
                    if abstract_value is ValueState.UNKNOWN
                    else ShapeState.UNDEFINED
                )

        handler = self.attr_handlers.get(node.attr)
        if handler:
            return handler(node)
        return None

    def binop_shape(self, node: ast.BinOp) -> Shape | ShapeState | None:
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
        if isinstance(left_shape, ShapeState):
            return left_shape
        if isinstance(right_shape, ShapeState):
            return right_shape

        handler = self.binop_handlers.get(type(node.op))
        if handler:
            return handler(node, left_shape, right_shape)
        return None

    # User-defined function and method analysis
    def initialize_instance(
        self, node: ast.Call, class_def: ast.ClassDef, abstract_object: AbstractObject
    ) -> None:
        """Analyze `__init__` and populate the instance's abstract attributes."""
        initializer = self._find_class_method(class_def, "__init__")
        if initializer is not None:
            self._analyze_instance_method_call(node, initializer, abstract_object)
        self.initialized_instances[id(abstract_object)] = abstract_object

    @staticmethod
    def _find_class_method(
        class_def: ast.ClassDef, method_name: str
    ) -> ast.FunctionDef | None:
        """Find a method defined directly on a class."""
        return next(
            (
                statement
                for statement in class_def.body
                if isinstance(statement, ast.FunctionDef)
                and statement.name == method_name
            ),
            None,
        )

    def _analyze_instance_method_call(
        self, node: ast.Call, method: ast.FunctionDef, abstract_object: AbstractObject
    ) -> Shape | ShapeState | None:
        """Analyze a method call with `self` bound to its abstract instance."""
        if not method.args.args:
            return None

        active_call = f"{id(abstract_object)}:{method.name}"
        if active_call in self.active_calls:
            return None

        self.active_calls.add(active_call)
        self.env.push_child()
        hints_were_active = self.diagnostics.active_hint_collection
        method_key = (id(abstract_object.class_def), method.name)
        has_instance_state = (
            method.name == "__init__"
            or id(abstract_object) in self.initialized_instances
        )
        collect_method_hints = has_instance_state and (
            method_key not in self.methods_analyzed_for_hints
        )
        if collect_method_hints:
            self.methods_analyzed_for_hints.add(method_key)
            self.diagnostics.activate_hints()
        else:
            self.diagnostics.deactivate_hints()
        try:
            parameters = method.args.args
            self.env.set_object(parameters[0].arg, abstract_object)
            defaults = method.args.defaults
            num_required = len(parameters) - len(defaults)
            keyword_arguments = {
                keyword.arg: keyword.value
                for keyword in node.keywords
                if keyword.arg is not None
            }

            for parameter_index, parameter in enumerate(parameters[1:], start=1):
                argument = None
                positional_index = parameter_index - 1
                if positional_index < len(node.args):
                    argument = node.args[positional_index]
                elif parameter.arg in keyword_arguments:
                    argument = keyword_arguments[parameter.arg]
                elif parameter_index >= num_required:
                    argument = defaults[parameter_index - num_required]

                self._bind_method_parameter(parameter.arg, argument)

            return self._infer_function_body_return_shape(method.body)
        finally:
            self.env.pop_child()
            self.active_calls.remove(active_call)
            if hints_were_active:
                self.diagnostics.activate_hints()
            else:
                self.diagnostics.deactivate_hints()

    def _bind_method_parameter(
        self, parameter_name: str, argument: ast.AST | None
    ) -> None:
        """Bind one method argument to abstract local state."""
        if argument is None:
            self.env.set_object(parameter_name, None)
            self.env.set_shape(parameter_name, ShapeState.UNKNOWN)
            self.env.set_scalar(parameter_name, ValueState.UNKNOWN)
            return

        abstract_value: AbstractObject | ScalarValue | Tensor | ValueState | None = None
        if isinstance(argument, ast.Name):
            abstract_value = self.env.get_object(argument.id)
            if abstract_value is None:
                abstract_value = self.env.get_scalar(argument.id)
            if abstract_value is None:
                shape = self.env.get_shape(argument.id)
                if isinstance(shape, (Shape, ShapeState)):
                    abstract_value = Tensor(shape)
        elif isinstance(argument, ast.Attribute) and isinstance(
            argument.value, ast.Name
        ):
            abstract_value = self.env.get_attribute(argument.value.id, argument.attr)
        elif isinstance(argument, ast.Constant) and isinstance(
            argument.value, (int, float)
        ):
            abstract_value = ScalarValue(argument.value)

        self.env.set_object(
            parameter_name,
            abstract_value if isinstance(abstract_value, AbstractObject) else None,
        )
        if isinstance(abstract_value, Tensor):
            self.env.set_shape(parameter_name, abstract_value)
        elif isinstance(abstract_value, ScalarValue):
            self.env.set_shape(parameter_name, Shape(()))
            self.env.set_scalar(parameter_name, abstract_value)
        elif abstract_value is not None:
            self.env.set_shape(parameter_name, ShapeState.UNKNOWN)
            self.env.set_scalar(parameter_name, ValueState.UNKNOWN)
        else:
            shape = self.shape(argument)
            self.env.set_shape(
                parameter_name,
                shape if shape is not None else ShapeState.UNKNOWN,
            )

    def analyze_function_definition(
        self, node: ast.FunctionDef
    ) -> Shape | ShapeState | None:
        """Analyze a function body in an isolated scope for definition-time hints."""

        func_name = node.name

        # does not handle recursion
        if func_name in self.active_calls:
            return None

        self.active_calls.add(func_name)

        # Keep function-local bindings separate from the caller's scope.
        self.env.push_child()

        try:
            self._bind_function_parameters(None, node)

            if node.returns:
                return self.extractor.annotation_shape(node.returns)

            return self._infer_function_body_return_shape(node.body)
        finally:
            self.env.pop_child()
            self.active_calls.remove(func_name)

    def _analyze_function_call(
        self, node: ast.Call, func_node: ast.FunctionDef
    ) -> Shape | ShapeState | None:
        """Analyze a function call in an isolated scope and infer its return shape."""

        func_name = func_node.name

        # does not handle recursion
        if func_name in self.active_calls:
            return None

        self.active_calls.add(func_name)

        # Keep function-local bindings separate from the caller's scope.
        self.env.push_child()
        hints_were_active = self.diagnostics.active_hint_collection
        self.diagnostics.deactivate_hints()
        try:
            self._bind_function_parameters(node, func_node)

            if func_node.returns:
                return self.extractor.annotation_shape(func_node.returns)

            return self._infer_function_body_return_shape(func_node.body)
        finally:
            self.env.pop_child()
            self.active_calls.remove(func_name)
            if hints_were_active:
                self.diagnostics.activate_hints()

    def _infer_function_body_return_shape(
        self, body: list[ast.stmt]
    ) -> Shape | ShapeState | None:
        """Visit a function body and infer the shape of its final return value.

        Args:
            body: Statements inside the function definition.

        Returns:
            The inferred return shape tuple, or None.
        """
        inferred_return_shape = None
        for stmt in body:
            self.walker.visit(stmt)
            if isinstance(stmt, ast.Return):
                inferred_return_shape = self.shape(stmt.value)
            if isinstance(inferred_return_shape, Shape):
                self.diagnostics.hint(stmt, inferred_return_shape)
        return inferred_return_shape

    def _bind_function_parameters(
        self, node: ast.Call | None, func_node: ast.FunctionDef
    ) -> None:
        """Bind call arguments, defaults, and annotations to function parameters.

        Args:
            node: The call supplying arguments, or None during definition analysis.
            func_node: The function whose parameters are being bound.
        """
        args_args = func_node.args.args
        defaults = func_node.args.defaults
        num_required = len(args_args) - len(defaults)
        provided_args = node.args if node is not None else []

        for i, param in enumerate(args_args):
            expr_node = None
            scalar_val = None
            if i < len(provided_args):
                expr_node = provided_args[i]
            elif i >= num_required:
                expr_node = defaults[i - num_required]

            arg_shape = self.shape(expr_node) if expr_node is not None else None
            if arg_shape is not None:
                self.env.set_shape(param.arg, arg_shape)
            else:
                self.env.set_shape(param.arg, ShapeState.UNKNOWN)

            if isinstance(expr_node, ast.Constant) and isinstance(
                expr_node.value, (int, float)
            ):
                scalar_val = expr_node.value
            if scalar_val is not None:
                self.env.set_scalar(param.arg, scalar_val)

            if param.annotation and isinstance(arg_shape, Shape):
                expected_shape = self.extractor.annotation_shape(param.annotation)
                if isinstance(expected_shape, Shape) and expected_shape != arg_shape:
                    self.diagnostics.error(
                        node if node is not None else func_node,
                        ErrorCode.ANNOTATION,
                        (
                            f"Argument annotated as {expected_shape}, "
                            f"but expression has the shape {arg_shape}. "
                        ),
                    )

    def subscript_shape(self, node: ast.Subscript) -> Shape | ShapeState | None:
        """Infers shape resulting from array slicing and indexing operations (`a[1:3]`).

        Args:
            node: The `ast.Subscript` node containing the slice/index expression.

        Returns:
            The sliced shape tuple, or None.
        """
        if (
            isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Attribute)
            and node.value.func.attr == "meshgrid"
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, int)
        ):
            return self.shape(node.value)

        shape = self.shape(node.value)
        if isinstance(shape, ShapeState):
            return shape
        if shape is None:
            return None

        slice_node = node.slice
        slices = slice_node.elts if isinstance(slice_node, ast.Tuple) else [slice_node]

        result_shape: list[int] = []
        shape_idx = 0

        for s in slices:
            if isinstance(s, ast.Constant) and s.value is Ellipsis:
                ellipsis_count = len(shape) - len(slices) + 1
                for _ in range(max(0, ellipsis_count)):
                    if shape_idx < len(shape):
                        result_shape.append(shape[shape_idx])
                        shape_idx += 1
                continue

            if isinstance(s, ast.Constant) and s.value is None:
                result_shape.append(1)
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
            elif isinstance(s, (ast.List, ast.Tuple)):
                index_shape = self.extractor.literal_shape(s)
                if isinstance(index_shape, Shape):
                    result_shape.extend(index_shape.dims)
                elif isinstance(index_shape, ShapeState):
                    return index_shape
                shape_idx += 1
            else:
                advanced_shape = self.shape(s)
                if isinstance(advanced_shape, Shape):
                    result_shape.extend(advanced_shape.dims)
                elif isinstance(advanced_shape, ShapeState):
                    return advanced_shape
                shape_idx += 1

        while shape_idx < len(shape):
            result_shape.append(shape[shape_idx])
            shape_idx += 1

        return Shape(tuple(result_shape))
