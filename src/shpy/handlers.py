import ast
import math
from collections.abc import Callable
from typing import Any, Protocol

from shpy.diagnostics import Diagnostics, ErrorCode
from shpy.environment import ShapeState
from shpy.extractor import Extractor


class HandlerResolver(Protocol):
    def shape(
        self, node: ast.AST | None
    ) -> tuple[int | str, ...] | ShapeState | None: ...


def map_handlers(handler_mappings: list[tuple]) -> dict[Any, Callable]:
    return {op: handler for handler, ops in handler_mappings for op in ops}


def _broadcast_shapes(
    shape1: tuple[int | str, ...], shape2: tuple[int | str, ...]
) -> tuple[int | str, ...] | None:
    """Broadcasts two shape tuples following standard NumPy right-to-left rules.

    Args:
        shape1: The first shape tuple.
        shape2: The second shape tuple.

    Returns:
        The broadcasted result shape tuple, or None if shapes are incompatible.
    """
    len1, len2 = len(shape1), len(shape2)
    max_len = max(len1, len2)

    p1 = (1,) * (max_len - len1) + shape1
    p2 = (1,) * (max_len - len2) + shape2

    result = []
    for d1, d2 in zip(p1, p2, strict=False):
        if d1 == d2:
            result.append(d1)
        elif d1 == 1:
            result.append(d2)
        elif d2 == 1:
            result.append(d1)
        else:
            return None
    return tuple(result)


class Handler:
    """Handles primarily NumPy specific functions"""

    def __init__(
        self, extractor: Extractor, diagnostic: Diagnostics, resolver: HandlerResolver
    ) -> None:
        self.extractor: Extractor = extractor
        self.diagnostics: Diagnostics = diagnostic
        self.resolver: HandlerResolver = resolver
        handler_mappings: list[tuple] = [
            (
                lambda node: (
                    self.extractor.literal_shape(node.args[0]) if node.args else None
                ),
                ["array"],
            ),
            (
                lambda node: (
                    self.extractor.expr_shape(node.args[0]) if node.args else None
                ),
                ["zeros", "ones", "empty", "full"],
            ),
            (self._infer_reshape, ["reshape"]),
            (self._infer_flatten, ["flatten", "ravel"]),
            (self._infer_squeeze, ["squeeze"]),
            (self._infer_expand_dims, ["expand_dims"]),
            (self._infer_swapaxes, ["swapaxes"]),
            (self._infer_resize, ["resize"]),
            (
                self._infer_random_shape,
                ["rand", "randn", "random", "random_sample", "ranf", "sample"],
            ),
            (self._infer_randint_shape, ["randint"]),
            (self._infer_random_distribution_shape, ["uniform", "normal"]),
            (self._infer_reduction, ["sum", "mean", "prod", "min", "max"]),
            (self._infer_arange, ["arange"]),
            (self._infer_linspace, ["linspace", "logspace", "geomspace"]),
            (self._infer_meshgrid, ["meshgrid"]),
        ]
        attr_mappings: list[tuple] = [
            (self._infer_transpose, ["T"]),
        ]

        binop_mappings: list[tuple] = [
            (self._infer_matmult_shape, [ast.MatMult]),
            (self._infer_elementwise_shape, [ast.Add, ast.Sub, ast.Mult, ast.Div]),
        ]

        self.call_handlers = map_handlers(handler_mappings)

        self.attr_handlers = map_handlers(attr_mappings)

        self.binop_handlers = map_handlers(binop_mappings)

    def _infer_reshape(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers and validates shape changes resulting from a reshape operation.

        Args:
            node: The `ast.Call` node representing the reshape operation.

        Returns:
            The new shape tuple after validation, or None if validation fails.
        """
        old_shape, args = self._infer_call_target(node)

        if old_shape is None:
            return None
        if old_shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN

        if not args:
            self.diagnostics.error(
                node,
                ErrorCode.RESHAPE,
                "reshape called without a new shape argument. ",
            )
            return None

        new_shape_arg = args[0]
        new_shape = self.extractor.expr_shape(new_shape_arg)

        if new_shape is None:
            return None
        if new_shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN

        if any(isinstance(d, str) or d < 0 for d in old_shape):
            self.diagnostics.error(
                node,
                ErrorCode.RESHAPE,
                "All dimensions in the original shape must be positive. ",
            )
            return None
        old_total = math.prod([d for d in old_shape if isinstance(d, int)])

        # accepts -1 (NumPy wildcard) but rejects other negative values
        if any(isinstance(d, str) or d < -1 for d in new_shape):
            self.diagnostics.error(
                node,
                ErrorCode.RESHAPE,
                "New shape dimensions cannot be -2 or smaller. ",
            )
            return None

        if new_shape.count(-1) > 1:
            self.diagnostics.error(
                node,
                ErrorCode.RESHAPE,
                "New shape cannot have multiple -1 dimensions. ",
            )
            return None
        new_total = math.prod(d for d in new_shape if d != -1 and isinstance(d, int))

        # If a -1 is used, check if divisible
        if (
            new_shape.count(-1) == 1
            and isinstance(new_total, int)
            and new_total > 0
            and isinstance(old_total, int)
            and old_total % new_total == 0
        ) or old_total == new_total:
            inferred_dim = (
                old_total // new_total
                if isinstance(old_total, int) and isinstance(new_total, int)
                else -1
            )
            return tuple(inferred_dim if d == -1 else d for d in new_shape)

        self.diagnostics.error(
            node,
            ErrorCode.RESHAPE,
            f"Cannot reshape array of size {old_total} into shape {new_shape}. ",
        )
        return None

    def _infer_flatten(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers the resulting shape for flatten or ravel operations.

        Args:
            node: The `ast.Call` node representing the flatten/ravel call.

        Returns:
            A 1-dimensional tuple containing the product of the original shape, or None.
        """
        shape, _ = self._infer_call_target(node)
        if shape is not None and shape is not ShapeState.UNKNOWN:
            return (math.prod(shape),)
        if shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN
        return None

    def _infer_squeeze(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers shape changes from squeeze operations, optionally along specific axes.

        Args:
            node: The `ast.Call` node representing the squeeze call.

        Returns:
            The squeezed shape tuple, or None if invalid.
        """
        shape, args = self._infer_call_target(node)
        if shape is None:
            return None
        if shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN

        axis_node = args[0] if args else None
        if axis_node is None:
            axis_node = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "axis"),
                None,
            )

        if axis_node is None:
            return tuple(dimension for dimension in shape if dimension != 1)

        axis_shape = self.extractor.expr_shape(axis_node)
        if axis_shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN
        if axis_shape is None:
            return None

        axes: list[int] = []
        for axis in axis_shape:
            if not isinstance(axis, int) or not -len(shape) <= axis < len(shape):
                return None
            normalized_axis = axis % len(shape)
            if normalized_axis in axes or shape[normalized_axis] != 1:
                return None
            axes.append(normalized_axis)

        return tuple(
            dimension for index, dimension in enumerate(shape) if index not in axes
        )

    def _infer_expand_dims(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers shape changes from expanding array dimensions.

        Args:
            node: The `ast.Call` node representing the expand_dims call.

        Returns:
            The expanded shape tuple with inserted dimensions, or None.
        """
        shape, args = self._infer_call_target(node)
        if shape is None:
            return None
        if shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN

        axis_node = args[0] if args else None
        if axis_node is None:
            axis_node = next(
                (keyword.value for keyword in node.keywords if keyword.arg == "axis"),
                None,
            )

        axis_shape = (
            self.extractor.expr_shape(axis_node) if axis_node is not None else None
        )
        if axis_shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN
        if axis_shape is None or len(axis_shape) != 1:
            return None

        axis = axis_shape[0]
        if not isinstance(axis, int) or not -len(shape) - 1 <= axis <= len(shape):
            return None

        insert_at = axis if axis >= 0 else len(shape) + axis + 1
        return (*shape[:insert_at], 1, *shape[insert_at:])

    def _infer_swapaxes(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers shape changes from swapping two axes of an array.

        Args:
            node: The `ast.Call` node representing the swapaxes call.

        Returns:
            The shape tuple with swapped axes, or None if axes are invalid.
        """
        shape, args = self._infer_call_target(node)
        if shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN
        if shape is None or len(args) < 2:
            return None

        first_axis_shape = self.extractor.expr_shape(args[0])
        second_axis_shape = self.extractor.expr_shape(args[1])
        if (
            first_axis_shape is ShapeState.UNKNOWN
            or second_axis_shape is ShapeState.UNKNOWN
        ):
            return ShapeState.UNKNOWN
        if (
            first_axis_shape is None
            or second_axis_shape is None
            or len(first_axis_shape) != 1
            or len(second_axis_shape) != 1
        ):
            return None

        first_axis = first_axis_shape[0]
        second_axis = second_axis_shape[0]
        dimension_count = len(shape)
        if not (
            isinstance(first_axis, int)
            and isinstance(second_axis, int)
            and -dimension_count <= first_axis < dimension_count
            and -dimension_count <= second_axis < dimension_count
        ):
            return None

        first_axis %= dimension_count
        second_axis %= dimension_count
        swapped_shape = list(shape)
        swapped_shape[first_axis], swapped_shape[second_axis] = (
            swapped_shape[second_axis],
            swapped_shape[first_axis],
        )
        return tuple(swapped_shape)

    def _infer_resize(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers shape for resize operations using call target extraction.

        Args:
            node: The `ast.Call` node representing the resize invocation.

        Returns:
            The target resized shape tuple, or None.
        """
        if (
            isinstance(node.func, ast.Attribute)
            and self.resolver.shape(node.func.value) is not None
        ):
            self.diagnostics.warning(
                node,
                "In-place resize detected. The shape may not be accurately inferred.",
            )

        shape, args = self._infer_call_target(node)
        if shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN
        if shape is None or not args:
            return None
        return self.extractor.expr_shape(args[0])

    def _infer_transpose(
        self, node: ast.Attribute
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers shape for attribute-based transposition (`.T`).

        Args:
            node: The `ast.Attribute` node referencing the transpose attribute.

        Returns:
            The reversed shape tuple, or None if uninferrable.
        """
        shape = self.resolver.shape(node.value)
        if shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN
        if shape is not None:
            return tuple(reversed(shape))
        return None

    def _infer_matmult_shape(
        self,
        node: ast.BinOp,
        left_shape: tuple[int | str, ...],
        right_shape: tuple[int | str, ...],
    ) -> tuple[int | str, ...] | None:
        """Infers and validates the shape resulting from matrix multiplication (`@`).

        Args:
            node: The `ast.BinOp` node for the multiplication.
            left_shape: Shape tuple of the left operand.
            right_shape: Shape tuple of the right operand.

        Returns:
            The resulting matrix multiplication shape, or None if dimensions mismatch.
        """

        # promote 1D shapes to 2D for uniform handling
        left_is_1d = len(left_shape) == 1
        right_is_1d = len(right_shape) == 1

        work_left = (1, *left_shape) if left_is_1d else left_shape
        work_right = (*right_shape, 1) if right_is_1d else right_shape

        batch_left, (m, n1) = work_left[:-2], work_left[-2:]
        batch_right, (n2, k) = work_right[:-2], work_right[-2:]

        if n1 != n2:
            left_name = ast.unparse(node.left)
            right_name = ast.unparse(node.right)
            self.diagnostics.error(
                node,
                ErrorCode.MATMUL,
                (
                    f"cannot multiply {left_name} {left_shape} and {right_name} "
                    f"{right_shape}: inner dimensions must match ({n1} != {n2}). "
                ),
            )
            return None

        broadcast_batch = _broadcast_shapes(batch_left, batch_right)
        if broadcast_batch is None:
            left_name = ast.unparse(node.left)
            right_name = ast.unparse(node.right)
            self.diagnostics.error(
                node,
                ErrorCode.MATMUL,
                (
                    f"cannot multiply {left_name} {left_shape} and {right_name} "
                    f"{right_shape}: batch dimensions {batch_left} and {batch_right} "
                    f"are incompatible. "
                ),
            )
            return None

        result = (*broadcast_batch, m, k)

        if left_is_1d:
            result = result[1:]
        if right_is_1d:
            result = result[:-1]

        return result

    def _infer_elementwise_shape(
        self,
        node: ast.BinOp,
        left_shape: tuple[int | str, ...],
        right_shape: tuple[int | str, ...],
    ) -> tuple[int | str, ...] | None:
        """Infers and validates the shape resulting from matrix multiplication (`@`).

        Args:
            node: The `ast.BinOp` node for the multiplication.
            left_shape: Shape tuple of the left operand.
            right_shape: Shape tuple of the right operand.

        Returns:
            The resulting matrix multiplication shape, or None if dimensions mismatch.
        """
        broadcasted = _broadcast_shapes(left_shape, right_shape)
        if broadcasted is None:
            left_name = ast.unparse(node.left)
            right_name = ast.unparse(node.right)
            self.diagnostics.error(
                node,
                ErrorCode.ELEMENTWISE,
                (
                    f"cannot combine {left_name} {left_shape} and {right_name} "
                    f"{right_shape} with element-wise operator. "
                ),
            )
            return None
        return broadcasted

    def _infer_random_shape(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers shape for random generation calls.

        Args:
            node: The `ast.Call` node representing the random function call.

        Returns:
            The inferred shape tuple, or None.
        """
        if len(node.args) == 1:
            extracted = self.extractor.expr_shape(node.args[0])
            if extracted is ShapeState.UNKNOWN:
                return ShapeState.UNKNOWN
            if extracted is not None:
                return extracted

        shape: list[int | str] = []
        for arg in node.args:
            extracted = self.extractor.expr_shape(arg)
            if extracted is ShapeState.UNKNOWN:
                return ShapeState.UNKNOWN
            if extracted and len(extracted) == 1:
                shape.append(extracted[0])
            else:
                return None
        return tuple(shape) if shape else None

    def _infer_randint_shape(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers shape for integer random generation functions via keyword or position.

        Args:
            node: The `ast.Call` node representing the randint call.

        Returns:
            The inferred shape tuple, or None.
        """
        size_node = next(
            (kw.value for kw in node.keywords if kw.arg == "size"),
            None,
        )
        if size_node is None:
            if len(node.args) >= 3:
                size_node = node.args[2]
            elif (
                len(node.args) == 1 and not isinstance(node.func, ast.Attribute)
            ) or len(node.args) == 2:
                return None

        if size_node is not None:
            return self.extractor.expr_shape(size_node)
        return None

    def _infer_random_distribution_shape(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers shape for continuous distribution functions using the size keyword.

        Args:
            node: The `ast.Call` node representing the distribution call.

        Returns:
            The inferred shape tuple, or None.
        """
        size_node = next(
            (kw.value for kw in node.keywords if kw.arg == "size"),
            None,
        )
        if size_node is not None:
            return self.extractor.expr_shape(size_node)

        return None

    def _infer_reduction(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers shape changes resulting from reduction operations (sum, mean, max).

        Args:
            node: The `ast.Call` node representing the reduction function.

        Returns:
            The reduced shape tuple, or None.
        """
        shape, args = self._infer_call_target(node)
        if shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN
        if shape is None:
            return None

        axis_node = next(
            (kw.value for kw in node.keywords if kw.arg == "axis"),
            args[0] if args else None,
        )

        has_axis_kw = any(kw.arg == "axis" for kw in node.keywords)

        if axis_node is None and not has_axis_kw:
            return ()

        axis_shape = (
            self.extractor.expr_shape(axis_node) if axis_node is not None else None
        )
        if axis_shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN
        if axis_shape is None:
            return None

        axes: list[int] = []
        for axis in axis_shape:
            if not isinstance(axis, int) or not -len(shape) <= axis < len(shape):
                return shape
            normalized_axis = axis + len(shape) if axis < 0 else axis
            if normalized_axis in axes:
                return shape
            axes.append(normalized_axis)

        keepdims = next(
            (keyword.value for keyword in node.keywords if keyword.arg == "keepdims"),
            None,
        )
        is_keepdims = isinstance(keepdims, ast.Constant) and keepdims.value is True

        if is_keepdims:
            return tuple(
                1 if index in axes else dimension
                for index, dimension in enumerate(shape)
            )

        return tuple(
            dimension for index, dimension in enumerate(shape) if index not in axes
        )

    def _infer_arange(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers shape for range-based array creation functions like `np.arange`.

        Args:
            node: The `ast.Call` node representing the arange invocation.

        Returns:
            A 1-dimensional shape tuple reflecting the calculated length, or None.
        """
        if len(node.args) < 2 and not any(
            kw.arg in ("start", "stop") for kw in node.keywords
        ):
            return None

        # If positional args are given as scalars: arange(start, stop, step)
        if len(node.args) >= 3:
            start_val = self.extractor.dim_value(node.args[0])
            stop_val = self.extractor.dim_value(node.args[1])
            step_val = self.extractor.dim_value(node.args[2])
            if (
                start_val is ShapeState.UNKNOWN
                or stop_val is ShapeState.UNKNOWN
                or step_val is ShapeState.UNKNOWN
            ):
                return ShapeState.UNKNOWN
            if (
                isinstance(start_val, (int, float))
                and isinstance(stop_val, (int, float))
                and isinstance(step_val, (int, float))
                and step_val != 0
            ):
                length = math.ceil((stop_val - start_val) / step_val)
                return (max(0, int(length)),)
        elif len(node.args) == 2:
            start_val = self.extractor.dim_value(node.args[0])
            stop_val = self.extractor.dim_value(node.args[1])
            if start_val is ShapeState.UNKNOWN or stop_val is ShapeState.UNKNOWN:
                return ShapeState.UNKNOWN
            if isinstance(start_val, (int, float)) and isinstance(
                stop_val, (int, float)
            ):
                length = math.ceil(stop_val - start_val)
                return (max(0, int(length)),)
        elif len(node.args) == 1:
            stop_val = self.extractor.dim_value(node.args[0])
            if stop_val is ShapeState.UNKNOWN:
                return ShapeState.UNKNOWN
            if isinstance(stop_val, (int, float)):
                return (max(0, int(stop_val)),)

        return None

    def _infer_linspace(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers shape for logarithmically or linearly spaced sequence functions.

        Args:
            node: The `ast.Call` node representing the invocation.

        Returns:
            A 1-dimensional tuple containing the size determined by `num`, or None.
        """
        # Check for keyword 'num'
        for kw in node.keywords:
            if kw.arg == "num":
                val = self.extractor.expr_shape(kw.value)
                if val is ShapeState.UNKNOWN:
                    return ShapeState.UNKNOWN
                if val and isinstance(val[0], int):
                    return val

        # Fallback : linspace(start, stop, num=50) -> num is usually 3rd arg
        if len(node.args) >= 3:
            val = self.extractor.expr_shape(node.args[2])
            if val is ShapeState.UNKNOWN:
                return ShapeState.UNKNOWN
            if val and isinstance(val[0], int):
                return val

        # Default num for linspace/logspace/geomspace is 50 if omitted
        return (50,)

    def _infer_meshgrid(
        self, node: ast.Call
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Infers shape coordinates resulting from coordinate matrix creation.

        Args:
            node: The `ast.Call` node representing the meshgrid call.

        Returns:
            The generated shape tuple, factoring in the chosen indexing style, or None.
        """
        input_shapes = []
        for arg in node.args:
            shape = self.resolver.shape(arg)
            if shape is ShapeState.UNKNOWN:
                return ShapeState.UNKNOWN
            if shape is not None and len(shape) == 1:
                input_shapes.append(shape[0])
            else:
                return None

        if input_shapes is None:
            return None

        # Check indexing style (default is 'xy' which swaps the first two dimensions)
        indexing = "xy"
        for kw in node.keywords:
            if (
                kw.arg == "indexing"
                and isinstance(kw.value, ast.Constant)
                and isinstance(kw.value.value, str)
            ):
                indexing = kw.value.value

        if indexing == "xy" and len(input_shapes) >= 2:
            input_shapes[0], input_shapes[1] = input_shapes[1], input_shapes[0]

        return tuple(input_shapes)

    def _infer_call_target(
        self, node: ast.Call
    ) -> tuple[tuple[int | str, ...] | ShapeState | None, list[ast.AST]]:
        """Normalizes a call node, returning target shape and remaining arguments.

        Handles both method calls (`arr.op(arg)`) and func calls (`np.op(arr, arg)`).

        Args:
            node: The `ast.Call` node to normalize.

        Returns:
            A tuple of (target_shape, remaining_args).
        """
        # Check if method call
        if isinstance(node.func, ast.Attribute):
            receiver_shape = self.resolver.shape(node.func.value)
            if receiver_shape is not None:
                return receiver_shape, list(node.args)

        # Else function call
        if node.args:
            target_shape = self.resolver.shape(node.args[0])
            return target_shape, list(node.args[1:])

        return None, list(node.args)
