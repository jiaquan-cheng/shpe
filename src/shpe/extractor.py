import ast
import math

from shpe.diagnostics import Diagnostics, ErrorCode
from shpe.environment import Environment, ShapeState


class Extractor:
    """Extracts the shape from the leaf nodes."""

    def __init__(self, env: Environment, diagnostics: Diagnostics) -> None:
        self.env = env
        self.diagnostics = diagnostics

    def dim_value(self, node: ast.AST | None) -> int | str | ShapeState | None:
        """Extracts a scalar integer/float value from an AST node for dim calculations.

        Args:
            node: The AST node containing a scalar value or variable name.

        Returns:
            The extracted integer/string value, or None if unextractable.
        """
        value: int | float | ShapeState | None = None

        if node is None:
            return None

        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value

        # Negative integer via unary minus
        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        ):
            value = node.operand.value
            if value.is_integer():
                return -int(value)

        if isinstance(node, ast.Name):
            value = self.env.get_scalar(node.id)

            if value is None:
                return None
            if value is ShapeState.UNKNOWN:
                return ShapeState.UNKNOWN

            if value is not None and not float(value).is_integer():
                self.diagnostics.error(
                    node,
                    ErrorCode.VALUE,
                    f"Scalar variable '{node.id}' must be an integer, got {value}.",
                )
                return None
            return int(value)

        return None

    def annotation_shape(
        self, node: ast.AST
    ) -> tuple[int | str, ...] | ShapeState | None:
        """Extracts explicit shape tuples from type annotations using `Annotated`.

        Args:
            node: The AST node representing the type annotation.

        Returns:
            The extracted shape tuple, or None.
        """
        if not isinstance(node, ast.Subscript):
            return None
        if not (isinstance(node.value, ast.Name) and node.value.id == "Annotated"):
            return None

        slice_node = node.slice
        if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) != 2:
            return None

        elt = slice_node.elts[1]
        if not isinstance(elt, (ast.Tuple, ast.List)):
            return None

        shape: list[int | str] = []
        for item in elt.elts:
            val = self.dim_value(item)
            if val is None:
                return None
            if val is ShapeState.UNKNOWN:
                return ShapeState.UNKNOWN
            shape.append(val)

        return tuple(shape)

    def expr_shape(self, node: ast.AST) -> tuple[int | str, ...] | ShapeState | None:
        """Extracts shape from a literal list/tuple of dim or a scalar expression.

        Args:
            node: The AST expression node representing dimensions.

        Returns:
            The shape tuple, or None.
        """
        top_val = self.dim_value(node)
        if top_val is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN
        if top_val is not None:
            return (top_val,)

        if not isinstance(node, (ast.List, ast.Tuple)):
            return None

        if not node.elts:
            return ()

        shape: list[int | str] = []
        for elt in node.elts:
            val = self.dim_value(elt)
            if val is ShapeState.UNKNOWN:
                return ShapeState.UNKNOWN
            if val is not None:
                shape.append(val)
            else:
                return None

        return tuple(shape) if shape is not None else None

    def literal_shape(self, node: ast.AST) -> tuple[int, ...] | ShapeState | None:
        """Calculates the recursive shape of nested literal lists or tuples.
        Example: np.array([[1, 2], [3, 4]]) has a shape of (2, 2).

        Args:
            node: The AST node representing literal collections.

        Returns:
            A tuple representing the multi-dimensional structure.
        """
        if isinstance(node, ast.Name):
            shape = self.env.get_shape(node.id)
            return shape
        if isinstance(node, ast.Name):
            return None
        if not isinstance(node, (ast.List, ast.Tuple)):
            return None
        if not node.elts:
            return None

        current_dim = len(node.elts)
        sub_shape = self.literal_shape(node.elts[0])
        if sub_shape is ShapeState.UNKNOWN:
            return ShapeState.UNKNOWN
        if sub_shape is None:
            return (current_dim,)
        return (current_dim, *sub_shape)

    def slice_dimension_length(self, current_dim: int | str, s: ast.Slice) -> int | str:
        """Calculates the resulting length of a single dim under a slice operation.

        Args:
            current_dim: The current dimension size.
            s: The `ast.Slice` node defining bounds and steps.

        Returns:
            The new dimension size after slicing.
        """
        if not isinstance(current_dim, int):
            return current_dim

        step_val = self.dim_value(s.step) if s.step is not None else 1
        if not isinstance(step_val, int) or step_val == 0:
            return current_dim

        if step_val > 0:
            start_val = self.dim_value(s.lower) if s.lower is not None else 0
            stop_val = self.dim_value(s.upper) if s.upper is not None else current_dim

            if not isinstance(start_val, int) or not isinstance(stop_val, int):
                return current_dim

            start = max(
                0,
                current_dim + start_val
                if start_val < 0
                else min(start_val, current_dim),
            )
            stop = max(
                0,
                current_dim + stop_val if stop_val < 0 else min(stop_val, current_dim),
            )
            effective_len = math.ceil((stop - start) / step_val)
        else:
            start_val = (
                self.dim_value(s.lower) if s.lower is not None else current_dim - 1
            )
            stop_val = self.dim_value(s.upper) if s.upper is not None else -1

            if not isinstance(start_val, int) or not isinstance(stop_val, int):
                return current_dim

            start = (
                max(-1, current_dim + start_val)
                if start_val < 0
                else min(start_val, current_dim - 1)
            )
            stop = (
                max(-1, current_dim + stop_val)
                if stop_val < -1
                else min(stop_val, current_dim)
            )
            effective_len = math.ceil((start - stop) / abs(step_val))

        return max(0, effective_len)
