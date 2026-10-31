import ast
import math

from shpe.diagnostics import Diagnostics, ErrorCode
from shpe.environment import Environment, ScalarValue, Shape, ShapeState, ValueState


class Extractor:
    """Extracts shapes and scalar values from leaf AST nodes."""

    def __init__(self, env: Environment, diagnostics: Diagnostics) -> None:
        self.env = env
        self.diagnostics = diagnostics

    def dim_value(self, node: ast.AST | None) -> int | ShapeState | None:
        """Extract an integer dimension or scalar value from an AST node."""
        if node is None:
            return None

        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value

        if (
            isinstance(node, ast.UnaryOp)
            and isinstance(node.op, ast.USub)
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
        ):
            value = node.operand.value
            if isinstance(value, float) and not value.is_integer():
                return None
            return -int(value)

        if isinstance(node, ast.Name):
            scalar = self.env.get_scalar(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            attribute_value = self.env.get_attribute(node.value.id, node.attr)
            scalar = (
                attribute_value if isinstance(attribute_value, ScalarValue) else None
            )
            if attribute_value is ValueState.UNKNOWN:
                return ShapeState.UNKNOWN
            if attribute_value is ValueState.UNDEFINED:
                return ShapeState.UNDEFINED
        else:
            return None

        if scalar is None:
            return None
        scalar_value = scalar.value
        if scalar_value is ValueState.UNKNOWN:
            return ShapeState.UNKNOWN
        if scalar_value is ValueState.UNDEFINED:
            return ShapeState.UNDEFINED
        if not isinstance(scalar_value, (int, float)):
            return None

        if not float(scalar_value).is_integer():
            self.diagnostics.error(
                node,
                ErrorCode.VALUE,
                f"Scalar variable '{ast.unparse(node)}' must be an integer, "
                f"got {scalar_value}.",
            )
            return None
        return int(scalar_value)

    def annotation_shape(self, node: ast.AST) -> Shape | ShapeState | None:
        """Extract an explicit shape from an ``Annotated`` type annotation."""
        if not isinstance(node, ast.Subscript):
            return None
        if not (isinstance(node.value, ast.Name) and node.value.id == "Annotated"):
            return None

        slice_node = node.slice
        if not isinstance(slice_node, ast.Tuple) or len(slice_node.elts) != 2:
            return None

        shape_node = slice_node.elts[1]
        if not isinstance(shape_node, (ast.Tuple, ast.List)):
            return None

        dims: list[int] = []
        for item in shape_node.elts:
            value = self.dim_value(item)
            if value is None:
                return None
            if isinstance(value, ShapeState):
                return value
            dims.append(value)
        return Shape(tuple(dims))

    def expr_shape(self, node: ast.AST) -> Shape | ShapeState | None:
        """Extract dimensions from a scalar or a tuple/list expression."""
        top_value = self.dim_value(node)
        if isinstance(top_value, ShapeState):
            return top_value
        if top_value is not None:
            return Shape((top_value,))

        if not isinstance(node, (ast.List, ast.Tuple)):
            return None
        if not node.elts:
            return Shape(())

        dims: list[int] = []
        for elt in node.elts:
            value = self.dim_value(elt)
            if isinstance(value, ShapeState):
                return value
            if value is None:
                return None
            dims.append(value)
        return Shape(tuple(dims))

    def literal_shape(self, node: ast.AST) -> Shape | ShapeState | None:
        """Calculate the recursive shape of nested literal lists or tuples."""
        if isinstance(node, ast.Name):
            return self.env.get_shape(node.id)
        if isinstance(node, ast.Constant):
            return Shape(())
        if not isinstance(node, (ast.List, ast.Tuple)):
            return None
        if not node.elts:
            return Shape((0,))

        sub_shape = self.literal_shape(node.elts[0])
        if isinstance(sub_shape, ShapeState):
            return sub_shape
        if sub_shape is None:
            return Shape((len(node.elts),))
        return Shape((len(node.elts), *sub_shape.dims))

    def slice_dimension_length(self, current_dim: int, slice_node: ast.Slice) -> int:
        """Calculate the length of a dimension after applying a slice."""
        step_value = (
            self.dim_value(slice_node.step) if slice_node.step is not None else 1
        )
        if not isinstance(step_value, int) or step_value == 0:
            return current_dim

        if step_value > 0:
            start_value = (
                self.dim_value(slice_node.lower) if slice_node.lower is not None else 0
            )
            stop_value = (
                self.dim_value(slice_node.upper)
                if slice_node.upper is not None
                else current_dim
            )
            if not isinstance(start_value, int) or not isinstance(stop_value, int):
                return current_dim

            start = max(
                0,
                current_dim + start_value
                if start_value < 0
                else min(start_value, current_dim),
            )
            stop = max(
                0,
                current_dim + stop_value
                if stop_value < 0
                else min(stop_value, current_dim),
            )
            length = math.ceil((stop - start) / step_value)
        else:
            start_value = (
                self.dim_value(slice_node.lower)
                if slice_node.lower is not None
                else current_dim - 1
            )
            stop_value = (
                self.dim_value(slice_node.upper) if slice_node.upper is not None else -1
            )
            if not isinstance(start_value, int) or not isinstance(stop_value, int):
                return current_dim

            start = (
                max(-1, current_dim + start_value)
                if start_value < 0
                else min(start_value, current_dim - 1)
            )
            stop = (
                max(-1, current_dim + stop_value)
                if stop_value < -1
                else min(stop_value, current_dim)
            )
            length = math.ceil((start - stop) / abs(step_value))

        return max(0, length)
