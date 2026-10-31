"""Shared helpers used by shape operation handlers."""

import ast
from collections.abc import Callable
from typing import Any, Protocol

from shpe.diagnostics import Diagnostics
from shpe.environment import Environment, Shape, ShapeState
from shpe.extractor import Extractor


class HandlerResolver(Protocol):
    def shape(self, node: ast.AST | None) -> Shape | ShapeState | None: ...


class BaseHandler:
    """Shared dependencies for callable operation handlers."""

    def __init__(
        self,
        extractor: Extractor,
        diagnostics: Diagnostics,
        env: Environment,
        resolver: HandlerResolver,
    ) -> None:
        self.extractor = extractor
        self.diagnostics = diagnostics
        self.env = env
        self.resolver: HandlerResolver = resolver


def map_handlers(handler_mappings: list[tuple]) -> dict[Any, Callable]:
    return {op: handler for handler, ops in handler_mappings for op in ops}


def infer_call_target(
    node: ast.Call, resolver: HandlerResolver, env: Environment
) -> tuple[Shape | ShapeState | None, list[ast.AST]]:
    """Return the receiver shape and operation arguments for method or NumPy calls."""
    if not isinstance(node.func, ast.Attribute):
        return None, list(node.args)

    # is module function call
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and env.is_module_alias(node.func.value.id)
    ):
        return resolver.shape(node.args[0]), list(node.args[1:])

    return resolver.shape(node.func.value), list(node.args)


def broadcast_shapes(shape1: Shape, shape2: Shape) -> Shape | None:
    dims1, dims2 = shape1.dims, shape2.dims
    len1, len2 = len(dims1), len(dims2)
    max_len = max(len1, len2)
    p1 = (1,) * (max_len - len1) + dims1
    p2 = (1,) * (max_len - len2) + dims2
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
    return Shape(tuple(result))
