"""Shared helpers used by shape operation handlers."""

import ast
from collections.abc import Callable
from typing import Any, Protocol

from shpe.diagnostics import Diagnostics
from shpe.environment import Environment, ShapeState
from shpe.extractor import Extractor


class HandlerResolver(Protocol):
    def shape(
        self, node: ast.AST | None
    ) -> tuple[int | str, ...] | ShapeState | None: ...


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
) -> tuple[tuple[int | str, ...] | ShapeState | None, list[ast.AST]]:
    """Return the receiver shape and operation arguments for method or NumPy calls."""
    if not isinstance(node.func, ast.Attribute):
        return None, list(node.args)

    # is module function call
    if (
        isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id in env.modules.values()
    ):
        return resolver.shape(node.args[0]), list(node.args[1:])

    return resolver.shape(node.func.value), list(node.args)


def broadcast_shapes(
    shape1: tuple[int | str, ...], shape2: tuple[int | str, ...]
) -> tuple[int | str, ...] | None:
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
