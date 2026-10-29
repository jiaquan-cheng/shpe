from __future__ import annotations

import ast
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TypedDict, overload


@dataclass
class Shape(Sequence[int]):
    dims: tuple[int, ...]

    def __len__(self) -> int:
        return len(self.dims)

    @overload
    def __getitem__(self, index: int) -> int: ...

    @overload
    def __getitem__(
        self, index: slice[int | None, int | None, int | None]
    ) -> tuple[int, ...]: ...

    def __getitem__(
        self, index: int | slice[int | None, int | None, int | None]
    ) -> int | tuple[int, ...]:
        return self.dims[index]

    def __iter__(self) -> Iterator[int]:
        return iter(self.dims)

    def __str__(self) -> str:
        return str(self.dims)


class ShapeState(Enum):
    UNKNOWN = "UNKNOWN"
    UNDEFINED = "UNDEFINED"


@dataclass
class Tensor:
    shape: Shape | ShapeState


class ValueState(Enum):
    UNKNOWN = "UNKNOWN"
    UNDEFINED = "UNDEFINED"


@dataclass
class ScalarValue:
    value: int | float | ValueState


@dataclass
class AbstractObject:
    name: str
    class_def: ast.ClassDef


class EnvironmentFrame(TypedDict):
    shapes: dict[str, Tensor]
    scalar_values: dict[str, ScalarValue]
    objects: dict[str, AbstractObject]
    functions: dict[str, ast.FunctionDef]
    modules: dict[str, str]
    classes: dict[str, ast.ClassDef]


class Environment:
    def __init__(self) -> None:
        self.stack: list[EnvironmentFrame] = []
        self.push_child()  # Initialize the root environment

    @property
    def shapes(self) -> dict[str, Tensor]:
        return self.stack[-1]["shapes"]

    @property
    def scalar_values(self) -> dict[str, ScalarValue]:
        return self.stack[-1]["scalar_values"]

    @property
    def objects(self) -> dict[str, AbstractObject]:
        return self.stack[-1]["objects"]

    @property
    def functions(self) -> dict[str, ast.FunctionDef]:
        return self.stack[-1]["functions"]

    @property
    def modules(self) -> dict[str, str]:
        return self.stack[-1]["modules"]

    @property
    def classes(self) -> dict[str, ast.ClassDef]:
        return self.stack[-1]["classes"]

    def __len__(self) -> int:
        return len(self.stack)

    def get_shape(self, name: str) -> Shape | ShapeState | None:
        for frame in reversed(self.stack):
            tensor = frame["shapes"].get(name)
            if tensor is not None:
                return tensor.shape
        return None

    def get_scalar(self, name: str) -> ScalarValue | None:
        for frame in reversed(self.stack):
            value = frame["scalar_values"].get(name)
            if value is not None:
                return value
        return None

    def get_function(self, name: str) -> ast.FunctionDef | None:
        for frame in reversed(self.stack):
            function = frame["functions"].get(name)
            if function is not None:
                return function
        return None

    def get_module(self, name: str) -> str | None:
        for frame in reversed(self.stack):
            module = frame["modules"].get(name)
            if module is not None:
                return module
        return None

    def get_class(self, name: str) -> ast.ClassDef | None:
        for frame in reversed(self.stack):
            class_def = frame["classes"].get(name)
            if class_def is not None:
                return class_def
        return None

    def get_object(self, name: str) -> AbstractObject | None:
        for frame in reversed(self.stack):
            abstract_object = frame["objects"].get(name)
            if abstract_object is not None:
                return abstract_object
        return None

    def set_shape(self, name: str, shape: Shape | ShapeState | Tensor) -> None:
        tensor = shape if isinstance(shape, Tensor) else Tensor(shape)
        self.stack[-1]["shapes"][name] = tensor

    def set_scalar(
        self, name: str, value: int | float | ValueState | ScalarValue
    ) -> None:
        scalar = value if isinstance(value, ScalarValue) else ScalarValue(value)
        self.stack[-1]["scalar_values"][name] = scalar

    def set_function(self, name: str, func: ast.FunctionDef) -> None:
        self.stack[-1]["functions"][name] = func

    def set_module(self, name: str, module: str) -> None:
        self.stack[-1]["modules"][name] = module

    def set_class(self, name: str, class_def: ast.ClassDef) -> None:
        """Sets a class definition in the current environment."""
        self.stack[-1]["classes"][name] = class_def

    def set_object(self, name: str, abstract_object: AbstractObject) -> None:
        self.stack[-1]["objects"][name] = abstract_object

    def push_child(self) -> None:
        """Creates a new child environment for local variables and shapes."""
        frame: EnvironmentFrame = {
            "shapes": {},
            "scalar_values": {},
            "objects": {},
            "functions": {},
            "modules": {},
            "classes": {},
        }
        self.stack.append(frame)

    def pop_child(self) -> None:
        """Removes the most recent child environment."""
        if len(self.stack) > 1:
            self.stack.pop()
        else:
            raise RuntimeError("Cannot pop the root environment.")
