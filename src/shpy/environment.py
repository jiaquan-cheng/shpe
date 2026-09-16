from enum import Enum
from typing import Any


class ShapeState(Enum):
    UNKNOWN = "UNKNOWN"


class Environment:
    def __init__(self) -> None:
        shapes: dict[str, tuple[Any, ...] | ShapeState] = {}
        scalar_values: dict[str, int | float | ShapeState] = {}
        self.stack: list[dict[str, Any]] = [
            {"shapes": shapes, "scalar_values": scalar_values}
        ]

    @property
    def shapes(self) -> dict[str, tuple[Any, ...] | ShapeState]:
        return self.stack[-1]["shapes"]

    @property
    def scalar_values(self) -> dict[str, int | float | ShapeState]:
        return self.stack[-1]["scalar_values"]

    def __len__(self) -> int:
        return len(self.stack)

    def get_shape(self, name: str) -> tuple[Any, ...] | ShapeState | None:
        for frame in reversed(self.stack):
            if name in frame["shapes"]:
                return frame["shapes"][name]
        return None

    def get_scalar(self, name: str) -> int | float | ShapeState | None:
        for frame in reversed(self.stack):
            if name in frame["scalar_values"]:
                return frame["scalar_values"][name]
        return None

    def set_shape(self, name: str, shape: tuple[Any, ...] | ShapeState) -> None:
        self.stack[-1]["shapes"][name] = shape

    def set_scalar(self, name: str, value: int | float | ShapeState) -> None:
        self.stack[-1]["scalar_values"][name] = value

    def push_child(self) -> None:
        """Creates a new child environment for local variables and shapes."""
        shapes: dict[str, tuple[Any, ...] | ShapeState] = {}
        scalar_values: dict[str, int | float | ShapeState] = {}
        self.stack.append({"shapes": shapes, "scalar_values": scalar_values})

    def pop_child(self) -> None:
        """Removes the most recent child environment."""
        if len(self.stack) > 1:
            self.stack.pop()
        else:
            raise RuntimeError("Cannot pop the root environment.")
