import ast
import traceback
from textwrap import dedent

import numpy as np
import pytest
import torch

from shpe.checker import Checker
from shpe.environment import ScalarValue, Shape, Tensor
from tests.checker_cases import (
    CONTEXT_CASES,
    CREATION_CASES,
    MATH_CASES,
    OBJECT_CASES,
    SHAPE_CASES,
)


def assert_runtime_value(
    code: str, label: str, runtime_value: object, static_value: object | None
) -> None:
    if hasattr(runtime_value, "shape"):
        assert isinstance(static_value, Tensor), (
            f"Checker did not infer tensor value '{label}'.\nSnippet:\n{code}"
        )
        assert isinstance(static_value.shape, Shape), (
            f"Checker could not infer the shape of '{label}'.\nSnippet:\n{code}"
        )
        actual_shape = tuple(runtime_value.shape)
        assert static_value.shape.dims == actual_shape, (
            f"  Snippet:\n{code}\n"
            f"  Value:    {label}\n"
            f"  Predicted: {static_value.shape.dims}\n"
            f"  Actual:    {actual_shape}"
        )
    elif isinstance(runtime_value, (int, float)) and not isinstance(
        runtime_value, bool
    ):
        assert isinstance(static_value, ScalarValue), (
            f"Checker did not track scalar value '{label}'.\nSnippet:\n{code}"
        )
        assert static_value.value == runtime_value, (
            f"Checker inferred the wrong value for '{label}': "
            f"predicted {static_value.value}, actual {runtime_value}."
        )


def checker_runtime_oracle(code: str) -> None:
    """Test the checker against a given code snippet."""
    cleaned_code = dedent(code).strip()

    tree = ast.parse(cleaned_code)

    checker = Checker()
    checker.visit(tree)

    runtime_namespace = {"np": np, "torch": torch}
    try:
        exec(cleaned_code, runtime_namespace)
    except Exception as e:
        tb = traceback.extract_tb(e.__traceback__)
        error_lines = {f.lineno for f in tb if f.filename == "<string>"}

        # We only check if we caught one of all runtime errors
        # because the same error can be raised at different lines
        matched = any(err.get("line") in error_lines for err in checker.errors)

        # If we didn't match any errors, we assert that there are warnings instead
        assert matched, (
            f"  Snippet:\n{cleaned_code}\n"
            f"  Exception at lines {error_lines}: {e}\n"
            f"  Checker errors: {checker.errors}"
        )
        return
    if checker.errors:
        pytest.fail(
            "  Checker found errors but no runtime exception was raised.\n"
            f"  Snippet:\n{cleaned_code}\n"
            f"  Checker errors: {checker.errors}"
        )

    class_names = {
        node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
    }
    for name, runtime_value in runtime_namespace.items():
        if name in checker.shapes and hasattr(runtime_value, "shape"):
            assert_runtime_value(
                cleaned_code, name, runtime_value, checker.shapes[name]
            )

        if type(runtime_value).__name__ not in class_names:
            continue

        abstract_object = checker.env.get_object(name)
        assert abstract_object is not None, (
            f"Checker did not track object '{name}' of class "
            f"'{type(runtime_value).__name__}'.\nSnippet:\n{cleaned_code}"
        )
        abstract_attributes = abstract_object.attributes
        assert isinstance(abstract_attributes, dict), (
            f"Checker did not track attributes for object '{name}'.\n"
            f"Snippet:\n{cleaned_code}"
        )
        for attribute, value in vars(runtime_value).items():
            assert_runtime_value(
                cleaned_code,
                f"{name}.{attribute}",
                value,
                abstract_attributes.get(attribute),
            )


@pytest.mark.parametrize(
    "code",
    CREATION_CASES,
)
def test_checker_creation(code: str) -> None:
    checker_runtime_oracle(code)


@pytest.mark.parametrize(
    "code",
    CONTEXT_CASES,
)
def test_checker_context(code: str) -> None:
    checker_runtime_oracle(code)


@pytest.mark.parametrize(
    "code",
    MATH_CASES,
)
def test_checker_math(code: str) -> None:
    checker_runtime_oracle(code)


@pytest.mark.parametrize(
    "code",
    SHAPE_CASES,
)
def test_checker_shape(code: str) -> None:
    checker_runtime_oracle(code)


@pytest.mark.parametrize("code", OBJECT_CASES)
def test_checker_custom_objects(code: str) -> None:
    checker_runtime_oracle(code)
