import ast
import traceback
from textwrap import dedent
from typing import Annotated

import numpy as np
import pytest

from shpy.checker import Checker
from tests.checker_cases import (
    ANNOTATION_CASES,
    CONTEXT_CASES,
    CREATION_CASES,
    MATH_CASES,
    SHAPE_CASES,
)

TEST_CASES = (
    CREATION_CASES + CONTEXT_CASES + ANNOTATION_CASES + MATH_CASES + SHAPE_CASES
)


def checker_runtime_oracle(code: str) -> None:
    """Test the checker against a given code snippet."""
    cleaned_code = dedent(code).strip()

    tree = ast.parse(cleaned_code)

    checker = Checker()
    checker.visit(tree)

    runtime_namespace = {"np": np, "Annotated": Annotated}
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

    for var, static_shape in checker.shapes.items():
        if var in runtime_namespace:
            runtime_val = runtime_namespace[var]
            if hasattr(runtime_val, "shape"):
                assert static_shape == runtime_val.shape, (
                    f"  Snippet:\n{cleaned_code}\n"
                    f"  Predicted: {static_shape}\n"
                    f"  Actual:    {runtime_val.shape}\n"
                    f"  Table:     {checker.shapes}"
                )


# @pytest.mark.parametrize(
#     "code",
#     ANNOTATION_CASES,
# )
# def test_checker_annotation(code: str) -> None:
#     checker_runtime_oracle(code)


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
