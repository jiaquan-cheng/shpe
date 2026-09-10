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


@pytest.mark.parametrize(
    "code",
    TEST_CASES,
)
def test_checker_runtime_oracle(code: str) -> None:

    cleaned_code = dedent(code).strip()

    tree = ast.parse(cleaned_code)

    body_code = ""
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

        assert matched, (
            f"  Snippet:\n{body_code}\n"
            f"  Exception at lines {error_lines}: {e}\n"
            f"  Checker errors: {checker.errors}"
        )
        return

    for var, static_shape in checker.shapes.items():
        if var in runtime_namespace:
            runtime_val = runtime_namespace[var]
            if hasattr(runtime_val, "shape"):
                assert static_shape == runtime_val.shape, (
                    f"  Snippet:\n{body_code}\n"
                    f"  Predicted: {static_shape}\n"
                    f"  Actual:    {runtime_val.shape}\n"
                    f"  Table:     {checker.shapes}"
                )
