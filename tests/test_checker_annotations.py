import ast
from textwrap import dedent

import pytest

from shpe.checker import Checker
from tests.checker_cases import ANNOTATION_CASES


@pytest.mark.parametrize(
    "code, expected_codes",
    ANNOTATION_CASES,
)
def test_checker_annotations(code: str, expected_codes: list[str]) -> None:
    tree = ast.parse(dedent(code).strip())

    checker = Checker()
    checker.visit(tree)

    assert [error["code"] for error in checker.errors] == expected_codes
