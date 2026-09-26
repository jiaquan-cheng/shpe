import pytest

ANNOTATION_CASES = [
    pytest.param(
        """
        n = 5
        m = 3.0
        a: Annotated[np.ndarray, (n, m)] = np.zeros((5, 3))
        """,
        [],
        id="annotation_success_variable_shape",
    ),
    pytest.param(
        """
        n = 5
        b: Annotated[np.ndarray, (n,)] = np.ones(n)
        """,
        [],
        id="annotation_success_variable_1d",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (3,)] = np.zeros((2,))
        """,
        ["Annotation"],
        id="annotation_mismatch_assignment",
    ),
    pytest.param(
        """
        def transform(
            x: Annotated[np.ndarray, (10, 20)],
        ) -> Annotated[np.ndarray, (20, 10)]:
            return x.T
        b = np.zeros((2, 3))
        c = transform(b)
        """,
        ["Annotation"],
        id="annotation_mismatch_argument",
    ),
    pytest.param(
        """
        def transform(
            x: Annotated[np.ndarray, (10, 20)],
        ) -> Annotated[np.ndarray, (20, 10)]:
            return x.T
        b = np.zeros((10, 20))
        c = transform(b)
        """,
        [],
        id="annotation_function_transform_success",
    ),
]
