import pytest

ANNOTATION_CASES = [
    pytest.param(
        """
        n = 5
        m = 3.0
        a: Annotated[np.ndarray, (n, m)] = np.zeros((5, 3))
        """,
        id="annotation_success_variable_shape",
    ),
    pytest.param(
        """
        n = 5
        b: Annotated[np.ndarray, (n,)] = np.ones(n)
        """,
        id="annotation_success_variable_1d",
    ),
    pytest.param(
        """
        f = 3.5
        c: Annotated[np.ndarray, (f,)] = np.ones(3)
        """,
        id="annotation_mismatch_float_scalar",
    ),
    pytest.param(
        """
        f = 3.5
        d: Annotated[np.ndarray, (3,)] = np.zeros((f, 2))
        """,
        id="annotation_mismatch_zeros_shape",
    ),
    pytest.param(
        """
        a: Annotated[np.ndarray, (9, 9)] = np.zeros((9, 9))
        def transform(
            x: Annotated[np.ndarray, (10, 20)],
        ) -> Annotated[np.ndarray, (20, 10)]:
            a = np.zeros((20, 10))
            return x.T
        b: Annotated[np.ndarray, (10, 20)] = np.zeros((10, 20))
        c: Annotated[np.ndarray, (20, 10)] = transform(b)
        """,
        id="annotation_function_transform",
    ),
]
