import pytest

MATH_CASES = [
    pytest.param(
        """
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        c = np.array([[9, 10], [11, 12]])
        d = a + b - c
        """,
        id="binary_operation_add_sub",
    ),
    pytest.param(
        """
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6], [7, 8]])
        c = np.array([[9, 10], [11, 12]])
        e = a / b * c
        """,
        id="binary_operation_div_mul",
    ),
    pytest.param(
        """
        a = np.array([[1, 2], [3, 4]])
        f: float = 2
        g = f * a
        """,
        id="binary_operation_scalar_mul",
    ),
    pytest.param(
        """
        h = np.ones((2, 1)) + np.ones((1, 3))
        """,
        id="binary_operation_broadcasting",
    ),
    pytest.param(
        """
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6, 7], [8, 9, 10]])
        c = a + b
        """,
        id="binary_operation_mismatch_add",
    ),
    pytest.param(
        """
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6, 7], [8, 9, 10]])
        d = a - b
        """,
        id="binary_operation_mismatch_sub",
    ),
    pytest.param(
        """
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6, 7], [8, 9, 10]])
        e = a * b
        """,
        id="binary_operation_mismatch_mul",
    ),
    pytest.param(
        """
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6, 7], [8, 9, 10]])
        f = a / b
        """,
        id="binary_operation_mismatch_div",
    ),
    pytest.param(
        """
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6, 7], [8, 9, 10]])
        c = a @ b
        """,
        id="matrix_multiplication",
    ),
    pytest.param(
        """
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6, 7], [8, 9, 10], [11, 12, 13]])
        c = a @ b
        """,
        id="matrix_multiplication_mismatch",
    ),
    pytest.param(
        """
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[5, 6, 7], [8, 9, 10]])
        c = np.array([[11, 12, 13], [14, 15, 16], [17, 18, 19]])
        d = a @ b @ c - b
        """,
        id="matrix_operation_chain",
    ),
    pytest.param(
        """
        a = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        b = np.array([[[9, 10, 11], [12, 13, 14]], [[15, 16, 17], [18, 19, 20]]])
        c = a @ b
        """,
        id="matrix_multiplication_multi_dimensions",
    ),
    pytest.param(
        """
        a: float = 2
        c = np.array([[1, 2, 3], [4, 5, 6]])
        d = a * c
        """,
        id="scalar_multiplication_float",
    ),
    pytest.param(
        """
        b: int = 3
        c = np.array([[1, 2, 3], [4, 5, 6]])
        e = b * c
        """,
        id="scalar_multiplication_int",
    ),
    pytest.param(
        """
        a = np.ones((2, 3, 4))
        b = np.sum(a, axis=0)
        """,
        id="reduction_np_sum_axis0",
    ),
    pytest.param(
        """
        a = np.ones((2, 3, 4))
        c = np.mean(a, axis=1)
        """,
        id="reduction_np_mean_axis1",
    ),
    pytest.param(
        """
        a = np.ones((2, 3, 4))
        d = np.prod(a, axis=-1)
        """,
        id="reduction_np_prod_axis_neg1",
    ),
    pytest.param(
        """
        a = np.ones((2, 3, 4))
        e = a.sum(axis=0)
        """,
        id="reduction_method_sum_axis0",
    ),
    pytest.param(
        """
        a = np.ones((2, 3, 4))
        f = a.mean(axis=1)
        """,
        id="reduction_method_mean_axis1",
    ),
    pytest.param(
        """
        a = np.ones((2, 3, 4))
        g = np.sum(a)
        """,
        id="reduction_np_sum_scalar",
    ),
    pytest.param(
        """
        a = np.ones((2, 3, 4))
        h = np.sum(a, axis=(1, 2))
        """,
        id="reduction_np_sum_tuple_axis",
    ),
    pytest.param(
        """
        a = np.ones((2, 3, 4))
        i = np.mean(a, axis=-3)
        """,
        id="reduction_np_mean_axis_neg3",
    ),
    pytest.param(
        """
        a = np.ones((2, 3, 4))
        b = np.sum(a, axis=3)
        """,
        id="reduction_np_sum_invalid_axis",
    ),
]
