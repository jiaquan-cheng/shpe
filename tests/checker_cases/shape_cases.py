import pytest

SHAPE_CASES = [
    pytest.param(
        """
        a = np.full((2, 3), 5)
        at = a.T
        """,
        id="transpose_variable",
    ),
    pytest.param(
        """
        b = np.full((2, 3), 5).T
        """,
        id="transpose_inline",
    ),
    pytest.param(
        """
        a = np.ones((2, 3))
        b = a.reshape(6)
        """,
        id="reshape_method_args",
    ),
    pytest.param(
        """
        a = np.ones((2, 3))
        c = a.reshape((3, 2))
        """,
        id="reshape_method_tuple",
    ),
    pytest.param(
        """
        d = np.full((4, 2, 2), 5).reshape((2, 8))
        """,
        id="reshape_inline_method",
    ),
    pytest.param(
        """
        d = np.full((4, 2, 2), 5)
        f = np.reshape(d, (8, -1))
        """,
        id="reshape_np_infer_dim",
    ),
    pytest.param(
        """
        d = np.full((4, 2, 2), 5)
        g = np.reshape(d, (2, 2, -1, 2))
        """,
        id="reshape_np_multiple_dims",
    ),
    pytest.param(
        """
        a = np.ones((2, 3))
        h = np.reshape(a, (-1, 2))
        """,
        id="reshape_np_infer_first",
    ),
    pytest.param(
        """
        a = np.ones((2, 3))
        b = a.reshape(7)
        """,
        id="reshape_mismatch_size",
    ),
    pytest.param(
        """
        a = np.ones((2, 3))
        c = a.reshape((4, 2))
        """,
        id="reshape_mismatch_tuple",
    ),
    pytest.param(
        """
        a = np.ones((2, 3))
        d = np.reshape(a, (-1, -1))
        """,
        id="reshape_mismatch_multiple_infer",
    ),
    pytest.param(
        """
        a = np.ones((2, 3))
        b = a.flatten()
        """,
        id="flatten_method",
    ),
    pytest.param(
        """
        a = np.ones((2, 3))
        b = a.ravel()
        """,
        id="ravel_method",
    ),
    pytest.param(
        """
        a = np.ones((2, 3))
        c = np.ravel(a)
        """,
        id="ravel_np",
    ),
    pytest.param(
        """
        a = np.ones((1, 2, 1, 3))
        b = a.squeeze()
        """,
        id="squeeze_method_all",
    ),
    pytest.param(
        """
        a = np.ones((1, 2, 1, 3))
        c = np.squeeze(a)
        """,
        id="squeeze_np_all",
    ),
    pytest.param(
        """
        a = np.ones((1, 2, 1, 3))
        d = a.squeeze(axis=0)
        """,
        id="squeeze_axis_int",
    ),
    pytest.param(
        """
        a = np.ones((1, 2, 1, 3))
        e = a.squeeze(axis=2)
        """,
        id="squeeze_axis_middle",
    ),
    pytest.param(
        """
        a = np.ones((1, 2, 1, 3))
        f = a.squeeze(axis=(0, 2))
        """,
        id="squeeze_axis_tuple",
    ),
    pytest.param(
        """
        a = np.ones((1, 2, 1, 3))
        g = a.squeeze(axis=-2)
        """,
        id="squeeze_axis_negative",
    ),
    pytest.param(
        """
        a = np.ones((2, 3))
        c = np.expand_dims(a, 2)
        """,
        id="expand_dims_np_int",
    ),
    pytest.param(
        """
        a = np.ones((2, 3))
        f = np.expand_dims(a, axis=2)
        """,
        id="expand_dims_np_axis",
    ),
    pytest.param(
        """
        a = np.ones((2, 3, 4))
        b = a.swapaxes(0, 2)
        """,
        id="swapaxes_method",
    ),
    pytest.param(
        """
        a = np.ones((2, 3, 4))
        c = np.swapaxes(a, 1, 2)
        """,
        id="swapaxes_np",
    ),
    pytest.param(
        """
        a = np.ones((2, 3, 4))
        e = np.swapaxes(a, -3, -1)
        """,
        id="swapaxes_negative",
    ),
    pytest.param(
        """
        a = np.ones((2, 3))
        b = a.resize((4, 1))
        """,
        id="resize_method_tuple",
    ),
    pytest.param(
        """
        a = np.ones((2, 3))
        c = np.resize(a, (3, 2))
        """,
        id="resize_np_tuple",
    ),
]
