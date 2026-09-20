import pytest

CONTEXT_CASES = [
    pytest.param(
        """
        a = np.array([[1, 2], [3, 4]])
        b = a
        """,
        id="context_variable_propagation",
    ),
    pytest.param(
        """
        a = np.ones((5, 5, 5))
        step_slice = a[0:5:2, :, ::2]
        """,
        id="context_slicing_step",
    ),
    pytest.param(
        """
        a = np.ones((5, 5, 5))
        scalar_multi = a[1, 2]
        """,
        id="context_slicing_scalar",
    ),
    pytest.param(
        """
        a = np.ones((5, 5, 5))
        ellipsis_slice = a[..., 0]
        """,
        id="context_slicing_ellipsis",
    ),
    pytest.param(
        """
        a = np.ones((5, 5, 5))
        negative_slice = a[-2:, :, 0]
        """,
        id="context_slicing_negative_slice",
    ),
    pytest.param(
        """
        a = np.ones((5, 5, 5))
        negative_index = a[-1]
        """,
        id="context_slicing_negative_index",
    ),
    pytest.param(
        """
        a = np.ones((5, 5, 5))
        reverse_slice = a[::-1, ::-1, ::-1]
        """,
        id="context_slicing_reverse",
    ),
    pytest.param(
        """
        global_var = np.zeros((2, 2))
        b = np.ones((3, 3))
        def unannotated_func():
            b = np.ones((2, 2))
            return global_var + b
        res = unannotated_func()
        """,
        id="context_func_scope_success",
    ),
    pytest.param(
        """
        global_var = np.zeros((2, 2))
        def unannotated_func():
            a = np.ones((5, 5))
            b = global_var + a
            return a
        res = unannotated_func()
        """,
        id="context_func_scope_mismatch",
    ),
    pytest.param(
        """
        def func(x=np.zeros((2, 3))):
            return x
        res = func()
        """,
        id="context_func_default_arg_inferred",
    ),
    pytest.param(
        """
        def func(x=np.zeros((2, 3))):
            return x
        res = func(np.zeros((4, 4)))
        """,
        id="context_func_default_arg_overridden",
    ),
    pytest.param(
        """
        def func(i=2):
            return np.zeros((i, 3))
        res = func()
        """,
        id="context_func_default_arg_value",
    ),
    pytest.param(
        """
        def func(y):
            def inner_func(x):
                return x
            return inner_func(y)
        res = func(np.zeros((4, 4)))
        """,
        id="nested_func_def",
    ),
    pytest.param(
        """
        def func():
            def inner_func(x=np.zeros((2, 3))):
                return x
            return inner_func()
        res = func()
        """,
        id="nested_func_default_arg",
    ),
]
