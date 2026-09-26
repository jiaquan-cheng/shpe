import pytest

CREATION_CASES = [
    pytest.param(
        """
        lst = [[1, 2], [3, 4]]
        arr = np.array(lst)
        """,
        id="list_to_nparray",
    ),
    pytest.param(
        """
        tpl = ((1, 2), (3, 4))
        arr = np.array(tpl)
        """,
        id="tuple_to_nparray",
    ),
    pytest.param(
        """
        a = np.array([[1, 2], [3, 4]])
        """,
        id="implicit_array_inference_array",
    ),
    pytest.param(
        """
        b = np.zeros((1, 1, 1, 1))
        """,
        id="implicit_array_inference_zeros",
    ),
    pytest.param(
        """
        c = np.ones((4, 2, 4))
        """,
        id="implicit_array_inference_ones",
    ),
    pytest.param(
        """
        d = np.empty((6, 3))
        """,
        id="implicit_array_inference_empty",
    ),
    pytest.param(
        """
        e = np.full((2, 3), 5)
        """,
        id="implicit_array_inference_full",
    ),
    pytest.param(
        """
        a = np.zeros(shape=(2, 3))
        b = np.full(shape=(4, 5), fill_value=1)
        """,
        id="keyword_shape_creation",
    ),
    pytest.param(
        """
        a = np.array(5)
        b = np.array([])
        """,
        id="scalar_and_empty_array_creation",
    ),
    pytest.param(
        """
        a: float = 2
        b: int = 3
        c = np.full((a, b), 5)
        """,
        id="variable_shape_creation_full",
    ),
    pytest.param(
        """
        a: float = 2
        b: int = 3
        d = np.zeros((a, b))
        """,
        id="variable_shape_creation_zeros",
    ),
    pytest.param(
        """
        a = np.random.rand(3, 2)
        """,
        id="random_generation_rand",
    ),
    pytest.param(
        """
        b = np.random.randn(5)
        """,
        id="random_generation_randn",
    ),
    pytest.param(
        """
        c = np.random.random((2, 2, 2))
        """,
        id="random_generation_random",
    ),
    pytest.param(
        """
        a = np.random.random(size=(2, 3))
        """,
        id="random_generation_keyword_size",
    ),
    pytest.param(
        """
        d = np.random.random_sample((2, 2, 2))
        """,
        id="random_generation_random_sample",
    ),
    pytest.param(
        """
        e = np.random.ranf((2, 2, 2))
        """,
        id="random_generation_ranf",
    ),
    pytest.param(
        """
        f = np.random.sample((2, 2, 2))
        """,
        id="random_generation_sample",
    ),
    pytest.param(
        """
        g = np.random.randint(0, 10, size=(10, 4))
        """,
        id="random_generation_randint",
    ),
    pytest.param(
        """
        h = np.random.uniform(0, 1, size=(3, 3))
        """,
        id="random_generation_uniform",
    ),
    pytest.param(
        """
        i = np.random.normal(0, 1, size=(4, 4))
        """,
        id="random_generation_normal",
    ),
    pytest.param(
        """
        seq = np.arange(0, 10, 1)
        """,
        id="sequence_arange",
    ),
    pytest.param(
        """
        space = np.linspace(0, 1, 5)
        """,
        id="sequence_linspace",
    ),
    pytest.param(
        """
        log_space = np.logspace(0, 2, 50)
        """,
        id="sequence_logspace",
    ),
    pytest.param(
        """
        geom_space = np.geomspace(1, 1000, 10)
        """,
        id="sequence_geomspace",
    ),
    pytest.param(
        """
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        grid = np.meshgrid(x, y)
        """,
        id="sequence_meshgrid",
    ),
    pytest.param(
        """
        x = np.array([1, 2, 3])
        y = np.array([4, 5])
        first = np.meshgrid(x, y)[0]
        """,
        id="sequence_meshgrid_indexed_result",
    ),
]
