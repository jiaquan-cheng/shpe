import pytest

OBJECT_CASES = [
    pytest.param(
        """
        import numpy as np

        class Batch:
            def __init__(self, data, steps):
                self.data = data
                self.steps = steps

        values = np.zeros((2, 3))
        batch = Batch(values, 3)
        """,
        id="object_constructor_fields",
    ),
    pytest.param(
        """
        import numpy as np

        class Batch:
            def __init__(self, data, steps):
                self.data = data
                self.steps = steps

            def add_batch_axis(self):
                self.data = np.expand_dims(self.data, axis=0)

            def set_steps(self, steps):
                self.steps = steps

            def get_data(self):
                return self.data

        batch = Batch(np.zeros((2, 3)), 3)
        batch.add_batch_axis()
        batch.set_steps(5)
        result = batch.get_data()
        """,
        id="object_method_updates_tensor_field",
    ),
    pytest.param(
        """
        import numpy as np

        class Config:
            def __init__(self, rows):
                self.rows = rows

            def make_tensor(self):
                return np.zeros((self.rows, 4))

        config = Config(3)
        result = config.make_tensor()
        """,
        id="object_scalar_field_used_as_tensor_dimension",
    ),
    pytest.param(
        """
        import numpy as np

        class Batch:
            def __init__(self, data, steps):
                self.data = data
                self.steps = steps

            def add_batch_axis(self):
                self.data = np.expand_dims(self.data, axis=0)

        small = Batch(np.zeros((2, 3)), 2)
        large = Batch(np.zeros((4, 5)), 4)
        alias = small
        alias.add_batch_axis()
        """,
        id="object_fields_isolated_between_instances",
    ),
]
