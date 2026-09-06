from typing import Annotated

import numpy as np

a = np.ones((3, 2))
b = np.zeros((2, 3))
c: Annotated[np.ndarray, (3, 2)] = a.T
d: Annotated[np.ndarray, (2, 3)] = b.T  # shpy: ignore
err_elem = a + b
err_matmul = a @ b.T
