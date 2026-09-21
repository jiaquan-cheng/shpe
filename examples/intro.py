from typing import Annotated

import numpy as np

a = np.array([[1, 2], [3, 4], [5, 6]])


def custom_function(i=2):
    a = np.zeros((i, 2))
    return a


if __name__ == "__main__":
    b = custom_function()
    c = a + b
    d = a @ b
    e = d @ a
    f: Annotated[np.ndarray, (3, 2)] = a.T
    f: Annotated[np.ndarray, (3, 2)] = a.T  # shpy: ignore
