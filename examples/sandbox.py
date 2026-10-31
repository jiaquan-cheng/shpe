import numpy as np


class Batch:
    def __init__(self, data, steps):
        self.data = data
        self.steps = steps


values = np.zeros((2, 3))
batch = Batch(values, 3)
