import numpy as np

arr = np.array([[1, 2], [3, 4], [5, 6]]).reshape((2, 3))


class MyClass:
    def __init__(self, data: np.ndarray):
        self.data: np.ndarray = data

    def expand_dims(self):
        self.data = np.expand_dims(self.data, axis=0)

    def get_data(self):
        return self.data


if __name__ == "__main__":
    my_instance = MyClass(arr)
    my_instance.expand_dims()
    result = my_instance.get_data()
