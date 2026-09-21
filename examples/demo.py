from typing import Annotated

import numpy as np

# examples of operations where shape inference is supported
# and where it is not supported

# Basic arrays creation
scalar_value = 5
float_scalar_value = 5.0
list = [[1, 2], [3, 4], [5, 6]]
tuple = ((1, 2), (3, 4), (5, 6))
nparray = np.array(list)
zeros = np.zeros((2, 3))
zeros_from_scalar_value = np.zeros((scalar_value, float_scalar_value))
full = np.full((3, 2), 5)
ones = np.ones((3, 2))
random = np.random.rand(3, 2)
variable_propogation = nparray

# math
elementwise_add = nparray + full
matrix_multiplication = nparray @ zeros
chained_multiplication = nparray @ zeros @ full

# shape manipulation
transpose = nparray.T
reshape = nparray.reshape((3, -1))
step_slice = nparray[0:3:2, :]
expanded = step_slice.expand_dims(1)
squeeze = expanded.squeeze()
swapped = reshape.swapaxes(0, 1)

# reductions
summed_axis = np.sum(swapped, axis=0)
mean_axis = swapped.mean(axis=1)


# functions
def custom_function(scalar_value=2):
    zeros_from_scalar_value = np.zeros((scalar_value, float_scalar_value))

    def inner_function():
        return zeros_from_scalar_value

    return inner_function()


function = custom_function()


# recursive function not supported by shape inference
def recursive_function(n):
    if n <= 0:
        return np.array([1])
    else:
        return np.array([n]) + recursive_function(n - 1)


recursive = recursive_function(3)

# not supported control flow
if True:
    nparray = np.array([1, 2, 3])
else:
    list = [[1, 2], [3, 4], [5, 6]]

while zeros.shape[0] < 5:
    zeros = np.zeros((10, 3))

for _ in range(3):
    ones = np.ones((3, 2))

# any variables touched will be marked as unknown
nparray = nparray
list = list
zeros = zeros
ones = ones

working_inference = np.full((3, 2), 5)
working_inference = working_inference

# unkown will be propogated
inference_breaks = ones + working_inference

# using Annotated, you can reintroduce shapes
# whenever unsupported operation or control flow is used,
# so the inference can still work downstream:
inference_repaired: Annotated[np.ndarray, (3, 2)] = inference_breaks
inference_repaired = inference_repaired

# combined with # shpy: ignore, you can ignore false errors
# and replace them with the correct shape

# main function will still be inferred:
if __name__ == "__main__":
    a = custom_function()
