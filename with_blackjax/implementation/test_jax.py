from jax import vmap
import numpy as np

def sum_row(w):
    return sum(w)

def square(w):
    return w*w

a = np.array([[1,2], [3,4]])

print(vmap(sum_row)(a))
print(vmap(square)(a))


param1 = [np.array([1,2]), np.array([3,4])]
param2 = [np.array([1,]), np.array([3,])]
particles = [param1, param2]

print(vmap(square)(particles))