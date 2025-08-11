import numpy as np

A = np.array([[2, 1],[5, 7]])
b = np.array([11, 13])

A_inv = np.linalg.inv(A)

x = np.dot(A_inv, b)
print(x)

