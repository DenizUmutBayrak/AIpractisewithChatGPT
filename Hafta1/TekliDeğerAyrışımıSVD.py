import numpy as np

matrix = np.array([[4, -2], [1, 1]])

U, s, Vt = np.linalg.svd(matrix)
print(f"U: {U}")
print(f"S: {s}")
print(f"Vt: {Vt}")