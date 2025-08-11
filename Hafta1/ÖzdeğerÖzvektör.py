import numpy as np

matrix = np.array([[4, -2], [1, 1]])

eigenvalues, eigenvectors = np.linalg.eig(matrix)
print(f"Eigenvalue = {eigenvalues}")
print(f"Eigenvector = {eigenvectors}")