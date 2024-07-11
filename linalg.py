import numpy as np

# Create vectors
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Create matrices
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

# Addition of vectors
sum_vector = a + b
print(f"Sum of vectors a and b: {sum_vector}")

# Subtraction of vectors
diff_vector = a - b
print(f"Difference of vectors a and b: {diff_vector}")

# Addition of matrices
sum_matrix = A + B
print(f"Sum of matrices A and B: {sum_matrix}")

# Subtraction of matrices
diff_matrix = A - B
print(f"Difference of matrices A and B: {diff_matrix}")

# Dot product of vectors
dot_product = np.dot(a, b)
print(f"Dot product of vectors a and b: {dot_product}")

# Create matrices
C = np.array([[1, 2, 3], [4, 5, 6]])
D = np.array([[7, 8, 9, 10], [11, 12, 13, 14], [15, 16, 17, 18]])

# Matrix multiplication
product_matrix = np.dot(C, D)
print(f"Product of matrices A and B: {product_matrix}")

# Create a vector
a = np.array([1, 1, 2])

# Calculate the magnitude of the vector
magnitude = np.linalg.norm(a)
print(f"Magnitude of vector a: {magnitude}")

# Transpose of a matrix
transpose_matrix = A.T
print(f"Transpose of matrix A: {transpose_matrix}")
