import numpy as np

x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([3, 5, 7, 9])

penalty = 0.1
columns = x.shape[1]
I = np.eye(columns)

w = np.linalg.inv(x.T @ x + penalty * I) @ x.T @ y
y_pred = x @ w

print(w) # ridge coefficients
print(y_pred)