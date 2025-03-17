import numpy as np

x = np.array([[1, 2], [2, 3], [3, 4], [4, 5]])
y = np.array([3, 5, 7, 9])

penalty = 0.1
columns = x.shape[1]
I = np.eye(columns)

B = np.linalg.inv(x.T @ x) @ x.T @ y
w = np.sign(B) * (np.abs(B) - penalty)
y_pred = x @ w

print(w) # lasso coefficients
print(y_pred)

# closed form cant be achieved with L1 due to the non differentiable nature of L1 penalty term