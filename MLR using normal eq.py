import numpy as np

x = np.array([[1,2],[2,3],[3,4],[4,5],[6,7],[7,8],[8,9]])
y = np.array([3,5,7,9,13,15,17])

b = np.c_[np.ones((x.shape[0], 1)), x]

theta = np.linalg.pinv(b.T @ b) @ b.T @ y

y_pred = b @ theta

print(y_pred)