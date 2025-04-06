import numpy as np

a = [[2, 1], [1, 2]]

aT = a.T @ a

trace = np.trace(aT)
det = np.linalg.det(aT)

discriminant = np.sqrt(trace**2 - 4 * det) # λ² - (trace)·λ + det = 0
eigen1 = np.sqrt((trace + discriminant) / 2)
eigen2 = np.sqrt((trace - discriminant) / 2)

print (eigen1, eigen2)

#returning singular values of SVD not the whole thing