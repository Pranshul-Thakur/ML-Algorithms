import numpy as np

v = [3,4]
L = [1,0]

l_normalised = L / np.linalg.norm(L)

orthognal_projection_of_vector = np.dot(l_normalised, v) * l_normalised

print(orthognal_projection_of_vector)