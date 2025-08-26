from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np

x = np.array([[2, 0],
              [0, 2],
              [3, 1],
              [1, 3]])

standardizer = StandardScaler()
x_std = standardizer.fit_transform(x)

pca = PCA(n_components=1)
x_reduced = pca.fit_transform(x_std)


print("Reduced Data (1D):\n", x_reduced)