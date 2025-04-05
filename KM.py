import numpy as np

X = np.array([[1, 2], [1, 4], [1, 0],
              [10, 2], [10, 4], [10, 0]])

k = 2
max_iters = 10

indices = np.random.choice(len(X), k, replace=False)
centroids = X[indices]

for _ in range(max_iters):
    distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)
    labels = np.argmin(distances, axis=1)

    new_centroids = []
    for i in range(k):
        cluster_points = X[labels == i]
        new_centroids.append(cluster_points.mean(axis=0))
    new_centroids = np.array(new_centroids)

    if np.allclose(centroids, new_centroids):
        break
    centroids = new_centroids

print(centroids)
print(labels)
