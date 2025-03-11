import numpy as np

x = [[1,2],[2,3],[3,4],[4,5],[6,7],[7,8],[8,9]]
y = [3,5,7,9,13,15,17]

def mean(val):
    return sum(val) / len(val)

def covariance(x, y):
    x_mean, y_mean = mean(x), mean(y)
    return sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(len(x)))

def variance(val):
    val_mean = mean(val)
    return sum((v - val_mean) ** 2 for v in val)

def coefficient(X, y):
    m1 = []
    for i in range(len(X[0])):
        X_I = [row[i] for row in X]
        m = covariance(X_I, y) / variance(X_I)
        m1.append(m)
    b = mean(y) - sum(m1[i] * mean([row[i] for row in X]) for i in range(len(m1)))
    return [b] + m1

def predict(x_new, b):
    return b[0] + sum(b[i + 1] * x_new[i] for i in range(len(x_new)))

b = coefficient(x, y)
x_test = np.array([1,2])

print(predict(x_test, b))