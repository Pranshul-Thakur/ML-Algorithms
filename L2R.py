import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7])
y = np.array([2, 3, 4, 5, 6, 7, 8])

penalty = 0.1

def mean(val):
    return sum(val) / len(val)

def covariance(x, y):
    x_mean, y_mean = mean(x), mean(y)
    return sum((x[i] - x_mean)*(y[i] - y_mean) for i in range(len(x))) / len(x)

def variance(val):
    val_mean = mean(val)
    return sum((x - val_mean) ** 2 for x in val) / len(val)

m = covariance(x, y) / (variance(x) + penalty * np.sum(np.abs(x)))
b = mean(y) - m * mean(x)

def predict(x_new):
    return b + m * x_new

x_test = 5
print(predict(x_test))