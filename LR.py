from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.model_selection import train_test_split
import numpy as np

x = [1,2,3,4,5,6,7]
y = [2,3,4,5,6,7,8]

def mean(val):
    return sum(val) / len(val)

def covariance(x, y):
    
    x_mean, y_mean = mean(x), mean(y)
    return sum((x[i] - x_mean) * (y[i] - y_mean) for i in range (len(x)))

def variance(values):
    mean_val = mean(values)
    return sum((x - mean_val) ** 2 for x in values)
    
m = covariance(x, y) / variance(x)

b = mean(y) - m * mean(x)

def predict(x_new):
    return b + m * x_new

x_test = 9
print(predict(x_test))