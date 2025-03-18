import numpy as np

x = np.array([[1, 2], [2, 3], [3, 1], [5, 4], [4, 5]])
y = np.array([[0], [0], [0], [1], [1]])


def sigmoid(val):
    return 1 / (1 + np.exp( -val))


def weights(n):
    return np.zeros((n, 1)), 0


def predict(x, w, b):
    model = np.dot(x, w) + b
    return sigmoid(model)

def gradient_descent(x, y, y_pred):
    n = len(y)
    gw = np.dot(x.T, (y_pred - y)) / n
    gb = np.sum(y_pred - y) / n
    return gw, gb

def train_gd(x, y, learning_rate = 0.01, epochs = 1000):
    n = x.shape[1]
    w, b = weights(n)
    
    for _ in range(epochs):
        y_pred = predict(x, w, b)
        gw, gb = gradient_descent(x, y ,y_pred)
        
        w -= learning_rate * gw
        b -= learning_rate * gb
    return w, b

def threshold(x, w, b, thresh = 0.5):
    y_pred = predict(x, w, b)
    return(y_pred >= thresh)

w, b = train_gd(x, y, learning_rate = 0.01, epochs = 1000)
prediction = threshold(x, w, b)

print(prediction.flatten())