import numpy as np

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)  # Unique class labels
        self.mean = {}  # Mean of each feature per class
        self.var = {}   # Variance of each feature per class
        self.priors = {}  # Prior probabilities of each class

        for c in self.classes:
            X_c = X[y == c]  # Extract all samples of class c
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-6  # Avoid division by zero
            self.priors[c] = X_c.shape[0] / X.shape[0]  # P(y)

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        posteriors = []
        
        for c in self.classes:
            prior = np.log(self.priors[c])  # log P(y)
            likelihood = np.sum(self._gaussian_log_prob(x, self.mean[c], self.var[c]))
            posteriors.append(prior + likelihood)
        return self.classes[np.argmax(posteriors)]  # Class with max probability

    def _gaussian_log_prob(self, x, mean, var):
        return -0.5 * np.log(2 * np.pi * var) - ((x - mean) ** 2) / (2 * var)


from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score

X, y = make_classification(n_samples=100, n_features=5, n_classes=2, random_state=42)
X = np.array(X)
y = np.array(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = NaiveBayes()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(y_pred)
print(accuracy)