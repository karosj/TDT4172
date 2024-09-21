import numpy as np

class LinearRegression():
    
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.beta = 0
        self.beta_1 = 0
        
        
    def fit(self, X, y):
        X = np.array(X)
        y = np.array(y)
        
        m = len(y)

        # gradient descent
        for _ in range(self.n_iters):
            y_pred = self.beta + self.beta_1 * X

            dtheta = 1 / m * np.sum(y_pred - y)
            theta2 = 1 / m * np.sum((y_pred - y) * X)

            self.beta -= self.lr * dtheta
            self.beta_1 -= self.lr * theta2

    
    def predict(self, X):
        X = np.array(X)
        return self.beta + self.beta_1 * X
