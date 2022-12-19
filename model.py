import copy
import numpy as np
from tqdm import tqdm


class LogisticRegression:
    def __init__(self, input_dim, learning_rate=0.1, regularization_strength=0.1, num_iterations=500):
        # Initialize the model's hyperparameters
        self.learning_rate = learning_rate
        self.regularization_strength = regularization_strength
        self.num_iterations = num_iterations
        self.weight = np.random.randn(input_dim, 1)
        self.bias = np.zeros(1)

    def forward_backward(self, x, y):
        """
            x: (N, d)
            y: (N, 1)
        """
        m = x.shape[0]

        pred = self.sigmoid(np.dot(x, self.weight) + self.bias)  
        cost = -(np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))) / m                 

        dZ = pred - y

        dw = (np.dot(x.transpose(1, 0), dZ))/m
        db = (np.sum(dZ))/m

        gradient = (dw, db)

        return gradient, cost
    
    def forward(self, x, y):
        m = x.shape[0]
        pred = self.sigmoid(np.dot(x, self.weight) + self.bias)  
        cost = -(np.sum(y * np.log(pred) + (1 - y) * np.log(1 - pred))) / m      
        return cost

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def fit(self, x, y):
        y = y.reshape(-1, 1)
        val_ratio = 0.05
        num_train = int((1 - val_ratio) * x.shape[0])
        perm = np.random.permutation(x.shape[0])
        train_x, train_y = x[perm[:num_train]], y[perm[:num_train]]
        val_x, val_y = x[perm[num_train:]], y[perm[num_train:]]

        best_val_cost = 1000
        best_weight = self.weight
        best_bias = self.bias
        for i in range(self.num_iterations):
            grads, cost = self.forward_backward(train_x, train_y)
            dw, db = grads

            self.weight = self.weight - self.learning_rate * dw
            self.bias = self.bias - self.learning_rate * db

            val_cost = self.forward(val_x, val_y)
            if val_cost < best_val_cost:
                best_bias = copy.deepcopy(self.bias)
                best_weight = copy.deepcopy(self.weight)
                best_val_cost = val_cost

        self.weight = best_weight
        self.bias = best_bias

    def predict(self, x):
        A = self.sigmoid(np.dot(x, self.weight) + self.bias)
        pred = np.where(A > 0.5, 1, 0)
        return pred