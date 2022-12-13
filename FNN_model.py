from sklearn.neural_network import MLPClassifier
import numpy as np

class FNN:

    def __init__(self, hidden_layer_sizes, activation, solver, alpha, batch_size, 
        learning_rate, power_t, max_iter, shuffle, beta_1, beta_2):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.solver = solver
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.power_t = power_t
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.beta_1 = beta_1
        self.beta_2 = beta_2

    def fit(self, x, y):
        mlp = MLPClassifier(hidden_layer_sizes=self.hidden_layer_sizes, alpha=self.alpha, batch_size=self.batch_size, 
            learning_rate=self.learning_rate, power_t=self.power_t,
            max_iter=self.max_iter, shuffle=self.shuffle, beta_1=self.beta_1, beta_2=self.beta_2)
        mlp.fit(x, y)
        self.clf = mlp

    def predict(self,x):
        clf = self.clf
        pred = clf.predict(x)
        return pred