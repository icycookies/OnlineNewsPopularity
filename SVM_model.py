from sklearn import svm
import numpy as np

class SVM:
    
    def __init__(self, kernel='linear', C=2, gamma=1, decision_function_shape='ovr'):
        # Initialize the model's hyperparameters
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.decision_function_shape = decision_function_shape
       
    def fit(self, x, y):
        clf = self.clf
        clf.fit(x, y.astype('int'))
        self.clf = clf

    def predict(self, x):
        clf = self.clf
        pred = clf.predict(x)
        return pred
