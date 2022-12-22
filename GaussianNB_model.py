from sklearn.naive_bayes import GaussianNB
import numpy as np

class naive_bayes:
    
    def __init__(self):
        self.clf = GaussianNB()

    def fit(self, x, y):
        clf = self.clf
        clf.fit(x,y)
        self.clf = clf

    def predict(self, x):
        clf = self.clf
        pred = clf.predict(x)
        return pred

    def predict_proba(self, x):
        clf = self.clf
        pred = clf.predict_proba(x)[:, 1]
        return pred