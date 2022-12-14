from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class PCA_KNN:
    
    def __init__(self, n_components, n_neighbors):
        self.n_components = n_components
        self.n_neighbors = n_neighbors

    def PCAprocess(self,x):
        pca = PCA(self.n_components).fit(x)
        return pca.transform(x)

    def fit(self, x, y):
        knn = KNeighborsClassifier(self.n_neighbors)
        x_pca = self.PCAprocess(x)
        knn.fit(x_pca, y)
        self.clf = knn

    def predict(self,x):
        clf = self.clf
        x_pca = self.PCAprocess(x)
        pred = clf.predict(x_pca)
        return pred