from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import numpy as np

class PCA_KNN:
    
    def __init__(self, n_components=5, n_neighbors=10):
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

    def get_params(self, deep=True):
        """Get parameters for this estimator.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        out = dict()
        for key in ['n_components', 'n_neighbors']:
            value = getattr(self, key, None)
            if deep and hasattr(value, 'get_params'):
                deep_items = value.get_params().items()
                out.update((key + '__' + k, val) for k, val in deep_items)
            out[key] = value
        return out

    def set_params(self, **params):
        """Set the parameters of this estimator.

        The method works on simple estimators as well as on nested objects
        (such as pipelines). The latter have parameters of the form
        ``<component>__<parameter>`` so that it's possible to update each
        component of a nested object.

        Returns
        -------
        self
        """
        if not params:
            # Simple optimization to gain speed (inspect is slow)
            return self
        valid_params = self.get_params(deep=True)

        for key, value in params.items():
            if key not in valid_params:
                raise ValueError('Invalid parameter %s for estimator %s. '
                                 'Check the list of available parameters '
                                 'with `estimator.get_params().keys()`.' %
                                 (key, self))
            setattr(self, key, value)
            valid_params[key] = value

        return self