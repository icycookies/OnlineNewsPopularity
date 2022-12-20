from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.base import BaseEstimator

class FNN(BaseEstimator):

    def __init__(self, hidden_layer_sizes=(64, 128), activation="relu", batch_size=32, lr=1e-3, max_iter=200):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.activation = activation
        self.batch_size = batch_size
        self.lr = lr
        self.max_iter = max_iter

    def fit(self, x, y):
        mlp = MLPClassifier(
            hidden_layer_sizes=self.hidden_layer_sizes, 
            activation=self.activation, 
            batch_size=self.batch_size, 
            learning_rate_init=self.lr,
            max_iter=self.max_iter
        )
        mlp.fit(x, y)
        self.clf = mlp

    def predict(self,x):
        clf = self.clf
        pred = clf.predict(x)
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
        for key in ['hidden_layer_sizes', 'activation', 'batch_size', 'lr', 'max_iter']:
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