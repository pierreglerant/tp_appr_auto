import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelPropagation

from utils import dict_dict_first_key

class WeldLabelPropagation:
    """
    TODO
    """
    def __init__(self, thresholds):
        """
        TODO
        """
        self.thresholds = thresholds
        self.model = LabelPropagation(kernel='knn', n_neighbors=25)
        self.y = pd.DataFrame()
        self.accuracy_per_column = dict()
        
    def fit(self, X, Y):
        """
        TODO
        """
        for col_name in Y.columns:
            bin_col_name = 'Passed ' + col_name
            threshold = self.thresholds[col_name][dict_dict_first_key(self.thresholds, col_name)]
            Y[bin_col_name] = self._binarize(Y[col_name], threshold)   
            Y = Y.drop(columns=col_name)

            self.model.fit(X, Y[bin_col_name])
            Y[bin_col_name] = self.model.transduction_
        
        self.y = np.where(Y[Y.columns[:-1]].sum(axis=1) == 4, 1,
                          np.where(Y[Y.columns[-1]] == 1, 1, 0))
        self.Y_ = Y

    def avg_accuracy(self, X, Y):
        """
        TODO
        """
        cumulative_acc = 0.
        for col_name in Y.columns:
            y_hat = self.model.predict(X)

            bin_col_name = 'Passed ' + col_name
            threshold = self.thresholds[col_name][dict_dict_first_key(self.thresholds, col_name)]
            y = self._binarize(Y[col_name], threshold) 
            acc = accuracy_score(y, y_hat)
            self.accuracy_per_column[col_name] = acc
            cumulative_acc += acc
        
        return cumulative_acc / len(Y.columns)


    def _binarize(self, y, threshold):
        """
        TODO
        """
        y = np.where(y >= threshold, 1, 0)
        y = pd.DataFrame(y)
        y = np.where(y.isna(), -1, y)
        print(y.shape)
        return y

