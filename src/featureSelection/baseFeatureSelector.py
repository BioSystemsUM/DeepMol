from Dataset import Dataset
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np

class BaseFeatureSelector(object):
    """Abstract class for feature selection.
    A `BaseFeatureSelector` uses features present in a Dataset object
    to select the most important ones. FeatureSelectors which are subclasses of
    this class should always operate over Dataset Objects.

    Child classes need to implement the _select_features method for
    performing feature selection.
    """

    def __init__(self):
        if self.__class__ == BaseFeatureSelector:
            raise Exception('Abstract class BaseFeatureSelector should not be instantiated')

        self.features2keep = None

    def featureSelection(self, dataset: Dataset):
        #TODO: review comments
        """Perform feature selection for the molecules present in the dataset.
        Parameters
        ----------
        dataset: Dataset object ...
        Returns
        -------
        features: np.ndarray
          A numpy array containing a featurized representation of `datapoints`.
        """
        features = dataset.features

        features, self.features2keep = self._featureSelector(np.stack(features, axis=0))

        dataset.features = np.asarray(features)

        dataset.features2keep = self.features2keep

        return dataset


class LowVarianceFS(BaseFeatureSelector):
    '''
    ...
    '''

    def __init__(self, threshold: float = 0.3):
        '''
        :param threshold:
        '''

        self.param = threshold

    def _featureSelector(self, features):
        """Returns indexes of features to keep."""
        vt = VarianceThreshold(threshold=self.param)
        tr = vt.fit_transform(features)
        return tr, vt.get_support(indices=True)

