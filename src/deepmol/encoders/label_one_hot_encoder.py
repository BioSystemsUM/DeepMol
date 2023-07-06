from sklearn.preprocessing import OneHotEncoder

from deepmol.base import Transformer
from deepmol.datasets import Dataset


class LabelOneHotEncoder(Transformer):
    """
    Class that encodes labels as one-hot vectors.
    This class is used to encode labels as one-hot vectors. This is useful for classification tasks.

    Attributes
    ----------
    encoder: OneHotEncoder
        Scikit-learn one-hot encoder.
    """

    def __init__(self):
        """
        Initialize this label encoder.
        """
        super().__init__()
        self.encoder = OneHotEncoder()

    def _fit(self, dataset: Dataset) -> 'LabelOneHotEncoder':
        """
        Fit this label encoder.

        Parameters
        ----------
        dataset: Dataset
            Dataset to fit on.

        Returns
        -------
        LabelOneHotEncoder
            Fitted label encoder.
        """
        # if values are integers, convert them to strings
        y = dataset.y.astype(str) if dataset.y.dtype == int else dataset.y
        # reshape if single feature
        y = y.reshape(-1, 1) if len(dataset.y.shape) == 1 else y
        self.encoder.fit(y)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        """
        Transform the labels of a dataset.

        Parameters
        ----------
        dataset: Dataset
            Dataset to transform.

        Returns
        -------
        Dataset
            Transformed dataset.
        """
        # if values are integers, convert them to strings
        y = dataset.y.astype(str) if dataset.y.dtype == int else dataset.y
        # reshape if single feature
        y = y.reshape(-1, 1) if len(dataset.y.shape) == 1 else y
        dataset._y = self.encoder.transform(y).toarray()
        return dataset
