from sklearn.preprocessing import LabelEncoder as SKLabelEncoder

from deepmol.base import Transformer
from deepmol.datasets import Dataset


class LabelEncoder(Transformer):
    """
    Class that encodes labels as integers.

    This class is used to encode labels as integers. This is useful for classification tasks.

    Attributes
    ----------
    encoder: SKLabelEncoder
        Scikit-learn label encoder.
    """

    def __init__(self):
        """
        Initialize this label encoder.
        """
        super().__init__()
        self.encoder = SKLabelEncoder()

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
        dataset._y = self.encoder.transform(dataset.y)
        return dataset

    def _fit(self, dataset: Dataset) -> 'LabelEncoder':
        """
        Fit this label encoder.

        Parameters
        ----------
        dataset: Dataset
            Dataset to fit on.

        Returns
        -------
        LabelEncoder
            Fitted label encoder.
        """
        self.encoder.fit(dataset.y)
        return self
