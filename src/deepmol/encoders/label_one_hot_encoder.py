from sklearn.preprocessing import OneHotEncoder

from deepmol.base import Transformer
from deepmol.datasets import Dataset


class LabelOneHotEncoder(Transformer):

    def __init__(self):
        super().__init__()
        self.encoder = OneHotEncoder()

    def _fit(self, dataset: Dataset) -> 'LabelOneHotEncoder':
        # if values are integers, convert them to strings
        y = dataset.y.astype(str) if dataset.y.dtype == int else dataset.y
        # reshape if single feature
        y = y.reshape(-1, 1) if len(dataset.y.shape) == 1 else y
        self.encoder.fit(y)
        return self

    def _transform(self, dataset: Dataset) -> Dataset:
        # if values are integers, convert them to strings
        y = dataset.y.astype(str) if dataset.y.dtype == int else dataset.y
        # reshape if single feature
        y = y.reshape(-1, 1) if len(dataset.y.shape) == 1 else y
        dataset._y = self.encoder.transform(y).toarray()
        return dataset
