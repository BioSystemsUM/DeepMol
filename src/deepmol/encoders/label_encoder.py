from sklearn.preprocessing import LabelEncoder as SKLabelEncoder

from deepmol.base import Transformer
from deepmol.datasets import Dataset


class LabelEncoder(Transformer):

    def __init__(self):
        super().__init__()
        self.encoder = SKLabelEncoder()

    def _transform(self, dataset: Dataset) -> Dataset:
        dataset._y = self.encoder.transform(dataset.y)
        return dataset

    def _fit(self, dataset: Dataset) -> 'LabelEncoder':
        self.encoder.fit(dataset.y)
        return self
