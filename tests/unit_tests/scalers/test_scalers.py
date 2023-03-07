import os
from abc import ABC, abstractmethod
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from deepmol.datasets import SmilesDataset
from tests import TEST_DIR


class ScalersTestCase(ABC):

    def setUp(self) -> None:
        dataset = os.path.join(TEST_DIR, "data", "train_dataset.csv")
        td = pd.read_csv(dataset, sep=',')
        # pick the first 100 rows and 100 columns
        y = td.y.values[:100]
        feature_names = td.columns.values[3:103]
        x = td.loc[:, feature_names].values[:100]
        # add some random values between -10 and 10 in the first column
        x[:, 0] = np.random.randint(-10, 10, x.shape[0])
        self.dataset = MagicMock(spec=SmilesDataset,
                                 X=x,
                                 y=y,
                                 feature_names=feature_names)
        self.dataset.X = x
        self.dataset.y = y
        self.dataset.__len__.return_value = len(self.dataset.smiles)

        x = np.array([[0, 1], [2, 3], [4, 5]])
        y = np.array([0, 1, 2])
        self.polynomial_features = MagicMock(spec=SmilesDataset,
                                             X=x,
                                             y=y)

    def tearDown(self) -> None:
        if os.path.exists("test_scaler.pkl"):
            os.remove("test_scaler.pkl")

    @abstractmethod
    def test_scaler(self):
        raise NotImplementedError
