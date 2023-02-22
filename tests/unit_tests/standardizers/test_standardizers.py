import os
from abc import ABC, abstractmethod
from unittest.mock import MagicMock

import pandas as pd

from deepmol.datasets import NumpyDataset

from tests import TEST_DIR


class StandardizerBaseTestCase(ABC):

    def setUp(self) -> None:
        data_path = os.path.join(TEST_DIR, 'data/test_to_convert_to_sdf.csv')
        self.original_smiles = pd.read_csv(data_path, sep=',').Smiles.values
        self.mock_dataset = MagicMock(spec=NumpyDataset, mols=self.original_smiles)

    @abstractmethod
    def test_standardize(self):
        raise NotImplementedError
