import os
from abc import ABC, abstractmethod
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from deepmol.datasets import SmilesDataset

from tests import TEST_DIR


class StandardizerBaseTestCase(ABC):

    def setUp(self) -> None:
        data_path = os.path.join(TEST_DIR, 'data/test_to_convert_to_sdf.csv')
        self.original_smiles = pd.read_csv(data_path, sep=',').Smiles.values
        self.original_smiles = np.append(self.original_smiles, ['CC(=O)[O-].NC', 'C1=CC=CC=C1('])
        self.mock_dataset = MagicMock(spec=SmilesDataset, mols=self.original_smiles)

    @abstractmethod
    def test_standardize(self):
        raise NotImplementedError
