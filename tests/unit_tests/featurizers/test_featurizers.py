from abc import abstractmethod, ABC

import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles

from deepmol.datasets import SmilesDataset
from deepmol.scalers import StandardScaler

from tests import TEST_DIR


class FeaturizerTestCase(ABC):

    def setUp(self) -> None:
        data_path = os.path.join(TEST_DIR, 'data/test_to_convert_to_sdf.csv')
        self.original_smiles = pd.read_csv(data_path, sep=',').Smiles.values
        mols = [self._smiles_to_mol(s) for s in self.original_smiles]
        self.mock_dataset = MagicMock(spec=SmilesDataset,
                                      smiles=np.array(self.original_smiles),
                                      mols=np.array(mols),
                                      ids=np.arange(len(self.original_smiles)))
        self.original_smiles_with_invalid = np.append(self.original_smiles, ['CC(=O)[O-].NC', 'C1=CC=CC=C1('])
        mols = [self._smiles_to_mol(s) for s in self.original_smiles_with_invalid]
        self.mock_dataset_with_invalid = MagicMock(spec=SmilesDataset,
                                                   smiles=np.array(self.original_smiles_with_invalid),
                                                   mols=np.array(mols),
                                                   ids=np.arange(len(self.original_smiles_with_invalid)))

        self.mock_scaler = MagicMock(spec=StandardScaler)

    def tearDown(self) -> None:
        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')

    @staticmethod
    def _smiles_to_mol(smiles):
        try:
            return MolFromSmiles(smiles)
        except:
            return None

    @abstractmethod
    def test_featurize(self):
        raise NotImplementedError

    @abstractmethod
    def test_featurize_with_nan(self):
        raise NotImplementedError
