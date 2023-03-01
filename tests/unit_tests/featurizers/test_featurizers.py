from abc import abstractmethod, ABC

import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from deepmol.datasets import SmilesDataset
from deepmol.loaders.loaders import CSVLoader

from tests import TEST_DIR


class FeaturizerTestCase(ABC):

    def setUp(self) -> None:
        # TODO: remove the dependencies on other modules of DeepMol (use mocks instead)
        self.data_path = os.path.join(TEST_DIR, 'data')

        dataset = os.path.join(self.data_path, "test_to_convert_to_sdf.csv")
        loader = CSVLoader(dataset,
                           mols_field='Standardized_Smiles',
                           labels_fields='Class')

        self.mini_dataset_to_test = loader.create_dataset()

        dataset = os.path.join(self.data_path, "invalid_smiles_dataset.csv")
        loader = CSVLoader(dataset,
                           mols_field='Standardized_Smiles',
                           labels_fields='Class')

        self.dataset_invalid_smiles = loader.create_dataset()

        self.mol2vec_model = os.path.join(os.path.abspath(os.curdir), "compound_featurization", "mol2vec_models",
                                          "model_300dim.pkl")

        data_path = os.path.join(TEST_DIR, 'data/test_to_convert_to_sdf.csv')
        self.original_smiles = pd.read_csv(data_path, sep=',').Smiles.values
        self.mock_dataset = MagicMock(spec=SmilesDataset,
                                      mols=self.original_smiles,
                                      ids=np.arange(len(self.original_smiles)))
        self.original_smiles_with_invalid = np.append(self.original_smiles, ['CC(=O)[O-].NC', 'C1=CC=CC=C1('])
        self.mock_dataset_with_invalid = MagicMock(spec=SmilesDataset,
                                                   mols=self.original_smiles_with_invalid,
                                                   ids=np.arange(len(self.original_smiles_with_invalid)))

    @abstractmethod
    def test_featurize(self):
        raise NotImplementedError

    @abstractmethod
    def test_featurize_with_nan(self):
        raise NotImplementedError
