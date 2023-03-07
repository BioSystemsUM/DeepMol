import os
from abc import abstractmethod, ABC
from unittest.mock import MagicMock

import pandas as pd
from rdkit import DataStructs
from rdkit.Chem import MolFromSmiles

from deepmol.datasets import SmilesDataset

import numpy as np

from tests import TEST_DIR


class SplittersTestCase(ABC):

    def setUp(self) -> None:
        data_path = os.path.join(TEST_DIR, 'data/test_to_convert_to_sdf.csv')
        ttcts = pd.read_csv(data_path, sep=',')
        original_smiles = ttcts.Smiles.values
        original_y = ttcts.Class.values
        mols = np.array([self._smiles_to_mol(s) for s in original_smiles])
        # invalid molecules are removed by the SmilesDataset class
        valid = [i for i, mol in enumerate(mols) if mol is not None]
        self.mini_dataset_to_test = MagicMock(spec=SmilesDataset,
                                              smiles=original_smiles[valid],
                                              mols=mols[valid],
                                              ids=np.arange(len(original_smiles[valid])),
                                              y=original_y[valid],
                                              mode='classification')
        self.mini_dataset_to_test.__len__.return_value = len(self.mini_dataset_to_test.smiles)
        self.mini_dataset_to_test.select_to_split.side_effect = \
            lambda arg: MagicMock(spec=SmilesDataset,
                                  smiles=self.mini_dataset_to_test.smiles[arg],
                                  mols=self.mini_dataset_to_test.mols[arg],
                                  y=self.mini_dataset_to_test.y[arg],
                                  mode='classification')

        dataset = os.path.join(TEST_DIR, "data", "PC-3.csv")
        pc3 = pd.read_csv(dataset, sep=',')
        smiles = pc3.smiles.values
        y = pc3.pIC50.values
        mols = np.array([self._smiles_to_mol(s) for s in smiles])
        # invalid molecules are removed by the SmilesDataset class
        valid = [i for i, mol in enumerate(mols) if mol is not None]
        self.dataset_to_test = MagicMock(spec=SmilesDataset,
                                         smiles=smiles[valid],
                                         mols=mols[valid],
                                         ids=np.arange(len(smiles[valid])),
                                         y=y[valid],
                                         mode='regression')
        self.dataset_to_test.__len__.return_value = len(self.dataset_to_test.smiles)
        self.dataset_to_test.select_to_split.side_effect = \
            lambda arg: MagicMock(spec=SmilesDataset,
                                  smiles=self.dataset_to_test.smiles[arg],
                                  mols=self.dataset_to_test.mols[arg],
                                  y=self.dataset_to_test.y[arg],
                                  mode='regression')

        dataset = os.path.join(TEST_DIR, "data", "invalid_smiles_dataset.csv")
        isd = pd.read_csv(dataset, sep=',')
        smiles = isd.Standardized_Smiles.values
        y = isd.Class.values
        mols = np.array([self._smiles_to_mol(s) for s in smiles])
        # invalid molecules are removed by the SmilesDataset class
        valid = [i for i, mol in enumerate(mols) if mol is not None]
        self.invalid_smiles_dataset = MagicMock(spec=SmilesDataset,
                                                smiles=smiles[valid],
                                                mols=mols[valid],
                                                ids=np.arange(len(smiles[valid])),
                                                y=y[valid],
                                                mode='classification')
        self.invalid_smiles_dataset.__len__.return_value = len(self.invalid_smiles_dataset.smiles)
        self.invalid_smiles_dataset.select_to_split.side_effect = \
            lambda arg: MagicMock(spec=SmilesDataset,
                                  smiles=self.invalid_smiles_dataset.smiles[arg],
                                  mols=self.invalid_smiles_dataset.mols[arg],
                                  y=self.invalid_smiles_dataset.y[arg],
                                  mode='classification')

        dataset = os.path.join(TEST_DIR, "data", "preprocessed_dataset.csv")
        pdwf = pd.read_csv(dataset, sep=',')
        smiles = pdwf.Standardized_Smiles.values
        y = pdwf.Class.values
        mols = np.array([self._smiles_to_mol(s) for s in smiles])
        # invalid molecules are removed by the SmilesDataset class
        valid = [i for i, mol in enumerate(mols) if mol is not None]
        self.binary_dataset = MagicMock(spec=SmilesDataset,
                                        smiles=smiles[valid],
                                        mols=mols[valid],
                                        ids=np.arange(len(smiles[valid])),
                                        y=y[valid],
                                        mode='classification')
        self.binary_dataset.__len__.return_value = len(self.binary_dataset.smiles)
        self.binary_dataset.select_to_split.side_effect = lambda arg: MagicMock(spec=SmilesDataset,
                                                                                smiles=self.binary_dataset.smiles[arg],
                                                                                mols=self.binary_dataset.mols[arg],
                                                                                y=self.binary_dataset.y[arg],
                                                                                mode='classification')

        dataset = os.path.join(TEST_DIR, "data", "train_dataset.csv")
        td = pd.read_csv(dataset, sep=',')
        y = td.y.values
        feature_names = td.columns.values[3:]
        x = td.loc[:, feature_names].values
        self.dataset_for_k_split = MagicMock(spec=SmilesDataset,
                                             X=x,
                                             y=y,
                                             feature_names=feature_names,
                                             mode='classification')
        self.dataset_for_k_split.X = x
        self.dataset_for_k_split.y = y
        self.dataset_for_k_split.__len__.return_value = len(self.dataset_for_k_split.smiles)
        self.dataset_for_k_split.select_to_split.side_effect = lambda arg: MagicMock(spec=SmilesDataset,
                                                                                     x=self.dataset_for_k_split.X[arg],
                                                                                     y=self.dataset_for_k_split.y[arg],
                                                                                     mode='classification')

    def tearDown(self) -> None:
        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')

    @staticmethod
    def _smiles_to_mol(smiles):
        try:
            return MolFromSmiles(smiles)
        except:
            return None

    @staticmethod
    def _calculate_mean_fingerprints_smilarity(fps):
        # Calculate pairwise Tanimoto similarity
        similarity_matrix = np.zeros((len(fps), len(fps)))
        for i in range(len(fps)):
            for j in range(i + 1, len(fps)):
                similarity_matrix[i, j] = DataStructs.TanimotoSimilarity(fps[i], fps[j])
                similarity_matrix[j, i] = similarity_matrix[i, j]  # matrix is symmetric
        return np.mean(similarity_matrix)

    @abstractmethod
    def test_split(self):
        raise NotImplementedError

    @abstractmethod
    def test_k_fold_split(self):
        raise NotImplementedError
