import os
from abc import ABC, abstractmethod
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from rdkit.Chem import MolFromSmiles

from deepmol.datasets import SmilesDataset
from tests import TEST_DIR


class UnsupervisedBaseTestCase(ABC):

    def setUp(self) -> None:
        dataset = os.path.join(TEST_DIR, "data", "train_dataset.csv")
        td = pd.read_csv(dataset, sep=',')
        # pick first 125 and last 125 rows
        td = pd.concat([td.iloc[:125], td.iloc[-125:]])
        smiles = td.mols.values
        y = td.y.values
        feature_names = td.columns.values[3:]
        x = td.loc[:, feature_names].values
        self.dataset = MagicMock(spec=SmilesDataset,
                                 smiles=smiles,
                                 X=x,
                                 y=y,
                                 feature_names=feature_names,
                                 mode='classification',
                                 ids=[str(i) for i in range(len(smiles))],
                                 label_names=['label'])
        self.dataset.X = x

        dataset = os.path.join(TEST_DIR, "data", "PC-3.csv")
        pc3 = pd.read_csv(dataset, sep=',', nrows=250)
        smiles = pc3.smiles.values
        y = pc3.pIC50.values
        # random features of 0 and 1s
        x = np.random.randint(2, size=(len(smiles), 10))
        self.regression_dataset = MagicMock(spec=SmilesDataset,
                                            smiles=smiles,
                                            X=x,
                                            ids=np.arange(len(smiles)),
                                            y=y,
                                            mode='regression',
                                            label_names=['pIC50'])
        self.regression_dataset.X = x

    def tearDown(self) -> None:
        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')
        if os.path.exists('test_components.png'):
            os.remove('test_components.png')
        if os.path.exists('test_explained_variance.png'):
            os.remove('test_explained_variance.png')

    @staticmethod
    def _smiles_to_mol(smiles):
        try:
            return MolFromSmiles(smiles)
        except:
            return None

    @abstractmethod
    def test_run_unsupervised(self):
        raise NotImplementedError
