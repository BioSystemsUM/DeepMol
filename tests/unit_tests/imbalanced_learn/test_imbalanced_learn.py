import copy
import os
from unittest import TestCase

import numpy as np
from imblearn.over_sampling import SMOTE as SMOTE_IB
from rdkit.Chem import MolFromSmiles

from deepmol.datasets import SmilesDataset
from deepmol.imbalanced_learn import RandomOverSampler, SMOTE, ClusterCentroids, RandomUnderSampler, SMOTEENN, \
    SMOTETomek
from unit_tests._mock_utils import SmilesDatasetMagicMock


class TestImbalancedLearn(TestCase):

    def setUp(self) -> None:
        # create a dataset with 14 samples and 5 features
        x = np.random.randint(0, 10, size=(14, 5))
        # y with 10 samples of class 0 and 4 samples of class 1
        y = np.array([1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0])
        # ids 10 characters from the alphabet
        ids = np.array([''.join(np.random.choice(list('abcdefghij'), 14)) for _ in range(14)])
        # smiles with one more C than the last one
        smiles = np.array(['C' * (i + 1) for i in range(14)])
        # mols
        mols = np.array([MolFromSmiles(s) for s in smiles])
        self.imbalanced_dataset = SmilesDatasetMagicMock(spec=SmilesDataset,
                                                         X=x,
                                                         smiles=smiles,
                                                         mols=mols,
                                                         y=y,
                                                         ids=ids)

    def tearDown(self) -> None:
        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')

    def test_random_over_sampler(self):
        df = copy.deepcopy(self.imbalanced_dataset)
        new_df = RandomOverSampler().sample(df)
        self.assertTrue(new_df._X.shape[0] > df.X.shape[0])
        # assert under-represented class is oversampled
        self.assertTrue(np.sum(new_df._y == 1) > np.sum(df.y == 1))
        # assert ids are not duplicated
        self.assertEqual(len(np.unique(new_df._ids)), len(new_df._ids))
        self.assertEqual(len(new_df._ids), len(new_df._smiles), len(new_df._mols))

    def test_smote(self):
        df = copy.deepcopy(self.imbalanced_dataset)
        new_df = SMOTE(k_neighbors=1).sample(df)
        self.assertTrue(new_df._X.shape[0] > df.X.shape[0])
        # assert under-represented class is oversampled
        self.assertTrue(np.sum(new_df._y == 1) > np.sum(df.y == 1))
        # assert ids are not duplicated
        self.assertEqual(len(np.unique(new_df._ids)), len(new_df._ids))
        self.assertEqual(len(new_df._ids), len(new_df._smiles), len(new_df._mols))

    def test_cluster_centroids(self):
        df = copy.deepcopy(self.imbalanced_dataset)
        new_df = ClusterCentroids().sample(df)
        self.assertTrue(new_df._X.shape[0] < df.X.shape[0])
        # assert over-represented class is undersampled
        self.assertTrue(np.sum(new_df._y == 0) < np.sum(df.y == 0))
        # assert ids are not duplicated
        self.assertEqual(len(np.unique(new_df._ids)), len(new_df._ids))
        self.assertEqual(len(new_df._ids), len(new_df._smiles), len(new_df._mols))

    def test_random_under_sampler(self):
        df = copy.deepcopy(self.imbalanced_dataset)
        new_df = RandomUnderSampler().sample(df)
        self.assertTrue(new_df._X.shape[0] < df.X.shape[0])
        # assert over-represented class is undersampled
        self.assertTrue(np.sum(new_df._y == 0) < np.sum(df.y == 0))
        # assert ids are not duplicated
        self.assertEqual(len(np.unique(new_df._ids)), len(new_df._ids))
        self.assertEqual(len(new_df._ids), len(new_df._smiles), len(new_df._mols))

    def test_SMOTEENN(self):
        df = copy.deepcopy(self.imbalanced_dataset)
        smote = SMOTE_IB(k_neighbors=1)
        new_df = SMOTEENN(smote=smote).sample(df)
        # assert under-represented class is oversampled
        self.assertTrue(np.sum(new_df._y == 1) >= np.sum(df.y == 1))
        # assert over-represented class is undersampled
        self.assertTrue(np.sum(new_df._y == 0) <= np.sum(df.y == 0))
        # assert ids are not duplicated
        self.assertEqual(len(np.unique(new_df._ids)), len(new_df._ids))
        self.assertEqual(len(new_df._ids), len(new_df._smiles), len(new_df._mols))

    def test_SMOTETomek(self):
        df = copy.deepcopy(self.imbalanced_dataset)
        smote = SMOTE_IB(k_neighbors=2)
        new_df = SMOTETomek(smote=smote, random_state=42).sample(df)
        # assert under-represented class is oversampled
        self.assertTrue(np.sum(new_df._y == 1) >= np.sum(df.y == 1))
        # assert over-represented class is undersampled
        self.assertTrue(np.sum(new_df._y == 0) <= np.sum(df.y == 0))
        # assert ids are not duplicated
        self.assertEqual(len(np.unique(new_df._ids)), len(new_df._ids))
        self.assertEqual(len(new_df._ids), len(new_df._smiles), len(new_df._mols))
