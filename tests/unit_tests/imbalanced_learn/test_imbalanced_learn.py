import copy
import os
from unittest import TestCase

import numpy as np
import pandas as pd

from deepmol.datasets import SmilesDataset
from deepmol.imbalanced_learn import RandomOverSampler, SMOTE
from tests import TEST_DIR
from unit_tests._mock_utils import SmilesDatasetMagicMock


class TestImbalancedLearn(TestCase):

    def setUp(self) -> None:
        # create a dataset with 10 samples and 5 features
        x = np.random.randint(0, 10, size=(10, 5))
        # y with 8 samples of class 0 and 2 samples of class 1
        y = np.zeros(8)
        y = np.append(y, np.ones(2))
        # ids 10 characters from the alphabet
        ids = np.array([''.join(np.random.choice(list('abcdefghij'), 10)) for _ in range(10)])
        self.imbalanced_dataset = SmilesDatasetMagicMock(spec=SmilesDataset,
                                                         X=x,
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

    def test_smote(self):
        df = copy.deepcopy(self.imbalanced_dataset)
        new_df = SMOTE().sample(df)
        self.assertTrue(new_df._X.shape[0] > df.X.shape[0])
        # assert under-represented class is oversampled
        self.assertTrue(np.sum(new_df._y == 1) > np.sum(df.y == 1))
        # assert ids are not duplicated
        self.assertEqual(len(np.unique(new_df._ids)), len(new_df._ids))
