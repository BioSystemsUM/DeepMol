import copy
import os
from unittest import TestCase

import numpy as np
import pandas as pd

from deepmol.datasets import SmilesDataset
from deepmol.feature_selection import BorutaAlgorithm, LowVarianceFS, KbestFS, PercentilFS, RFECVFS, SelectFromModelFS
from deepmol.loggers import Logger

from tests import TEST_DIR
from tests.unit_tests._mock_utils import SmilesDatasetMagicMock


class TestFeatureSelectors(TestCase):

    def setUp(self) -> None:
        dataset = os.path.join(TEST_DIR, "data", "train_dataset.csv")
        td = pd.read_csv(dataset, sep=',', nrows=250)
        y = td.y.values
        feature_names = td.columns.values[3:]
        x = td.loc[:, feature_names].values
        # add some random values between 0 and 5 to the features in the first 10 columns
        x[:, :10] += np.random.randint(0, 5, size=(x.shape[0], 10))
        self.features_dataset = SmilesDatasetMagicMock(spec=SmilesDataset,
                                                       X=x,
                                                       y=y,
                                                       feature_names=feature_names,
                                                       mode='classification')

    def tearDown(self) -> None:
        # Close logger file handlers to release the file
        singleton_instance = Logger()
        singleton_instance.close_handlers()
        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')

    def validate_feature_selector(self, feature_selector, **kwargs):
        df = copy.deepcopy(self.features_dataset)

        def side_effect(arg):
            df.X = df.X[:, arg]
            df.feature_names = df.feature_names[arg]

        self.features_dataset.select_features_by_index.side_effect = side_effect
        feature_selector(**kwargs).select_features(df, inplace=True)
        self.assertLessEqual(len(self.features_dataset.feature_names), len(df.feature_names))
        self.assertLessEqual(self.features_dataset.X.shape[1], df.X.shape[1])

    def test_feature_selectors(self):
        self.validate_feature_selector(LowVarianceFS, threshold=0.99)
        self.validate_feature_selector(KbestFS, k=10)
        self.validate_feature_selector(PercentilFS, percentil=10)
        self.validate_feature_selector(RFECVFS, step=10)
        self.validate_feature_selector(SelectFromModelFS, threshold=0.1)
        self.validate_feature_selector(BorutaAlgorithm, max_iter=3, n_estimators=50, support_weak=True)
