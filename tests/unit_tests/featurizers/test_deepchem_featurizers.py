from copy import copy
from unittest import TestCase

import numpy as np

from compound_featurization.deepchem_featurizers import WeaveFeat, CoulombFeat
from unit_tests.featurizers.test_featurizers import FeaturizerTestCase


class TestDeepChemFeaturizers(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        pass

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mini_dataset_to_test.mols)
        to_add = np.zeros(4)
        ids_to_add = np.array([5, 6, 7, 8])

        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))
        self.mini_dataset_to_test.ids = np.concatenate((self.mini_dataset_to_test.y, ids_to_add))

        dataset = copy(self.mini_dataset_to_test)
        WeaveFeat().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])