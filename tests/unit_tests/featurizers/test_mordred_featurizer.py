from copy import copy
from unittest import TestCase

import numpy as np

from compoundFeaturization.mordredDescriptors import MordredFeaturizer
from unit_tests.featurizers.test_featurizers import FeaturizerTestCase


class TestMordredDescriptors(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        dataset_rows_number = len(self.mini_dataset_to_test.mols)
        MordredFeaturizer().featurize(self.mini_dataset_to_test, remove_nans_axis=1)

        self.assertEqual(dataset_rows_number, self.mini_dataset_to_test.X.shape[0])
        self.assertEqual(self.mini_dataset_to_test.X.shape[1], 1280)

        dataset_rows_number = len(self.dataset_to_test.mols)
        MordredFeaturizer().featurize(self.dataset_to_test, remove_nans_axis=1)

        self.assertEqual(dataset_rows_number, self.dataset_to_test.X.shape[0])
        self.assertEqual(self.mini_dataset_to_test.X.shape[1], 1280)

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mini_dataset_to_test.mols)
        to_add = np.zeros(4)

        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))

        dataset = copy(self.mini_dataset_to_test)
        MordredFeaturizer().featurize(dataset, remove_nans_axis=1)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])



