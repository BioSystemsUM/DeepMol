from copy import copy
from unittest import TestCase

from compoundFeaturization.mol2vec import Mol2Vec
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase
import numpy as np


class TestMol2Vec(FeaturizerTestCase, TestCase):
    def test_featurize(self):
        dataset_rows_number = len(self.mini_dataset_to_test.mols)
        Mol2Vec().featurize(
            self.mini_dataset_to_test)
        self.assertEqual(dataset_rows_number, self.mini_dataset_to_test.X.shape[0])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mini_dataset_to_test.mols)
        to_add = np.zeros(4)

        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))

        dataset = copy(self.mini_dataset_to_test)
        Mol2Vec(pretrain_model_path="../../../src/compoundFeaturization/mol2vec_models/model_300dim.pkl").featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])
