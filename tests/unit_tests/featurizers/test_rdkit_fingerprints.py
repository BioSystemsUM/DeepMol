import os
from copy import copy
from unittest import TestCase

from compoundFeaturization.rdkitFingerprints import AtomPairFingerprintCallbackHash, MorganFingerprint, \
    MACCSkeysFingerprint, \
    LayeredFingerprint, RDKFingerprint, AtomPairFingerprint
from loaders.Loaders import CSVLoader
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase
import numpy as np


class TestRDKitFingerprints(FeaturizerTestCase, TestCase):

    def test_featurize_morgan(self):
        dir_path = os.path.join(os.path.dirname(os.path.abspath(".")))
        dataset = os.path.join(dir_path, "tests", "data", "test_dataset.csv")
        loader = CSVLoader(dataset,
                           mols_field='mols',
                           labels_fields='y')

        test_dataset = loader.create_dataset()

        MorganFingerprint().featurize(test_dataset)

    def test_featurize(self):
        # test Atom Pair fingerprints (without NaN generation)
        dataset_rows_number = len(self.mini_dataset_to_test.mols)
        AtomPairFingerprint().featurize(self.mini_dataset_to_test)
        self.assertEqual(dataset_rows_number, self.mini_dataset_to_test.X.shape[0])

        MorganFingerprint().featurize(self.mini_dataset_to_test)
        self.assertEqual(dataset_rows_number, self.mini_dataset_to_test.X.shape[0])

        MACCSkeysFingerprint().featurize(self.mini_dataset_to_test)
        self.assertEqual(dataset_rows_number, self.mini_dataset_to_test.X.shape[0])

        LayeredFingerprint().featurize(self.mini_dataset_to_test)
        self.assertEqual(dataset_rows_number, self.mini_dataset_to_test.X.shape[0])

        RDKFingerprint().featurize(self.mini_dataset_to_test)
        self.assertEqual(dataset_rows_number, self.mini_dataset_to_test.X.shape[0])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mini_dataset_to_test.mols)
        to_add = np.zeros(4)

        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))

        dataset = copy(self.mini_dataset_to_test)
        AtomPairFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        dataset = copy(self.mini_dataset_to_test)
        MorganFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        dataset = copy(self.mini_dataset_to_test)
        MACCSkeysFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        dataset = copy(self.mini_dataset_to_test)
        LayeredFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        dataset = copy(self.mini_dataset_to_test)
        RDKFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])
