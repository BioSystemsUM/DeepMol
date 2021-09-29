from copy import copy
from unittest import TestCase

from compoundFeaturization.rdkitFingerprints import AtomPairFingerprint, MorganFingerprint, MACCSkeysFingerprint, \
    LayeredFingerprint, RDKFingerprint
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase
import numpy as np


class TestRDKitFingerprints(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        # test Atom Pair fingerprints (without NaN generation)
        dataset_rows_number = len(self.dataset_to_test.mols)
        AtomPairFingerprint().featurize(self.dataset_to_test)
        self.assertEqual(dataset_rows_number, self.dataset_to_test.X.shape[0])

        MorganFingerprint().featurize(self.dataset_to_test)
        self.assertEqual(dataset_rows_number, self.dataset_to_test.X.shape[0])

        MACCSkeysFingerprint().featurize(self.dataset_to_test)
        self.assertEqual(dataset_rows_number, self.dataset_to_test.X.shape[0])

        LayeredFingerprint().featurize(self.dataset_to_test)
        self.assertEqual(dataset_rows_number, self.dataset_to_test.X.shape[0])

        RDKFingerprint().featurize(self.dataset_to_test)
        self.assertEqual(dataset_rows_number, self.dataset_to_test.X.shape[0])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.dataset_to_test.mols)
        to_add = np.zeros(4)

        self.dataset_to_test.mols = np.concatenate((self.dataset_to_test.mols, to_add))
        self.dataset_to_test.y = np.concatenate((self.dataset_to_test.y, to_add))

        dataset = copy(self.dataset_to_test)
        AtomPairFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        dataset = copy(self.dataset_to_test)
        MorganFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        dataset = copy(self.dataset_to_test)
        MACCSkeysFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        dataset = copy(self.dataset_to_test)
        LayeredFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

        dataset = copy(self.dataset_to_test)
        RDKFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset.X.shape[0])

