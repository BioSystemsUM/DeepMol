from copy import copy
from unittest import TestCase

from deepmol.compound_featurization import MorganFingerprint, \
    MACCSkeysFingerprint, \
    LayeredFingerprint, RDKFingerprint, AtomPairFingerprint
from tests.unit_tests.featurizers.test_featurizers import FeaturizerTestCase


class TestRDKitFingerprints(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        # test Atom Pair fingerprints (without NaN generation)
        dataset_rows_number = len(self.mock_dataset.mols)
        AtomPairFingerprint().featurize(self.mock_dataset)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])

        MorganFingerprint().featurize(self.mock_dataset)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])

        MACCSkeysFingerprint().featurize(self.mock_dataset)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])

        LayeredFingerprint().featurize(self.mock_dataset)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])

        RDKFingerprint().featurize(self.mock_dataset)
        self.assertEqual(dataset_rows_number, self.mock_dataset._X.shape[0])

    def test_featurize_with_nan(self):
        dataset_rows_number = len(self.mock_dataset_with_invalid.mols) - 1  # one mol has invalid smiles

        dataset = copy(self.mock_dataset_with_invalid)
        AtomPairFingerprint(n_jobs=1).featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

        dataset = copy(self.mock_dataset_with_invalid)
        MorganFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

        dataset = copy(self.mock_dataset_with_invalid)
        MACCSkeysFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

        dataset = copy(self.mock_dataset_with_invalid)
        LayeredFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

        dataset = copy(self.mock_dataset_with_invalid)
        RDKFingerprint().featurize(dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])
