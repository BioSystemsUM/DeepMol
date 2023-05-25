from unittest import TestCase

import numpy as np

from deepmol.compound_featurization import SmilesOneHotEncoder
from unit_tests.featurizers.test_featurizers import FeaturizerTestCase


class TestOneHotEncoding(FeaturizerTestCase, TestCase):
    def test_featurize(self):
        mock_dataset = self.mock_dataset.__copy__()
        dataset_rows_number = len(mock_dataset.mols)
        ohe = SmilesOneHotEncoder().fit(mock_dataset)
        dataset = ohe.transform(mock_dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

        reconstructed_dataset = ohe.inverse_transform(dataset)
        self.assertEqual(dataset_rows_number, reconstructed_dataset._X.shape[0])
        for i in range(dataset_rows_number):
            self.assertEqual(self.mock_dataset.smiles[i], reconstructed_dataset.smiles[i])

    def test_low_size(self):
        mock_dataset = self.mock_dataset.__copy__()
        dataset_rows_number = len(mock_dataset.mols)
        ohe = SmilesOneHotEncoder(max_length=10).fit(mock_dataset)
        dataset = ohe.transform(mock_dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

        reconstructed_dataset = ohe.inverse_transform(dataset)
        self.assertEqual(dataset_rows_number, reconstructed_dataset._X.shape[0])
        for i in range(dataset_rows_number):
            self.assertFalse(self.mock_dataset.smiles[i] == reconstructed_dataset.smiles[i])

    def test_fit_transform_featurize(self):
        mock_dataset = self.mock_dataset.__copy__()
        dataset_rows_number = len(mock_dataset.mols)
        ohe = SmilesOneHotEncoder().fit_transform(mock_dataset)
        self.assertEqual(dataset_rows_number, ohe._X.shape[0])
        ohe2 = SmilesOneHotEncoder().featurize(mock_dataset)
        self.assertEqual(dataset_rows_number, ohe2._X.shape[0])
        self.assertEqual(ohe._X.shape, ohe2._X.shape)
        for i in range(dataset_rows_number):
            self.assertTrue(np.array_equal(ohe._X[i], ohe2._X[i]))
