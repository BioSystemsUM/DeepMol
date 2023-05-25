from unittest import TestCase

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

        # TODO: continue here
        # This test fails because the max_length is too low
        # reconstructed_dataset = ohe._inverse_transform(dataset)
        # self.assertEqual(dataset_rows_number, reconstructed_dataset._X.shape[0])
        # for i in range(dataset_rows_number):
        #     self.assertEqual(self.mock_dataset.smiles[i], reconstructed_dataset.smiles[i])
