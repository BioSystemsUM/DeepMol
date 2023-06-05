from unittest import TestCase

import numpy as np

from deepmol.compound_featurization import SmilesOneHotEncoder
from deepmol.tokenizers.kmer_smiles_tokenizer import KmerSmilesTokenizer
from unit_tests.featurizers.test_featurizers import FeaturizerTestCase


class TestOneHotEncoder(FeaturizerTestCase, TestCase):

    def test_featurize(self):
        mock_dataset = self.mock_dataset.__copy__()
        ohe = SmilesOneHotEncoder()
        df = ohe.featurize(mock_dataset)
        self.assertEqual(len(self.mock_dataset.smiles), df._X.shape[0])
        reconstructed_smiles = ohe.inverse_transform(mock_dataset._X)
        for i in range(len(self.mock_dataset.smiles)):
            self.assertEqual(self.mock_dataset.smiles[i], reconstructed_smiles[i])

    def test_low_size(self):
        mock_dataset = self.mock_dataset.__copy__()
        dataset_rows_number = len(mock_dataset.mols)
        ohe = SmilesOneHotEncoder(max_length=10).fit(mock_dataset)
        dataset = ohe.transform(mock_dataset)
        self.assertEqual(dataset_rows_number, dataset._X.shape[0])

        reconstructed_smiles = ohe.inverse_transform(dataset._X)
        for i in range(dataset_rows_number - 1):  # last one has less than 10 characters
            self.assertFalse(self.mock_dataset.smiles[i] == reconstructed_smiles[i])
        self.assertEqual(self.mock_dataset.smiles[-1], reconstructed_smiles[-1])

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

    def test_one_hot_encoder_kmers(self):
        mock_dataset = self.mock_dataset.__copy__()
        ohe = SmilesOneHotEncoder(tokenizer=KmerSmilesTokenizer())
        df = ohe.featurize(mock_dataset)
        self.assertEqual(len(self.mock_dataset.smiles), df._X.shape[0])
        reconstructed_smiles = ohe.inverse_transform(mock_dataset._X)
        for i in range(len(self.mock_dataset.smiles)):
            self.assertEqual(self.mock_dataset.smiles[i], reconstructed_smiles[i])
