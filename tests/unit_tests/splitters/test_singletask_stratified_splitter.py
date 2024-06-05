from unittest import TestCase

import numpy as np

from deepmol.splitters import SingletaskStratifiedSplitter
from tests.unit_tests.splitters.test_splitters import SplittersTestCase


class SingleTaskStratifiedSplitterTestCase(SplittersTestCase, TestCase):

    def test_split(self):
        stss_splitter = SingletaskStratifiedSplitter()

        train_dataset, test_dataset = stss_splitter.train_test_split(self.mini_dataset_to_test, seed=123)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 5)
        self.assertEqual(len(test_dataset.smiles), 2)

    def test_k_fold_split(self):
        stss_splitter = SingletaskStratifiedSplitter()

        folds = stss_splitter.k_fold_split(self.dataset_for_k_split, k=3, seed=123)

        for train_df, valid_df in folds:
            self.assertEqual(len(train_df.y) + len(valid_df.y), len(self.dataset_for_k_split.y))

            self.assertAlmostEqual(
                len(train_df.y[train_df.y == 1]) / len(train_df.y),
                len(self.dataset_for_k_split.y[self.dataset_for_k_split.y == 1]) / len(self.dataset_for_k_split.y),
                delta=0.01)
            self.assertAlmostEqual(
                len(valid_df.y[valid_df.y == 1]) / len(valid_df.y),
                len(self.dataset_for_k_split.y[self.dataset_for_k_split.y == 1]) / len(self.dataset_for_k_split.y),
                delta=0.01)

    def test_similarity_splitter_larger_dataset(self):
        stss_splitter = SingletaskStratifiedSplitter()

        with self.assertRaises(ValueError):
            stss_splitter.train_test_split(self.dataset_to_test)

        train_dataset, test_dataset = stss_splitter.train_test_split(self.dataset_to_test,
                                                                     frac_train=0.9,
                                                                     seed=123,
                                                                     force_split=True)
        self.assertEqual(len(train_dataset.smiles), int(0.9 * len(self.dataset_to_test.smiles)))
        self.assertAlmostEqual(len(test_dataset.smiles), int(0.1 * len(self.dataset_to_test.smiles)), delta=1)

    def test_similarity_splitter_larger_dataset_binary_classification(self):
        stss_splitter = SingletaskStratifiedSplitter()

        train_dataset, test_dataset = stss_splitter.train_test_split(self.binary_dataset)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertAlmostEqual(
            len(train_dataset.y[train_dataset.y == 1]) / len(train_dataset.y),
            len(self.binary_dataset.y[self.binary_dataset.y == 1]) / len(self.binary_dataset.y),
            delta=0.01)
        self.assertAlmostEqual(
            len(test_dataset.y[test_dataset.y == 1]) / len(test_dataset.y),
            len(self.binary_dataset.y[self.binary_dataset.y == 1]) / len(self.binary_dataset.y),
            delta=0.01)

        train_dataset, valid_dataset, test_dataset = stss_splitter.train_valid_test_split(self.binary_dataset,
                                                                                          frac_train=0.8,
                                                                                          frac_valid=0.1)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertGreater(len(train_dataset.smiles), len(valid_dataset.smiles))

        self.assertAlmostEqual(
            len(train_dataset.y[train_dataset.y == 1]) / len(train_dataset.y),
            len(self.binary_dataset.y[self.binary_dataset.y == 1]) / len(self.binary_dataset.y),
            delta=0.01)
        self.assertAlmostEqual(
            len(test_dataset.y[test_dataset.y == 1]) / len(test_dataset.y),
            len(self.binary_dataset.y[self.binary_dataset.y == 1]) / len(self.binary_dataset.y),
            delta=0.01)
        self.assertAlmostEqual(
            len(valid_dataset.y[valid_dataset.y == 1]) / len(valid_dataset.y),
            len(self.binary_dataset.y[self.binary_dataset.y == 1]) / len(self.binary_dataset.y),
            delta=0.01)

    def test_similarity_splitter_invalid_smiles(self):
        stss_splitter = SingletaskStratifiedSplitter()

        train_dataset, test_dataset = stss_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 2)
        self.assertEqual(len(test_dataset.smiles), 1)

    def test_similarity_splitter_invalid_smiles_and_nan(self):
        to_add = np.zeros(4)
        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))

        stss_splitter = SingletaskStratifiedSplitter()

        train_dataset, test_dataset = stss_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 2)
        self.assertEqual(len(test_dataset.smiles), 1)
