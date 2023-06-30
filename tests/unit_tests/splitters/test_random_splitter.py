from unittest import TestCase

import numpy as np

from deepmol.splitters import RandomSplitter
from unit_tests.splitters.test_splitters import SplittersTestCase


class RandomSplitterTestCase(SplittersTestCase, TestCase):

    def test_split(self):
        random_splitter = RandomSplitter()

        train_dataset, test_dataset = random_splitter.train_test_split(self.mini_dataset_to_test)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 5)
        self.assertEqual(len(test_dataset.smiles), 2)

    def test_k_fold_split(self):
        random_splitter = RandomSplitter()

        folds = random_splitter.k_fold_split(self.dataset_for_k_split, k=3, seed=123)

        for train_df, valid_df in folds:
            self.assertEqual(len(train_df.y) + len(valid_df.y), len(self.dataset_for_k_split.y))

    def test_random_splitter_larger_dataset(self):
        random_splitter = RandomSplitter()

        train_dataset, test_dataset = random_splitter.train_test_split(self.dataset_to_test)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 3435)
        self.assertEqual(len(test_dataset.smiles), 859)

    def test_similarity_splitter_larger_dataset_binary_classification(self):
        random_splitter = RandomSplitter()

        train_dataset, test_dataset = random_splitter.train_test_split(self.binary_dataset)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertAlmostEqual(len(train_dataset.smiles), int(len(self.binary_dataset.smiles) * 0.8), delta=1)
        self.assertAlmostEqual(len(test_dataset.smiles), int(len(self.binary_dataset.smiles) * 0.2), delta=1)

        train_dataset, valid_dataset, test_dataset = random_splitter.train_valid_test_split(self.binary_dataset,
                                                                                            frac_train=0.8,
                                                                                            frac_valid=0.1,
                                                                                            seed=123)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertGreater(len(train_dataset.smiles), len(valid_dataset.smiles))
        self.assertAlmostEqual(len(train_dataset.smiles), int(len(self.binary_dataset.smiles) * 0.8), delta=1)
        self.assertAlmostEqual(len(valid_dataset.smiles), int(len(self.binary_dataset.smiles) * 0.1), delta=1)
        self.assertAlmostEqual(len(test_dataset.smiles), int(len(self.binary_dataset.smiles) * 0.1), delta=1)

    def test_similarity_splitter_invalid_smiles(self):
        random_splitter = RandomSplitter()

        train_dataset, test_dataset = random_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 2)
        self.assertEqual(len(test_dataset.smiles), 1)

    def test_similarity_splitter_invalid_smiles_and_nan(self):
        to_add = np.zeros(4)
        self.mini_dataset_to_test.mols = np.concatenate((self.mini_dataset_to_test.mols, to_add))
        self.mini_dataset_to_test.y = np.concatenate((self.mini_dataset_to_test.y, to_add))

        random_splitter = RandomSplitter()

        train_dataset, test_dataset = random_splitter.train_test_split(self.invalid_smiles_dataset)
        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 2)
        self.assertEqual(len(test_dataset.smiles), 1)
