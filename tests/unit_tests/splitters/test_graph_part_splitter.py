from unittest import TestCase

import numpy as np

from deepmol.splitters import RandomSplitter
from deepmol.splitters.graph_part_similarity_split import GraphPartSimilaritySplitter
from tests.unit_tests.splitters.test_splitters import SplittersTestCase


class GraphPartSplitterTestCase(SplittersTestCase, TestCase):

    def test_split(self):
        random_splitter = RandomSplitter()

        train_dataset, test_dataset = random_splitter.train_test_split(self.mini_dataset_to_test)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 5)
        self.assertEqual(len(test_dataset.smiles), 2)

    def test_k_fold_split(self):
        pass

    def test_random_splitter_larger_dataset(self):
        random_splitter = GraphPartSimilaritySplitter()

        train_dataset, test_dataset = random_splitter.train_test_split(self.dataset_to_test)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 3506)
        self.assertEqual(len(test_dataset.smiles), 653)

    def test_random_splitter_larger_dataset_valid_test(self):
        random_splitter = GraphPartSimilaritySplitter()

        train_dataset, validation_dataset, test_dataset = random_splitter.train_valid_test_split(self.dataset_to_test, frac_test=0.1, frac_valid=0.1)

        self.assertGreater(len(train_dataset.smiles), len(test_dataset.smiles))
        self.assertEqual(len(train_dataset.smiles), 3506)
        self.assertEqual(len(test_dataset.smiles), 344)
        self.assertEqual(len(validation_dataset.smiles), 309)

    def test_similarity_splitter_stratified(self):
        # Import the required libraries
        from collections import Counter

        # Modified test for stratification
        random_splitter = GraphPartSimilaritySplitter(stratified=True)

        # Perform the train-test split
        train_dataset, test_dataset = random_splitter.train_test_split(
            self.binary_dataset, frac_train=0.8
        )

        # Validate the stratification
        # Assuming `dataset_to_test` has a `labels` attribute or similar for stratification
        train_labels = [sample for sample in train_dataset.y]
        test_labels = [sample for sample in test_dataset.y]

        # Count occurrences of each label in the train and test splits
        train_label_counts = Counter(train_labels)
        test_label_counts = Counter(test_labels)

        # Check that the distribution is approximately proportional to the original dataset
        original_label_counts = Counter([sample for sample in self.binary_dataset.y])

        for label in original_label_counts:
            train_ratio = train_label_counts[label] / original_label_counts[label]
            test_ratio = test_label_counts[label] / original_label_counts[label]
            
            # Assert that the train and test ratios are approximately equal to the split fractions
            self.assertAlmostEqual(train_ratio, 0.8, delta=0.2)
            self.assertAlmostEqual(test_ratio, 0.2, delta=0.2)

