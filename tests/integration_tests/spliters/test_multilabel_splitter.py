import os
from unittest import TestCase

import numpy as np

from deepmol.loaders import CSVLoader
from deepmol.splitters.multitask_splitter import MultiTaskStratifiedSplitter
from tests.unit_tests.splitters.test_splitters import SplittersTestCase

from tests import TEST_DIR


class MultiTaskStratifierSplitterTestCase(SplittersTestCase, TestCase):

    def multi_task_splitter_tests(self, dataset, frac_train, frac_validation, frac_test):
        train_dataset, validation_dataset, test_dataset = \
            MultiTaskStratifiedSplitter().train_valid_test_split(dataset, frac_train=frac_train,
                                                                 frac_valid=frac_validation,
                                                                 frac_test=frac_test)

        self.assertAlmostEqual(len(train_dataset.smiles), len(dataset.smiles) * frac_train,
                               delta=100)
        self.assertAlmostEqual(len(validation_dataset.smiles), len(dataset.smiles) * frac_validation,
                               delta=60)
        self.assertAlmostEqual(len(test_dataset.smiles), len(dataset.smiles) * frac_test,
                               delta=60)

        num_ones_per_column_train = np.sum(train_dataset.y == 1, axis=0)
        num_ones_per_column_validation = np.sum(validation_dataset.y == 1, axis=0)
        num_ones_per_column_test = np.sum(test_dataset.y == 1, axis=0)

        total_train_fracs = 0
        total_validation_fracs = 0
        total_test_fracs = 0
        # check that the number of ones per column is similar in train, validation and test
        for i in range(len(num_ones_per_column_train)):
            total = num_ones_per_column_train[i] + num_ones_per_column_validation[i] + num_ones_per_column_test[i]
            total_train_fracs += num_ones_per_column_train[i] / total
            total_validation_fracs += num_ones_per_column_validation[i] / total
            total_test_fracs += num_ones_per_column_test[i] / total

        self.assertAlmostEqual(total_train_fracs / dataset.y.shape[1], frac_train, delta=0.1)
        self.assertAlmostEqual(total_validation_fracs / dataset.y.shape[1], frac_validation, delta=0.1)
        self.assertAlmostEqual(total_test_fracs / dataset.y.shape[1], frac_test, delta=0.1)

    def test_train_test_split(self):
        dataset = os.path.join(TEST_DIR, 'data', 'multilabel_classification_dataset.csv')
        loader = CSVLoader(dataset_path=dataset,
                           smiles_field='smiles',
                           id_field='ids',
                           labels_fields=['C00341', 'C01789', 'C00078', 'C00049', 'C00183', 'C03506', 'C00187',
                                          'C00079', 'C00047', 'C01852', 'C00407', 'C00129', 'C00235', 'C00062',
                                          'C00353', 'C00148', 'C00073', 'C00108', 'C00123', 'C00135', 'C00448',
                                          'C00082', 'C00041'],
                           mode='auto')
        # create the dataset
        dataset = loader.create_dataset(sep=',', header=0)

        train_dataset, test_dataset = \
            MultiTaskStratifiedSplitter().train_test_split(dataset, frac_train=0.8)

        self.assertAlmostEqual(len(train_dataset.smiles), len(dataset.smiles) * 0.8,
                               delta=100)
        self.assertAlmostEqual(len(test_dataset.smiles), len(dataset.smiles) * 0.2,
                               delta=60)

    def test_split(self):
        dataset = os.path.join(TEST_DIR, 'data', 'multilabel_classification_dataset.csv')
        loader = CSVLoader(dataset_path=dataset,
                           smiles_field='smiles',
                           id_field='ids',
                           labels_fields=['C00341', 'C01789', 'C00078', 'C00049', 'C00183', 'C03506', 'C00187',
                                          'C00079', 'C00047', 'C01852', 'C00407', 'C00129', 'C00235', 'C00062',
                                          'C00353', 'C00148', 'C00073', 'C00108', 'C00123', 'C00135', 'C00448',
                                          'C00082', 'C00041'],
                           mode='auto')
        # create the dataset
        dataset = loader.create_dataset(sep=',', header=0)
        self.multi_task_splitter_tests(dataset, 0.8, 0.1, 0.1)
        self.multi_task_splitter_tests(dataset, 0.8, 0.0, 0.2)
        self.multi_task_splitter_tests(dataset, 0.6, 0.3, 0.1)
        self.multi_task_splitter_tests(dataset, 0.6, 0.2, 0.2)
        self.multi_task_splitter_tests(dataset, 0.5, 0.3, 0.2)

    def test_k_fold_split(self):
        dataset = os.path.join(TEST_DIR, 'data', 'multilabel_classification_dataset.csv')
        loader = CSVLoader(dataset_path=dataset,
                           smiles_field='smiles',
                           id_field='ids',
                           labels_fields=['C00341', 'C01789', 'C00078', 'C00049', 'C00183', 'C03506', 'C00187',
                                          'C00079', 'C00047', 'C01852', 'C00407', 'C00129', 'C00235', 'C00062',
                                          'C00353', 'C00148', 'C00073', 'C00108', 'C00123', 'C00135', 'C00448',
                                          'C00082', 'C00041'],
                           mode='auto')
        # create the dataset
        dataset = loader.create_dataset(sep=',', header=0)
        datasets = \
            MultiTaskStratifiedSplitter().k_fold_split(dataset, k=5)
        for train_dataset, test_dataset in datasets:
            self.assertAlmostEqual(len(train_dataset.smiles), len(dataset.smiles) * 0.8,
                                   delta=100)
            self.assertAlmostEqual(len(test_dataset.smiles), len(dataset.smiles) * 0.2,
                                   delta=60)
            num_ones_per_column_train = np.sum(train_dataset.y == 1, axis=0)
            num_ones_per_column_test = np.sum(test_dataset.y == 1, axis=0)
            # check that the number of ones per column is similar in train, validation and test
            total_train_fracs = 0
            total_test_fracs = 0
            # check that the number of ones per column is similar in train, validation and test
            for i in range(len(num_ones_per_column_train)):
                total = num_ones_per_column_train[i] + num_ones_per_column_test[i]
                total_train_fracs += num_ones_per_column_train[i] / total
                total_test_fracs += num_ones_per_column_test[i] / total

            self.assertAlmostEqual(total_train_fracs / dataset.y.shape[1], 0.8, delta=0.1)
            self.assertAlmostEqual(total_test_fracs / dataset.y.shape[1], 0.2, delta=0.1)
