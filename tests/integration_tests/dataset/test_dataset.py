import os
from unittest import TestCase

from deepmol.loaders import CSVLoader
from deepmol.loaders.loaders import CSVLoaderForMaskedLM
from tests import TEST_DIR


class TestDataset(TestCase):

    def setUp(self):
        self.data_path = os.path.join(TEST_DIR, 'data')
        dataset = os.path.join(self.data_path, "balanced_mini_dataset.csv")
        loader = CSVLoader(dataset,
                           smiles_field='Smiles',
                           labels_fields=['Class'])

        self.small_dataset_to_test = loader.create_dataset(sep=";")

        self.data_path = os.path.join(TEST_DIR, 'data')
        dataset = os.path.join(self.data_path, "invalid_smiles_dataset.csv")
        loader = CSVLoader(dataset,
                           smiles_field='Standardized_Smiles',
                           labels_fields=['Class'])

        self.small_dataset_to_test_with_invalid = loader.create_dataset(sep=",")

        multilabel_classification_df = os.path.join(TEST_DIR, 'data', "multilabel_classification_dataset.csv")
        loader = CSVLoader(dataset_path=multilabel_classification_df,
                           smiles_field='smiles',
                           id_field='ids',
                           labels_fields=['C00341', 'C01789', 'C00078', 'C00049', 'C00183', 'C03506', 'C00187',
                                          'C00079', 'C00047', 'C01852', 'C00407', 'C00129', 'C00235', 'C00062',
                                          'C00353', 'C00148', 'C00073', 'C00108', 'C00123', 'C00135', 'C00448',
                                          'C00082', 'C00041'],
                           mode='auto')
        # create the dataset
        self.multilabel_classification = loader.create_dataset(sep=',', header=0, nrows=100)

        multilabel_classification_df = os.path.join(TEST_DIR, 'data', "multilabel_classification_dataset.csv")
        loader = CSVLoaderForMaskedLM(dataset_path=multilabel_classification_df,
                           smiles_field='smiles',
                           id_field='ids',
                           labels_fields=['C00341', 'C01789', 'C00078', 'C00049', 'C00183', 'C03506', 'C00187',
                                          'C00079', 'C00047', 'C01852', 'C00407', 'C00129', 'C00235', 'C00062',
                                          'C00353', 'C00148', 'C00073', 'C00108', 'C00123', 'C00135', 'C00448',
                                          'C00082', 'C00041'],
                           mode='auto')
        self.dataset_for_masked_learning = loader.create_dataset(sep=',', header=0, nrows=100)
