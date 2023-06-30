import os
from unittest import TestCase

from deepmol.loaders import CSVLoader
from tests import TEST_DIR


class TestDataset(TestCase):

    def setUp(self):
        self.data_path = os.path.join(TEST_DIR, 'data')
        dataset = os.path.join(self.data_path, "balanced_mini_dataset.csv")
        loader = CSVLoader(dataset,
                           smiles_field='Smiles',
                           labels_fields=['Class'])

        self.small_dataset_to_test = loader.create_dataset(sep=";")
