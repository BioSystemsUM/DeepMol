from unittest import TestCase
import os

import pandas as pd
from deepmol.loaders.loaders import CSVLoader

from tests import TEST_DIR


class TestMultiprocessing(TestCase):

    def setUp(self) -> None:
        self.data_path = os.path.join(TEST_DIR, 'data')
        dataset = os.path.join(self.data_path, "balanced_mini_dataset.csv")
        loader = CSVLoader(dataset,
                           mols_field='Smiles',
                           labels_fields='Class')

        self.small_dataset_to_test = loader.create_dataset(sep=";")

        self.small_pandas_dataset = pd.read_csv(dataset, sep=";")

        dataset = os.path.join(self.data_path, "dataset_last_version2.csv")
        loader = CSVLoader(dataset,
                           mols_field='Smiles',
                           labels_fields='Class')
        self.big_dataset_to_test = loader.create_dataset(sep=";")
        self.big_pandas_dataset = pd.read_csv(dataset, sep=";")
