import logging
import os
from unittest import TestCase

import pandas as pd

from deepmol.loaders import CSVLoader
from deepmol.loggers.logger import Logger
from tests import TEST_DIR


class TestLogger(TestCase):

    def setUp(self) -> None:
        self.data_path = os.path.join(TEST_DIR, 'data')

        self.logger = Logger(file_path=os.path.join(TEST_DIR, "test.log"),
                             level=logging.DEBUG)

        dataset = os.path.join(self.data_path, "balanced_mini_dataset.csv")
        loader = CSVLoader(dataset,
                           smiles_field='Smiles',
                           labels_fields='Class')

        self.small_dataset_to_test = loader.create_dataset(sep=";")

        self.small_pandas_dataset = pd.read_csv(dataset, sep=";")

        dataset = os.path.join(self.data_path, "dataset_last_version2.csv")
        loader = CSVLoader(dataset,
                           smiles_field='Smiles',
                           labels_fields='Class')
        self.big_dataset_to_test = loader.create_dataset(sep=";")
        self.big_pandas_dataset = pd.read_csv(dataset, sep=";")

    @classmethod
    def tearDownClass(cls):
        log_file_name = os.path.join(TEST_DIR, "test.log")
        if os.path.exists(log_file_name):
            os.remove(log_file_name)


