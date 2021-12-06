import os
from unittest import TestCase

from compoundFeaturization.rdkitFingerprints import MorganFingerprint
from loaders.Loaders import CSVLoader
import pandas as pd


class TestDataset(TestCase):

    def setUp(self) -> None:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(".")))
        dataset = os.path.join(dir_path, "tests", "data", "test_to_convert_to_sdf.csv")
        loader = CSVLoader(dataset,
                           mols_field='Standardized_Smiles',
                           labels_fields='Class')

        self.dataset_to_test = loader.create_dataset()
        MorganFingerprint().featurize(self.dataset_to_test)



    def test_load_dataset_with_features(self):
        dir_path = os.path.join(os.path.dirname(os.path.abspath(".")))
        dataset = os.path.join(dir_path, "tests", "data", "train_dataset.csv")
        pandas_dset = pd.read_csv(dataset)
        columns = list(pandas_dset.columns[3:])

        loader = CSVLoader(dataset,
                           features_fields=columns,
                           mols_field='mols',
                           labels_fields='y')

        dataset = loader.create_dataset()
        self.assertEqual(len(dataset.features2keep), len(columns))



    def test_select_rows(self):
        self.dataset_to_test.select([0, 2], axis=0)

        self.assertEqual(self.dataset_to_test.X.shape[0], 2)

    def test_multiple_select_rows(self):
        self.dataset_to_test.select([0, 2, 3, 4], axis=0)
        self.assertEqual(self.dataset_to_test.X.shape[0], 4)

        self.dataset_to_test.select([2, 3], axis=0)
        self.assertEqual(self.dataset_to_test.X.shape[0], 2)

        self.dataset_to_test.select([0], axis=0)
        self.assertEqual(self.dataset_to_test.X.shape[0], 1)

    def test_select_columns(self):
        self.dataset_to_test.select([i for i in range(6)], axis=1)
        self.assertEqual(self.dataset_to_test.X.shape[1], 6)

    def test_sequential_multiselect_select_columns(self):
        self.dataset_to_test.select([i for i in range(6)], axis=1)
        self.assertEqual(self.dataset_to_test.X.shape[1], 6)

        self.dataset_to_test.select([i for i in range(3)], axis=1)
        self.assertEqual(self.dataset_to_test.X.shape[1], 3)

        self.dataset_to_test.select([i for i in range(1)], axis=1)
        self.assertEqual(self.dataset_to_test.X.shape[1], 1)

    def test_random_multiselect_select_columns(self):
        self.dataset_to_test.select([i for i in range(100)], axis=1)
        self.assertEqual(self.dataset_to_test.X.shape[1], 100)

        self.dataset_to_test.select([4, 60, 40, 20, 39], axis=1)
        self.assertEqual(self.dataset_to_test.X.shape[1], 5)

        self.dataset_to_test.select([4, 60, 40], axis=1)
        self.assertEqual(self.dataset_to_test.X.shape[1], 3)

        self.assertEqual(set(self.dataset_to_test.features2keep), {4, 60, 40})

    def test_select_features(self):
        self.dataset_to_test._select_features([i for i in range(100)])
        self.assertEqual(self.dataset_to_test.X.shape[1], 100)

        self.dataset_to_test._select_features([4, 60, 40, 20, 39])
        self.assertEqual(self.dataset_to_test.X.shape[1], 5)

    def test_remove_elements(self):
        self.dataset_to_test.remove_elements([0, 1, 3])
        self.assertEqual(self.dataset_to_test.X.shape[0], 2)

        self.dataset_to_test.remove_elements([0])
        self.assertEqual(self.dataset_to_test.X.shape[0], 1)
