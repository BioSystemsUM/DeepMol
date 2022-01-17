import os
from unittest import TestCase

from compoundFeaturization.rdkitDescriptors import TwoDimensionDescriptors
from compoundFeaturization.rdkitFingerprints import MorganFingerprint
from featureSelection.baseFeatureSelector import BorutaAlgorithm
from loaders.Loaders import CSVLoader
from scalers.sklearnScalers import StandardScaler


class TestFeatureSelectors(TestCase):

    def setUp(self) -> None:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(".")))
        dataset = os.path.join(dir_path, "tests", "data", "test_to_convert_to_sdf.csv")
        loader = CSVLoader(dataset,
                           mols_field='Standardized_Smiles',
                           labels_fields='Class')

        self.mini_dataset_to_test = loader.create_dataset()

        dataset = os.path.join(dir_path, "tests", "data", "PC-3.csv")
        loader = CSVLoader(dataset,
                           mols_field='smiles',
                           labels_fields='pIC50')

        self.dataset_to_test = loader.create_dataset()

        TwoDimensionDescriptors().featurize(self.mini_dataset_to_test)
        StandardScaler().fit_transform(self.mini_dataset_to_test)
        self.assertEqual(5, self.mini_dataset_to_test.X.shape[0])

        MorganFingerprint().featurize(self.dataset_to_test)
        self.assertEqual(4294, self.dataset_to_test.X.shape[0])

    def test_boruta_algorithm(self):

        BorutaAlgorithm(max_iter=5, n_estimators=100).select_features(self.mini_dataset_to_test)
        with self.assertRaises(Exception):
            print(self.mini_dataset_to_test.X)

        BorutaAlgorithm(task="regression", max_iter=5, n_estimators=100, support_weak=True)\
            .select_features(self.dataset_to_test)
        self.assertEqual(self.dataset_to_test.X.shape[1], 1)

    def test_boruta_algorithm_larger_dataset(self):
        import pandas as pd

        dir_path = os.path.join(os.path.dirname(os.path.abspath(".")))
        dataset = os.path.join(dir_path, "tests", "data", "train_dataset.csv")

        pandas_dset = pd.read_csv(dataset)
        columns = list(pandas_dset.columns[3:])

        loader = CSVLoader(dataset,
                           features_fields=columns,
                           mols_field='mols',
                           labels_fields='y')
        dataset = loader.create_dataset()
        BorutaAlgorithm(max_iter=3, n_estimators=100, support_weak=True).select_features(dataset)

