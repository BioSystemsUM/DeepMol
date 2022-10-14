import os
from unittest import TestCase

from deepmol.compound_featurization.rdkit_descriptors import TwoDimensionDescriptors
from deepmol.feature_selection import BorutaAlgorithm
from deepmol.loaders.loaders import CSVLoader

from tests import TEST_DIR


class TestFeatureSelectors(TestCase):

    def setUp(self) -> None:
        dataset = os.path.join(TEST_DIR, "data", "test_to_convert_to_sdf.csv")
        loader = CSVLoader(dataset,
                           mols_field='Standardized_Smiles',
                           labels_fields='Class')

        self.mini_dataset_to_test = loader.create_dataset()

    def test_boruta_algorithm(self):

        BorutaAlgorithm(max_iter=5, n_estimators=100).select_features(self.mini_dataset_to_test)
        with self.assertRaises(Exception):
            print(self.mini_dataset_to_test.X)

    def test_boruta_algorithm_larger_dataset(self):
        import pandas as pd
        dataset = os.path.join(TEST_DIR, "data", "test_to_convert_to_sdf.csv")
        pandas_dset = pd.read_csv(dataset)
        columns = list(pandas_dset.columns[3:])
        TwoDimensionDescriptors().featurize(self.mini_dataset_to_test)
        BorutaAlgorithm(max_iter=3, n_estimators=50, support_weak=True).select_features(self.mini_dataset_to_test)

        self.assertGreater(len(columns), self.mini_dataset_to_test.X.shape[1])

