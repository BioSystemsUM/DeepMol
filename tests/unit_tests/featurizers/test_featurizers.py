import sys
from abc import abstractmethod, ABC

import os

from loaders.loaders import CSVLoader


class FeaturizerTestCase(ABC):

    def setUp(self) -> None:
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(os.curdir)), 'tests', 'data')

        dataset = os.path.join(self.data_path, "test_to_convert_to_sdf.csv")
        loader = CSVLoader(dataset,
                           mols_field='Standardized_Smiles',
                           labels_fields='Class')

        self.mini_dataset_to_test = loader.create_dataset()

        dataset = os.path.join(self.data_path, "invalid_smiles_dataset.csv")
        loader = CSVLoader(dataset,
                           mols_field='Standardized_Smiles',
                           labels_fields='Class')

        self.dataset_invalid_smiles = loader.create_dataset()

        self.mol2vec_model = os.path.join(os.path.abspath(os.curdir), "compound_featurization", "mol2vec_models",
                                        "model_300dim.pkl")

    @abstractmethod
    def test_featurize(self):
        raise NotImplementedError

    @abstractmethod
    def test_featurize_with_nan(self):
        raise NotImplementedError
