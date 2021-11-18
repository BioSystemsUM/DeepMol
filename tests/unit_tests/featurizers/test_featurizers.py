import sys
from abc import abstractmethod, ABC

import os

from loaders.Loaders import CSVLoader


class FeaturizerTestCase(ABC):

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

    @abstractmethod
    def test_featurize(self):
        raise NotImplementedError

    @abstractmethod
    def test_featurize_with_nan(self):
        raise NotImplementedError
