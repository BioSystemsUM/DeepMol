import os
import sys
from abc import ABC, abstractmethod

from loaders.loaders import CSVLoader


class StandardizerBaseTestCase(ABC):

    def setUp(self) -> None:
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(os.curdir)), 'tests', 'data')
        dataset = os.path.join(self.data_path, "test_to_convert_to_sdf.csv")
        loader = CSVLoader(dataset,
                           mols_field='Smiles',
                           labels_fields='Class')

        self.dataset_to_test = loader.create_dataset()

    @abstractmethod
    def test_standardize(self):
        raise NotImplementedError
