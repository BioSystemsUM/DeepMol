import os
import sys
from abc import ABC, abstractmethod

from loaders.Loaders import CSVLoader


class StandardizerBaseTestCase(ABC):

    def setUp(self) -> None:
        dir_path = os.path.join(os.path.dirname(sys.path[1]), "src")
        dataset = os.path.join(dir_path, "data", "test_to_convert_to_sdf.csv")
        loader = CSVLoader(dataset,
                           mols_field='Smiles',
                           labels_fields='Class')

        self.dataset_to_test = loader.create_dataset()

    @abstractmethod
    def test_standardize(self):
        raise NotImplementedError
