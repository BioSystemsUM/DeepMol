import os
from abc import ABC, abstractmethod

from deepmol.loaders.loaders import CSVLoader

from tests import TEST_DIR
class StandardizerBaseTestCase(ABC):

    def setUp(self) -> None:
        self.data_path = os.path.join(TEST_DIR, 'data')
        dataset = os.path.join(self.data_path, "test_to_convert_to_sdf.csv")
        loader = CSVLoader(dataset,
                           mols_field='Smiles',
                           labels_fields='Class')

        self.dataset_to_test = loader.create_dataset()

    @abstractmethod
    def test_standardize(self):
        raise NotImplementedError
