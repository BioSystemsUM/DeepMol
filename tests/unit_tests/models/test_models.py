import os
from abc import ABC, abstractmethod

from deepmol.loaders import CSVLoader
from tests import TEST_DIR


class ModelsTestCase(ABC):

    def setUp(self) -> None:
        self.data_path = os.path.join(TEST_DIR, 'data')

        dataset = os.path.join(self.data_path, "balanced_mini_dataset.csv")
        loader = CSVLoader(dataset,
                           smiles_field='Smiles',
                           labels_fields=['Class'])

        self.mini_dataset_to_test = loader.create_dataset(sep=';')

        dataset = os.path.join(self.data_path, "train_dataset.csv")
        loader = CSVLoader(dataset,
                           smiles_field='mols',
                           labels_fields=['y'])

        self.larger_dataset_to_test = loader.create_dataset(sep=',')

    def tearDown(self) -> None:
        if os.path.exists('deepmol.log'):
            os.remove('deepmol.log')

    @abstractmethod
    def test_fit_predict_evaluate(self):
        raise NotImplementedError
