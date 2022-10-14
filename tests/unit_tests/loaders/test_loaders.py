import os
from unittest import TestCase

from deepmol.loaders.loaders import SDFLoader
from tests import TEST_DIR

class TestLoaders(TestCase):

    def test_sdf_loader(self) -> None:
        self.data_path = os.path.join(TEST_DIR, 'data')
        dataset = os.path.join(self.data_path, "results_test.sdf")
        loader = SDFLoader(dataset)
        dataset2 = loader.create_dataset()


