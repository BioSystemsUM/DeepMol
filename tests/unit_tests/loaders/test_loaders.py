import os
from unittest import TestCase

from loaders.loaders import SDFLoader


class TestLoaders(TestCase):

    def test_sdf_loader(self) -> None:
        self.data_path = os.path.join(os.path.dirname(os.path.abspath(os.curdir)), 'tests', 'data')
        dataset = os.path.join(self.data_path, "results_test.sdf")
        loader = SDFLoader(dataset)
        dataset2 = loader.create_dataset()


