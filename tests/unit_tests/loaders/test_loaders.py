import os
from unittest import TestCase

from loaders.Loaders import SDFLoader


class TestLoaders(TestCase):

    def test_sdf_loader(self) -> None:
        dir_path = os.path.join(os.path.dirname(os.path.abspath(".")))
        dataset = os.path.join(dir_path, "tests", "data", "results_test.sdf")
        loader = SDFLoader(dataset)
        dataset2 = loader.create_dataset()


