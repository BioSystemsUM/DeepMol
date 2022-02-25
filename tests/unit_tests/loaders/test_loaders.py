import os
from unittest import TestCase

from loaders.Loaders import SDFLoader


class TestLoaders(TestCase):

    def test_sdf_loader(self) -> None:
        dataset = os.path.join("../..", "data", "results_test.sdf")
        loader = SDFLoader(dataset)
        dataset2 = loader.create_dataset()


